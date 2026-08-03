"""QwenSlotGenerator: turn a text instruction into K conditioning embeddings.

Idea (Arm A, implicit reasoning, single forward):
  - Take the inner Qwen3 model (Qwen3Model, i.e. the decoder stack without the
    LM head).
  - Tokenize the instruction with the chat template (thinking disabled) and
    LEFT-pad it, so real tokens sit at the right end of the padded block.
  - Append K learnable "query" embeddings after the prompt tokens. Because Qwen
    is causal, each query token can attend to the whole (unmasked) prompt.
  - Run one forward pass and read the query positions' hidden states from a few
    layers (default [9, 18, 27], matching FLUX.2's Qwen3 text encoder), then
    concatenate those layers on the feature axis -> [B, K, num_layers * hidden].
    For Qwen3-4B (hidden=2560) and 3 layers this is 7680, which is exactly the
    context dim the FLUX.2 video expert already expects.

The forward is fully differentiable and NOT wrapped in no_grad, so the downstream
image/action loss trains both the K query embeddings and (for full fine-tuning)
all of Qwen's weights.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from einops import rearrange


class QwenSlotGenerator(nn.Module):
    def __init__(
        self,
        qwen_base: nn.Module,
        tokenizer,
        *,
        K: int = 16,
        output_layers: Sequence[int] = (9, 18, 27),
        max_length: int = 512,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        # `qwen_base` is the inner Qwen3Model (no lm_head). Registered as a
        # submodule so its parameters are part of this module's parameter tree.
        self.qwen = qwen_base
        # tokenizer is a plain attribute (not an nn.Module, no parameters).
        self.tokenizer = tokenizer
        self.K = int(K)
        self.output_layers = tuple(int(x) for x in output_layers)
        self.max_length = int(max_length)

        hidden = int(self.qwen.config.hidden_size)
        self.hidden = hidden
        self.out_dim = hidden * len(self.output_layers)

        # Learnable query embeddings live in Qwen's input-embedding space. Init
        # them at the scale of real token embeddings so the first forward stays
        # in-distribution for the frozen-at-start transformer.
        emb = self.qwen.get_input_embeddings().weight
        std = float(emb.detach().float().std().clamp(min=1e-4).item())
        query = torch.randn(self.K, hidden) * std
        self.query = nn.Parameter(query.to(dtype=emb.dtype))

        # Qwen must not use a KV cache when we run it with gradient checkpointing.
        self.qwen.config.use_cache = False
        if gradient_checkpointing:
            self.qwen.gradient_checkpointing_enable()

    def _tokenize(self, prompts: Sequence[str]):
        # Left padding: [pad ... pad, prompt tokens]. The K query tokens we append
        # afterwards then sit immediately after the real prompt tokens.
        self.tokenizer.padding_side = "left"
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": str(prompt)}]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            texts.append(text)
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return enc["input_ids"], enc["attention_mask"]

    def forward(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """prompts: list[str] of length B.
        returns (slots [B, K, out_dim], mask [B, K] bool)."""
        device = self.query.device
        input_ids, attn = self._tokenize(prompts)
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        batch_size = int(input_ids.shape[0])

        embed = self.qwen.get_input_embeddings()
        tok_embeds = embed(input_ids)  # [B, L, H] — carries grad via embed weight
        query = self.query.unsqueeze(0).expand(batch_size, -1, -1).to(tok_embeds.dtype)
        inputs_embeds = torch.cat([tok_embeds, query], dim=1)  # [B, L+K, H]

        query_mask = torch.ones(batch_size, self.K, dtype=attn.dtype, device=device)
        attn_full = torch.cat([attn, query_mask], dim=1)  # [B, L+K]

        # Position ids for left padding: count only real (unmasked) tokens.
        position_ids = attn_full.long().cumsum(dim=-1) - 1
        position_ids = position_ids.clamp(min=0)

        out = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_full,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = out.hidden_states  # tuple, length num_layers + 1
        # last K positions == the query tokens
        picks = [hidden_states[k][:, -self.K:, :] for k in self.output_layers]
        stacked = torch.stack(picks, dim=1)  # [B, C, K, H]
        slots = rearrange(stacked, "b c k h -> b k (c h)")  # [B, K, C*H]
        mask = torch.ones(batch_size, self.K, dtype=torch.bool, device=device)
        return slots, mask


if __name__ == "__main__":
    # Tiny shape/grad self-test with a stub that mimics the parts we use.
    class _Cfg:
        hidden_size = 8
        use_cache = True

    class _StubQwen(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Cfg()
            self.emb = nn.Embedding(32, 8)
            self.proj = nn.Linear(8, 8)

        def get_input_embeddings(self):
            return self.emb

        def gradient_checkpointing_enable(self):
            pass

        def forward(self, inputs_embeds=None, attention_mask=None, position_ids=None,
                    output_hidden_states=False, use_cache=False):
            h = self.proj(inputs_embeds)
            from types import SimpleNamespace
            return SimpleNamespace(hidden_states=[inputs_embeds, h, h * 2, h * 3])

    class _Tok:
        padding_side = "right"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False):
            return messages[0]["content"]

        def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None):
            ids = torch.randint(0, 32, (len(texts), max_length))
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    gen = QwenSlotGenerator(_StubQwen(), _Tok(), K=4, output_layers=(1, 2, 3), max_length=6,
                            gradient_checkpointing=False)
    s, m = gen(["put mug on shelf", "open the drawer"])
    assert s.shape == (2, 4, 24), s.shape
    assert m.shape == (2, 4) and m.dtype == torch.bool
    s.pow(2).mean().backward()
    assert gen.query.grad is not None
    print(f"[ok] slots={tuple(s.shape)} out_dim={gen.out_dim} query.grad=ok")
