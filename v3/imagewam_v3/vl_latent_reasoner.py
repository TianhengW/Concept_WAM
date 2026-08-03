"""VLLatentReasoner: Qwen3-VL as an in-graph, image-grounded reasoning generator
for FLUX.2 conditioning.

This is the v3 vision-language counterpart of the text-only ``LatentReasoner``
(see ``v2_1/imagewam_v2_1/latent_reasoner.py``). The text version reasons from the
instruction alone; here the initial observation frame ``t0`` is fed *into* the LLM
through Qwen3-VL's vision tower, so the reasoning trajectory is grounded in the
actual scene before it produces the conditioning representations for the MoT.

Per sample it produces, exactly like the text version and with the identical output
layout the FLUX.2 ``txt_in`` expects:

  prompt_hidden [B, L, d_out]  -- layers (9, 18, 27) concat over the (image+text)
                                  prompt, d_out = hidden(2560) * 3 = 7680;
  reason_hidden [B, K, d_out]  -- K reasoning representations appended after the
                                  prompt, produced by one of two modes:

    mode="thought":     K learnable thought embeddings appended to the multimodal
                        prompt, single differentiable forward.
    mode="latent_loop": Coconut-style continuous latent reasoning. After the
                        multimodal prefill (KV cache kept), the loop runs K steps;
                        each step feeds the previous step's last-layer hidden state
                        back as the next input embedding through a feedback
                        projector. No token is ever sampled, so the whole trajectory
                        stays differentiable.

The language model is fine-tuned with **full parameters** (no LoRA, matching v2 Arm A);
the vision tower is frozen by default (``freeze_vision``). Trainable = the whole LLM
decoder, the thought / begin-of-thought embeddings, and the feedback projector.
Checkpoints store only the trainable subset (so the frozen vision tower is not saved).

Qwen3-VL specifics handled here (vs. the plain-text version):
  * DeepStack: ``get_image_features`` returns both the image embeds (scattered into
    the input embeds at ``<image_pad>`` positions) and per-layer ``deepstack`` embeds
    that the text model injects at several decoder layers. Both are passed through
    during the prefill; the reasoning steps carry no visual tokens.
  * 3D M-RoPE: the prefill uses ``get_rope_index`` (returns [3, B, L] positions and
    per-sample ``rope_deltas``). Each reasoning step is a text-like continuation, so
    its position on all three axes is ``(prompt_len + step) + rope_delta`` -- the same
    formula Qwen3-VL uses during decoding.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from imagewam.utils.logging_config import get_logger

logger = get_logger(__name__)


class FeedbackProjector(nn.Module):
    """Map a last-layer hidden state back into input-embedding space.

    RMS-normalize (gain initialized to the embedding-table RMS so the output starts
    at embedding magnitude), then an identity-initialized linear map. Identical to the
    text LatentReasoner's projector.
    """

    def __init__(self, hidden_size: int, embed_rms: float):
        super().__init__()
        self.gain = nn.Parameter(torch.full((hidden_size,), float(embed_rms)))
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(hidden_size))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        x = x * self.gain.float()
        return self.proj(x.to(self.proj.weight.dtype))


class VLLatentReasoner(nn.Module):
    def __init__(
        self,
        vl_path: str,
        d_out: int = 7680,
        mode: str = "latent_loop",
        num_latent: int = 16,
        qwen_layers: Sequence[int] = (9, 18, 27),
        system_prompt: Optional[str] = None,
        max_prompt_len: int = 256,
        freeze_vision: bool = True,
        gradient_checkpointing: bool = False,
        condition_tokens: str = "reasoning",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if mode not in {"thought", "latent_loop"}:
            raise ValueError(f"Unsupported VLLatentReasoner mode: {mode!r}")
        if condition_tokens not in {"reasoning", "prompt"}:
            raise ValueError(f"Unsupported condition_tokens: {condition_tokens!r}")
        self.mode = mode
        # What is handed to the MoT txt branch:
        #   "reasoning" (Plan A, default): only the K reasoning representations -- a
        #       clean bottleneck (t0 semantics are compressed into K latents by the
        #       coconut loop; t0 pixels still enter MoT via the VAE ref-token path).
        #   "prompt": the full [prompt (incl. image tokens) || reasoning] sequence.
        self.condition_tokens = condition_tokens
        self.num_latent = int(num_latent)
        self.qwen_layers = tuple(int(l) for l in qwen_layers)
        self.d_out = int(d_out)
        self.system_prompt = system_prompt
        self.max_prompt_len = int(max_prompt_len)
        self.dim = self.d_out

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(vl_path)
        self.tok = self.processor.tokenizer
        # Left padding => every sample's last physical position is a real token, which
        # keeps the KV-cache continuation for the reasoning loop simple and correct.
        self.tok.padding_side = "left"

        full = Qwen3VLForConditionalGeneration.from_pretrained(vl_path, torch_dtype=torch_dtype)
        # Keep the Qwen3VLModel (vision tower + text decoder); drop the lm_head.
        # ``self.vl`` exposes get_image_features / get_placeholder_mask / get_rope_index
        # and the ``visual`` / ``language_model`` submodules.
        self.vl = full.model
        self.image_token_id = int(full.config.image_token_id)
        self.vl.requires_grad_(False)

        hidden_size = int(self.vl.config.text_config.hidden_size)
        if self.d_out != hidden_size * len(self.qwen_layers):
            raise ValueError(
                f"d_out={self.d_out} must equal hidden_size*len(qwen_layers)="
                f"{hidden_size * len(self.qwen_layers)}"
            )

        embed_weight = self.vl.get_input_embeddings().weight
        embed_rms = float(embed_weight.detach().float().pow(2).mean().sqrt())
        embed_std = float(embed_weight.detach().float().std())

        if self.mode == "thought":
            self.thought_embeddings = nn.Parameter(
                torch.randn(self.num_latent, hidden_size) * embed_std
            )
        else:
            self.bot_embedding = nn.Parameter(torch.randn(1, hidden_size) * embed_std)
            self.feedback = FeedbackProjector(hidden_size, embed_rms=embed_rms)

        self.freeze_vision = bool(freeze_vision)
        # Full fine-tune of the LLM; grad checkpointing only usable outside the
        # latent_loop (KV cache is required there and is incompatible with it).
        if gradient_checkpointing and self.mode == "thought":
            self.vl.language_model.gradient_checkpointing_enable()
        elif gradient_checkpointing:
            logger.warning(
                "gradient_checkpointing ignored in mode=latent_loop (needs KV cache)."
            )
        self.to(torch_dtype)
        self.set_trainable()
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "VLLatentReasoner(mode=%s, num_latent=%d, full-param LLM, freeze_vision=%s): "
            "%.1fM trainable params",
            self.mode,
            self.num_latent,
            self.freeze_vision,
            n_trainable / 1e6,
        )

    # ------------------------------------------------------------------ trainable
    def set_trainable(self) -> None:
        """Full-param LLM + thought/BoT embeddings + projector; vision per freeze_vision."""
        self.requires_grad_(False)
        self.vl.language_model.requires_grad_(True)
        if not self.freeze_vision:
            self.vl.visual.requires_grad_(True)
        if self.mode == "thought":
            self.thought_embeddings.requires_grad = True
        else:
            self.bot_embedding.requires_grad = True
            self.feedback.requires_grad_(True)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {
            k: v.detach().cpu()
            for k, v in self.state_dict().items()
            if k in trainable_names
        }

    # ------------------------------------------------------------------ helpers
    def _concat_layers(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat([hidden_states[l] for l in self.qwen_layers], dim=-1)

    def _build_inputs(self, prompts, images, device):
        texts = []
        for prompt in prompts:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": str(self.system_prompt)})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": str(prompt)},
                    ],
                }
            )
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
        enc = self.processor(
            text=texts,
            images=list(images),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_len,
        )
        return {k: v.to(device) for k, v in enc.items()}

    def _prefill(self, enc):
        """Run the multimodal prompt forward. Returns (out, rope_deltas, attention_mask)."""
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        pixel_values = enc["pixel_values"].to(self.vl.dtype)
        image_grid_thw = enc["image_grid_thw"]

        inputs_embeds = self.vl.get_input_embeddings()(input_ids)
        image_embeds, deepstack = self.vl.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.vl.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]

        position_ids, rope_deltas = self.vl.get_rope_index(
            input_ids, image_grid_thw, None, attention_mask=attention_mask
        )
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
        out = self.vl.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=True,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack,
            output_hidden_states=True,
        )
        return out, rope_deltas, attention_mask

    def _step_position_ids(self, cache_index, rope_deltas, B, device):
        """3D M-RoPE position for one continuation token at absolute ``cache_index``.

        Mirrors Qwen3-VL decoding: pos = cache_index + rope_delta, equal on all 3 axes.
        """
        pos = (torch.tensor([cache_index], device=device) + rope_deltas)  # [B, 1]
        return pos.view(1, B, 1).expand(3, B, 1)

    # ------------------------------------------------------------------ modes
    def _forward_thought(self, enc):
        out, rope_deltas, attention_mask = self._prefill(enc)
        prompt_hidden = self._concat_layers(out.hidden_states)
        B, L = attention_mask.shape
        K = self.num_latent
        past = out.past_key_values

        thought = self.thought_embeddings.to(prompt_hidden.dtype).unsqueeze(0).expand(B, -1, -1)
        step_mask = torch.cat([attention_mask, attention_mask.new_ones(B, K)], dim=1)
        # positions for the K thoughts continue after the prompt on all 3 axes.
        pos = (torch.arange(L, L + K, device=attention_mask.device).view(1, K) + rope_deltas)  # [B,K]
        position_ids = pos.unsqueeze(0).expand(3, B, K)
        cache_position = torch.arange(L, L + K, device=attention_mask.device)
        out2 = self.vl.language_model(
            input_ids=None,
            inputs_embeds=thought,
            attention_mask=step_mask,
            position_ids=position_ids,
            past_key_values=past,
            use_cache=True,
            cache_position=cache_position,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            output_hidden_states=True,
        )
        reason_hidden = self._concat_layers(out2.hidden_states)[:, -K:]
        return prompt_hidden, reason_hidden

    def _forward_latent_loop(self, enc):
        out, rope_deltas, attention_mask = self._prefill(enc)
        prompt_hidden = self._concat_layers(out.hidden_states)
        B, L = attention_mask.shape
        past = out.past_key_values

        step_mask = attention_mask
        embeds = self.bot_embedding.to(prompt_hidden.dtype).unsqueeze(0).expand(B, 1, -1)
        reason_steps = []
        for step in range(self.num_latent):
            step_mask = torch.cat([step_mask, step_mask.new_ones(B, 1)], dim=1)
            cache_index = L + step
            position_ids = self._step_position_ids(cache_index, rope_deltas, B, attention_mask.device)
            cache_position = torch.tensor([cache_index], device=attention_mask.device)
            step_out = self.vl.language_model(
                input_ids=None,
                inputs_embeds=embeds,
                attention_mask=step_mask,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
                cache_position=cache_position,
                visual_pos_masks=None,
                deepstack_visual_embeds=None,
                output_hidden_states=True,
            )
            past = step_out.past_key_values
            reason_steps.append(self._concat_layers(step_out.hidden_states)[:, -1])
            if step + 1 < self.num_latent:
                embeds = self.feedback(step_out.hidden_states[-1][:, -1]).unsqueeze(1)
        reason_hidden = torch.stack(reason_steps, dim=1)
        return prompt_hidden, reason_hidden

    # ------------------------------------------------------------------ forward
    def forward(self, prompts, images, device=None, dtype=None):
        """Encode (t0 image + instruction) + reasoning.

        Args:
            prompts: str or list[str] of instructions.
            images:  a single PIL image or list of PIL images (the t0 frame), one per prompt.
        Returns:
            (text_hidden, mask) for the MoT txt branch. Shape depends on condition_tokens:
              "reasoning" (default) -> ([B, K, d_out], [B, K] bool, all True)
              "prompt"              -> ([B, L+K, d_out], [B, L+K] bool)
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if not isinstance(images, (list, tuple)):
            images = [images]
        if len(images) != len(prompts):
            raise ValueError(f"#images ({len(images)}) must match #prompts ({len(prompts)})")
        target_device = device if device is not None else next(self.parameters()).device
        enc = self._build_inputs(prompts, images, next(self.parameters()).device)

        if self.mode == "thought":
            prompt_hidden, reason_hidden = self._forward_thought(enc)
        else:
            prompt_hidden, reason_hidden = self._forward_latent_loop(enc)

        B, K = reason_hidden.shape[0], reason_hidden.shape[1]
        if self.condition_tokens == "reasoning":
            # Plan A: only the K reasoning representations condition the MoT.
            text_hidden = reason_hidden
            mask = reason_hidden.new_ones(B, K, dtype=torch.bool)
        else:
            attention_mask = enc["attention_mask"]
            text_hidden = torch.cat([prompt_hidden, reason_hidden], dim=1)
            mask = torch.cat(
                [attention_mask.bool(), attention_mask.new_ones(B, K).bool()], dim=1
            )
        if dtype is not None:
            text_hidden = text_hidden.to(dtype)
        return text_hidden.to(target_device), mask.to(target_device)
