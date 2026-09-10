"""ImageWAMActionPatchVL: action-as-patch,但把冻结的 Qwen3-4B 文本编码器换成
冻结的、图像 grounded 的 Qwen3-VL-4B-Instruct 编码器。

相对 action-as-patch 的唯一变化:FLUX.2 的 text conditioning 改由
(当前帧 t0 + instruction) 一起过 Qwen3-VL 得到,取 layers (9,18,27) 拼成 7680 维
—— 与原 Qwen3-4B 路径完全相同的布局,纯编码、无任何 reasoning。VL 全程冻结
(特征提取器),与 base 冻结的 Qwen3-4B 对齐,所以唯一变量 = 编码器本身(且它看图)。
其余(单流 action codec、batch、lr、epoch、数据)与 action-as-patch 逐项一致。
"""
from __future__ import annotations

import torch

from exploration.action_img_patch.model import ImageWAMActionPatch
from imagewam.utils.logging_config import get_logger

logger = get_logger(__name__)


class VLPromptEncoder:
    """冻结的 Qwen3-VL,作为纯 (图+文) 特征提取器。刻意不做成 nn.Module(无可训练参数),
    对齐 base 里用 SimpleNamespace 包的冻结 Qwen3-4B 文本编码器。VL prefill 核心
    (DeepStack 图像注入 + 3D M-RoPE) 摘自已验证的 v3 VLLatentReasoner。"""

    def __init__(self, vl_path, qwen_layers=(9, 18, 27), max_prompt_len=128,
                 device="cuda", torch_dtype=torch.bfloat16):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(vl_path)
        self.tok = self.processor.tokenizer
        self.tok.padding_side = "left"
        full = Qwen3VLForConditionalGeneration.from_pretrained(vl_path, torch_dtype=torch_dtype)
        self.vl = full.model.to(device).eval()      # Qwen3VLModel: visual + language_model
        self.vl.requires_grad_(False)               # 冻结:纯特征提取器
        self.image_token_id = int(full.config.image_token_id)
        self.qwen_layers = tuple(int(l) for l in qwen_layers)
        self.max_prompt_len = int(max_prompt_len)
        hidden = int(self.vl.config.text_config.hidden_size)   # 2560
        self.dim = hidden * len(self.qwen_layers)              # 7680
        self.device = device
        self.torch_dtype = torch_dtype
        logger.info("VLPromptEncoder(Qwen3-VL, FROZEN, layers=%s, dim=%d, max_prompt_len=%d)",
                    self.qwen_layers, self.dim, self.max_prompt_len)

    def _concat_layers(self, hidden_states):
        return torch.cat([hidden_states[l] for l in self.qwen_layers], dim=-1)

    def _build_inputs(self, prompts, images):
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": str(prompt)}]}]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
        enc = self.processor(text=texts, images=list(images), return_tensors="pt",
                             padding=True, truncation=True, max_length=self.max_prompt_len)
        dev = next(self.vl.parameters()).device
        return {k: v.to(dev) for k, v in enc.items()}

    def _prefill(self, enc):
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        pixel_values = enc["pixel_values"].to(self.vl.dtype)
        image_grid_thw = enc["image_grid_thw"]
        inputs_embeds = self.vl.get_input_embeddings()(input_ids)
        image_embeds, deepstack = self.vl.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.vl.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]
        position_ids, _ = self.vl.get_rope_index(
            input_ids, image_grid_thw, None, attention_mask=attention_mask)
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
        out = self.vl.language_model(
            input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=None, use_cache=False,
            cache_position=cache_position, visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack, output_hidden_states=True)
        return out, attention_mask

    @torch.no_grad()
    def encode(self, prompts, images):
        if isinstance(prompts, str):
            prompts = [prompts]
        if not isinstance(images, (list, tuple)):
            images = [images]
        if len(images) != len(prompts):
            raise ValueError(f"#images({len(images)}) != #prompts({len(prompts)})")
        enc = self._build_inputs(prompts, images)
        out, attention_mask = self._prefill(enc)
        h = self._concat_layers(out.hidden_states)      # [B, L, 7680]
        return h, attention_mask.bool()


class ImageWAMActionPatchVL(ImageWAMActionPatch):
    """action-as-patch + 冻结图像 grounded 的 Qwen3-VL 文本编码器。"""

    def attach_vl_encoder(self, enc: VLPromptEncoder) -> None:
        self.vl_encoder = enc

    @staticmethod
    def _frame_to_pil(frames: torch.Tensor):
        """[B,3,H,W] in [-1,1] -> List[PIL] (uint8 RGB)。dataset mean=std=0.5,去归一化。"""
        from PIL import Image
        x = frames.float()
        x = (x * 0.5 + 0.5).clamp(0, 1)
        x = (x * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(x[i], mode="RGB") for i in range(x.shape[0])]

    @classmethod
    def _t0_to_pil(cls, video: torch.Tensor):
        """sample['video'] [B,C,T,H,W] in [-1,1] -> List[PIL] from t0=video[:,:,0]
        (与 VAE ref-token 用的是同一帧)。"""
        return cls._frame_to_pil(video[:, :, 0])

    def _encode_flux2_text(self, sample):
        cached = sample.get("text_hidden_states")
        if cached is not None:
            th = cached.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            tm = sample.get("text_attention_mask")
            if tm is None:
                raise ValueError("cached text_hidden_states must pair with text_attention_mask")
            return th, tm.to(device=self.device, dtype=torch.bool, non_blocking=True)
        enc = getattr(self, "vl_encoder", None)
        if enc is None:
            return ImageWAMActionPatch._encode_flux2_text(self, sample)
        prompt = sample.get("instruction", sample.get("prompt", sample.get("task")))
        if prompt is None:
            raise ValueError("VL text path needs an instruction/prompt/task field.")
        video = sample.get("video")
        if not (isinstance(video, torch.Tensor) and video.ndim == 5):
            raise ValueError("VL path needs online sample['video'] [B,C,T,H,W] for the t0 frame.")
        B = int(video.shape[0])
        prompts = [prompt] * B if isinstance(prompt, str) else list(prompt)
        images = self._t0_to_pil(video)
        th, mask = enc.encode(prompts, images)
        return th.to(device=self.device, dtype=self.torch_dtype), mask.to(device=self.device, dtype=torch.bool)

    def infer_action_flux2(self, prompt, input_image, *args, **kwargs):
        prev = getattr(self, "_infer_input_image", None)
        self._infer_input_image = input_image
        try:
            return super().infer_action_flux2(prompt, input_image, *args, **kwargs)
        finally:
            self._infer_input_image = prev

    def _prepare_flux2_infer_text(self, prompt, context, context_mask):
        if context is not None or context_mask is not None:
            return super()._prepare_flux2_infer_text(prompt, context, context_mask)
        enc = getattr(self, "vl_encoder", None)
        if enc is None:
            return super()._prepare_flux2_infer_text(prompt, context, context_mask)
        img = getattr(self, "_infer_input_image", None)
        if img is None:
            raise ValueError("VL eval needs the observation frame stashed as _infer_input_image.")
        if img.ndim == 3:
            img = img.unsqueeze(0)
        images = self._frame_to_pil(img)
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        th, mask = enc.encode(prompts, images)
        return th.to(device=self.device, dtype=self.torch_dtype), mask.to(device=self.device, dtype=torch.bool)
