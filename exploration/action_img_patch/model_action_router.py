"""ImageWAMActionRouter: patch-routed action readout (MoE-style, exploration).

Idea (2026-08-22, advisor meeting): the fixed action codec of action-as-patch
gives the model tokens whose SEMANTICS it never observes -- the tiling is
arbitrary. If action-as-patch works because the video model already contains
action-relevant information, then actions should be decodable directly from
IMAGE PATCH representations. Borrowing the MoE mechanism:

    image patches = experts, action chunk steps = routed queries.

Design (per discussion -- the action side NEVER enters FLUX):

  * The FLUX.2 forward is the PURE video model: sequence [txt | ref | noisy
    target], stock mask, video flow-matching loss -- bit-identical to the
    video-only baseline. No action tokens, no extra sequence positions.
  * A router sits OUTSIDE the transformer as a readout head. The "action
    representation" entering the router is a FIXED one-hot chunk-step code
    (H=16 steps; not trainable, per design). The trainable router maps it to a
    query, scores all noisy-target patch hiddens of the final block, selects
    the top-k patches (fixed k, MoE style), softmax-weights the selected
    scores, and aggregates their VALUE projections. That aggregated PATCH
    representation CONDITIONS the action head.

  * ANTI-COLLAPSE (2026-08-24): the first run's router collapsed
    (route_entropy 1.45->0.27, route_coverage 0.34->0.06) -- it memorised the
    training actions from a handful of patches: train loss_action ~0.001 yet
    eval SR only 54.9 (< AP 88.0, < base 78.4). Two fixes added here:
      - Switch-style load-balancing aux loss over the full patch softmax
        (importance x load) to spread which patches get selected across the
        batch, plus a top-k weight ENTROPY bonus so each query's aggregation
        stays soft. Both are opt-in via `router.lambda_balance` /
        `router.lambda_entropy`.
      - The deterministic MSE readout head is replaced by a SMALL FLOW-MATCHING
        velocity head conditioned on the routed patch representation. Instead of
        regressing the action (mode-averaging + easy to overfit), we denoise it:
        reuse the same continuous flow-matching scheduler, sample an INDEPENDENT
        action timestep, predict the velocity of a noised action conditioned on
        (routed rep, timestep). This models the (multimodal) action
        distribution and is self-regularising, mirroring why AP -- which
        denoises the action jointly -- generalises where the MSE readout did not.

  * Losses: video velocity MSE exactly as the baseline; action loss is
    flow-matching velocity MSE in the normalized action space, pad-masked,
    weighted by lambda_action; plus lambda_balance * load_balance and
    lambda_entropy * (-entropy). Ground-truth actions enter ONLY through the
    loss, never as model input.
  * Trainables: the FLUX.2 backbone (full fine-tune, protocol-aligned with
    the AP line) + the router. All new modules hang under `self.mot`
    (mot.action_router) so the trainer's dit-only optimizer collection and
    the checkpoint "mot" payload cover them automatically (lesson from the
    concept_bottleneck omission bug).

Inference: standard 20-step video denoising (identical to the video-only
model); then a short action-space ODE (action_flow_steps) decodes the chunk
from the FINAL step's routed patch representation.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from exploration.action_img_patch.model import FLUX2_TOKEN_DIM, ImageWAMActionPatch
from imagewam.utils.logging_config import get_logger

logger = get_logger(__name__)

ROUTER_DIM = 512
ROUTER_VALUE_DIM = 1024
ROUTER_TIME_DIM = 256
ACTION_CHUNK_MAX_HORIZON = 16
DEFAULT_ACTION_FLOW_STEPS = 10


class ImageWAMActionRouter(ImageWAMActionPatch):
    """Action readout by top-k routing over image patch representations,
    decoded with a small flow-matching head (+ anti-collapse regularisers)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router_topk = 8
        self.router_lambda_balance = 1.0e-2
        self.router_lambda_entropy = 1.0e-2
        self.action_flow_steps = DEFAULT_ACTION_FLOW_STEPS
        hidden = int(self.video_expert.transformer.hidden_size)
        router = nn.ModuleDict(
            {
                # Fixed one-hot step code -> query (the Linear IS the router's
                # trainable query table; the code itself is not trainable).
                "q_proj": nn.Linear(ACTION_CHUNK_MAX_HORIZON, ROUTER_DIM, bias=False),
                "k_proj": nn.Linear(hidden, ROUTER_DIM, bias=False),
                "v_proj": nn.Linear(hidden, ROUTER_VALUE_DIM, bias=False),
                # Small flow-matching action head: predicts the velocity of a
                # noised action conditioned on (routed patch rep, timestep).
                "flow_in": nn.Linear(self.action_patch_dim, ROUTER_VALUE_DIM),
                "flow_t": nn.Sequential(
                    nn.Linear(ROUTER_TIME_DIM, ROUTER_VALUE_DIM),
                    nn.SiLU(),
                    nn.Linear(ROUTER_VALUE_DIM, ROUTER_VALUE_DIM),
                ),
                "flow_head": nn.Sequential(
                    nn.LayerNorm(ROUTER_VALUE_DIM),
                    nn.Linear(ROUTER_VALUE_DIM, ROUTER_VALUE_DIM),
                    nn.SiLU(),
                    nn.Linear(ROUTER_VALUE_DIM, self.action_patch_dim),
                ),
            }
        ).to(device=self.device, dtype=self.torch_dtype)
        # Register under mot: the trainer optimizes model.dit(=mot).parameters()
        # and checkpoints save mot.state_dict() -- anything outside is silently
        # neither trained nor saved.
        self.mot.action_router = router
        self.mot.register_buffer(
            "action_step_code",
            torch.eye(ACTION_CHUNK_MAX_HORIZON, device=self.device, dtype=self.torch_dtype),
        )
        n_params = sum(p.numel() for p in router.parameters())
        logger.info(
            "ImageWAMActionRouter: hidden=%d router_dim=%d value_dim=%d topk=%d flow_steps=%d "
            "| lambda_balance=%.1e lambda_entropy=%.1e | +%.2fM trainable params "
            "(router+flow head under mot; step codes fixed one-hot; NO action tokens in the FLUX sequence)",
            hidden, ROUTER_DIM, ROUTER_VALUE_DIM, self.router_topk, self.action_flow_steps,
            self.router_lambda_balance, self.router_lambda_entropy, n_params / 1e6,
        )

    def configure_router(
        self,
        topk: Optional[int] = None,
        lambda_balance: Optional[float] = None,
        lambda_entropy: Optional[float] = None,
        action_flow_steps: Optional[int] = None,
    ) -> None:
        if topk is not None:
            self.router_topk = int(topk)
        if lambda_balance is not None:
            self.router_lambda_balance = float(lambda_balance)
        if lambda_entropy is not None:
            self.router_lambda_entropy = float(lambda_entropy)
        if action_flow_steps is not None:
            self.action_flow_steps = int(action_flow_steps)
        logger.info(
            "ImageWAMActionRouter configured: topk=%d lambda_balance=%.1e lambda_entropy=%.1e flow_steps=%d",
            self.router_topk, self.router_lambda_balance, self.router_lambda_entropy, self.action_flow_steps,
        )

    # ---------------------------------------------------------------- routing
    def _route(self, img_hidden: torch.Tensor, target_len: int, horizon: int):
        """img_hidden: final-block img-stream hiddens [B, (cond+)S, hid].

        Returns (routed [B,H,dv], metrics dict, aux dict with 'balance' and
        'entropy' scalar losses). Routing depends only on the fixed step query
        and the patch hiddens -- NOT on the (noised) action -- so it is computed
        once and shared across all flow-matching steps.
        """
        if horizon > ACTION_CHUNK_MAX_HORIZON:
            raise ValueError(f"horizon {horizon} > max {ACTION_CHUNK_MAX_HORIZON}")
        router = self.mot.action_router
        p_hidden = img_hidden[:, img_hidden.shape[1] - target_len :]    # patch "experts" [B,S,hid]
        batch_size = int(p_hidden.shape[0])
        n_patch = int(p_hidden.shape[1])

        q = router["q_proj"](self.mot.action_step_code[:horizon])       # [H,d]
        q = q[None].expand(batch_size, -1, -1)                          # [B,H,d]
        k = router["k_proj"](p_hidden)                                  # [B,S,d]
        scores = (q.float() @ k.float().transpose(1, 2)) / (ROUTER_DIM ** 0.5)  # [B,H,S]
        full_probs = torch.softmax(scores, dim=-1)                     # [B,H,S]

        topk = min(self.router_topk, n_patch)
        top_scores, top_idx = scores.topk(topk, dim=-1)                # [B,H,K]
        weights = torch.softmax(top_scores, dim=-1)                    # [B,H,K]

        v = router["v_proj"](p_hidden)                                 # [B,S,dv]
        idx = top_idx.reshape(batch_size, -1)                          # [B,H*K]
        v_sel = torch.gather(
            v, 1, idx[..., None].expand(-1, -1, v.shape[-1])
        ).reshape(batch_size, -1, topk, v.shape[-1])                   # [B,H,K,dv]
        routed = (weights[..., None].to(v_sel.dtype) * v_sel).sum(dim=-2)  # [B,H,dv]

        # Switch-style load-balancing: importance carries the gradient (soft
        # gate mass per patch), load is a detached selection count. Minimising
        # importance*load pushes probability mass off over-selected patches.
        importance = full_probs.mean(dim=(0, 1))                       # [S] differentiable
        with torch.no_grad():
            sel_mask = torch.zeros_like(scores)
            sel_mask.scatter_(-1, top_idx, 1.0)
            load = sel_mask.mean(dim=(0, 1))                           # [S] detached
        balance_loss = (importance * load).sum() * float(n_patch)

        # Entropy bonus on the top-k aggregation weights (return as a loss to be
        # MINIMISED -> negative entropy; smaller loss == higher entropy).
        entropy = -(weights * weights.clamp_min(1e-9).log()).sum(-1).mean()
        entropy_loss = -entropy

        with torch.no_grad():
            n_unique = sum(int(torch.unique(top_idx[b]).numel()) for b in range(batch_size))
            coverage = n_unique / float(max(batch_size, 1) * n_patch)
        metrics = {
            "route_entropy": float(entropy.detach().item()),
            "route_coverage": float(coverage),
            "route_balance": float(balance_loss.detach().item()),
        }
        aux = {"balance": balance_loss, "entropy": entropy_loss}
        return routed, metrics, aux

    # ----------------------------------------------------------- flow head
    @staticmethod
    def _timestep_embed(sigma: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal embedding of a unit-scaled sigma in [0,1] -> [..., dim]."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=sigma.device, dtype=torch.float32)
            / max(half, 1)
        )
        args = sigma.float().reshape(-1, 1) * freqs[None] * 1000.0     # [B,half]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)    # [B,2*half]
        if emb.shape[-1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[-1]))
        return emb

    def _flow_velocity(
        self, noisy_action: torch.Tensor, sigma: torch.Tensor, routed: torch.Tensor
    ) -> torch.Tensor:
        """Predict flow-matching velocity for the noised action chunk.

        noisy_action [B,H,A], sigma [B] in [0,1], routed [B,H,dv] -> v [B,H,A]."""
        router = self.mot.action_router
        t_emb = self._timestep_embed(sigma, ROUTER_TIME_DIM).to(routed.dtype)   # [B,time]
        t_emb = router["flow_t"](t_emb)                                         # [B,dv]
        h = router["flow_in"](noisy_action.to(routed.dtype)) + routed + t_emb[:, None, :]
        return router["flow_head"](h).float()                                   # [B,H,A]

    # ---------------------------------------------------------- training loss
    def _training_loss_flux2(self, sample, tiled: bool = False):
        inputs = self.build_inputs_flux2(sample, tiled=tiled)
        target_latent = inputs["target_latent"]  # [B, S, 128]
        action = inputs["action"].float()        # [B, H, A]
        batch_size = int(target_latent.shape[0])
        target_len = int(target_latent.shape[1])
        horizon = int(action.shape[1])

        timestep = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=target_latent.dtype,
        )
        noise_img = torch.randn_like(target_latent)
        noisy_img = self.train_video_scheduler.add_noise(target_latent, noise_img, timestep)
        target_v_img = self.train_video_scheduler.training_target(target_latent, noise_img, timestep)

        # PURE video forward: no action tokens anywhere in the sequence.
        video_pre = self.video_expert.pre_dit(
            x=noisy_img,
            timestep=self._scheduler_timestep_to_unit(timestep, self.train_video_scheduler),
            context=inputs["text_hidden_states"],
            context_mask=inputs["text_attention_mask"],
            ref_image_hidden_states=inputs["ref_image_latents"],
            target_img_ids=inputs["target_img_ids"],
            ref_img_ids=inputs["ref_img_ids"],
        )
        attention_mask = self._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=int(video_pre["target_len"]),
            cond_len=int(video_pre["cond_len"]),
            action_len=0,
            device=noisy_img.device,
            text_attention_mask=video_pre["text_mask"],
        )
        tokens_out = self._forward_flux2_video_only(video_pre, attention_mask)

        # Video loss: identical to the baseline (velocity MSE, flow weight).
        pred_img = self.video_expert.post_dit(tokens_out, video_pre)[:, :target_len]
        weight = self.train_video_scheduler.training_weight(timestep)
        video_loss_per_sample = (
            F.mse_loss(pred_img.float(), target_v_img.float(), reduction="none").flatten(1).mean(dim=1)
        )
        video_weight = weight.to(video_loss_per_sample.device, dtype=video_loss_per_sample.dtype)
        loss_video = (video_loss_per_sample * video_weight).mean()

        # Action loss: FLOW MATCHING from the routed patch representation.
        # Independent action timestep; predict velocity of the noised action.
        routed, route_metrics, route_aux = self._route(tokens_out["img"], target_len, horizon)
        t_act = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype,
        )
        noise_act = torch.randn_like(action)
        noisy_act = self.train_video_scheduler.add_noise(action, noise_act, t_act)
        target_v_act = self.train_video_scheduler.training_target(action, noise_act, t_act)
        sigma_act = t_act.float() / float(self.train_video_scheduler.num_train_timesteps)
        pred_v_act = self._flow_velocity(noisy_act, sigma_act, routed)
        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_v_act,
            target_action=target_v_act.float(),
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=None,
        )
        loss_action = action_loss_per_sample.mean()

        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action
            + self.router_lambda_balance * route_aux["balance"]
            + self.router_lambda_entropy * route_aux["entropy"]
        )
        metrics = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_balance": self.router_lambda_balance * float(route_aux["balance"].detach().item()),
            "loss_entropy": self.router_lambda_entropy * float(route_aux["entropy"].detach().item()),
        }
        metrics.update(route_metrics)
        return loss_total, metrics

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def infer_action_flux2(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        """Standard video denoising (identical to the video-only model); then a
        short action-space flow-matching ODE decodes the action chunk from the
        FINAL step's routed patch representations."""
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")

        text_hidden, text_mask = self._prepare_flux2_infer_text(prompt, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="action-router inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(input_image, time_value=10.0)
        batch_size = int(ref_tokens.shape[0])
        target_len = int(ref_tokens.shape[1])
        horizon = int(action_horizon)

        target_img_ids = ref_img_ids.clone()
        target_img_ids[..., 0] = 0.0

        generator = None
        if seed is not None:
            generator = torch.Generator(device=rand_device)
            generator.manual_seed(int(seed))
        latents = torch.randn(
            (batch_size, target_len, FLUX2_TOKEN_DIM),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        scheduler = self.infer_video_scheduler
        timesteps, deltas = scheduler.build_inference_schedule(
            num_inference_steps=int(num_inference_steps),
            device=self.device,
            dtype=self.torch_dtype,
            shift_override=sigma_shift,
        )

        attention_mask = None
        tokens_out = None
        for step_t, step_delta in zip(timesteps, deltas):
            timestep = step_t.expand(batch_size).to(device=self.device, dtype=latents.dtype)
            video_pre = self.video_expert.pre_dit(
                x=latents,
                timestep=self._scheduler_timestep_to_unit(timestep, scheduler),
                context=text_hidden,
                context_mask=text_mask,
                ref_image_hidden_states=ref_tokens,
                target_img_ids=target_img_ids,
                ref_img_ids=ref_img_ids,
            )
            if attention_mask is None:
                attention_mask = self._build_mot_attention_mask_flux2(
                    batch_size=batch_size,
                    txt_len=int(video_pre["txt_len"]),
                    target_len=int(video_pre["target_len"]),
                    cond_len=int(video_pre["cond_len"]),
                    action_len=0,
                    device=self.device,
                    text_attention_mask=video_pre["text_mask"],
                )
            tokens_out = self._forward_flux2_video_only(video_pre, attention_mask)
            pred = self.video_expert.post_dit(tokens_out, video_pre)
            latents = scheduler.step(pred[:, :target_len], step_delta, latents)

        # Routing is action-independent -> compute once from the final hiddens.
        routed, _, _ = self._route(tokens_out["img"], target_len, horizon)

        # Action-space flow-matching ODE (noise -> action).
        a_generator = None
        if seed is not None:
            a_generator = torch.Generator(device=rand_device)
            a_generator.manual_seed(int(seed) + 1)
        a_lat = torch.randn(
            (batch_size, horizon, self.action_patch_dim),
            generator=a_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device)
        a_timesteps, a_deltas = scheduler.build_inference_schedule(
            num_inference_steps=int(self.action_flow_steps),
            device=self.device,
            dtype=torch.float32,
            shift_override=sigma_shift,
        )
        num_tt = float(self.train_video_scheduler.num_train_timesteps)
        for a_t, a_delta in zip(a_timesteps, a_deltas):
            sigma = (a_t / num_tt).expand(batch_size)
            v_act = self._flow_velocity(a_lat, sigma, routed)
            a_lat = scheduler.step(v_act, a_delta, a_lat)

        return {"action": a_lat[0].detach().to(device="cpu", dtype=torch.float32)}
