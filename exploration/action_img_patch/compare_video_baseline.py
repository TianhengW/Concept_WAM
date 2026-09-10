"""公平对比:官方 MoT baseline(HF ImageWAM-FLUX.2-4B-RoboTwin) vs patch 90k,
未来帧预测 MSE(pred, GT)。同一 val 样本、同 infer_video_flux2、20步、seed=0。
串行加载(先 baseline 全跑、释放、再 patch)防 OOM。"""
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from imagewam.utils.config_resolvers import register_default_resolvers
register_default_resolvers()

FLUX2_MODEL = "/storage/yukaichengLab/share/model/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"
AE_MODEL = "/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors"
FLUX2_SRC = "/storage/yukaichengLab/mazijian/wth/flux2"
QWEN3 = "/storage/yukaichengLab/share/model/Qwen3-4B"
PATCH_CKPT = "/storage/yukaichengLab/mazijian/wth/ImageWAM/runs/robotwin_flux2_klein_4b_actionpatch_full/2026-08-10_19-24-58/checkpoints/weights/step_090000.pt"
BASE_CKPT = "/storage/yukaichengLab/share/model/ImageWAM-FLUX.2-4B-RoboTwin/model.pt"
CONFIG_DIR = "/storage/yukaichengLab/mazijian/wth/ImageWAM/configs"
OUT = "/storage/yukaichengLab/mazijian/wth/ImageWAM/exploration/action_img_patch/video_pred_baseline_vs_patch.png"
SAMPLE_IDX = [0, 300, 700, 1100, 1600, 2200, 2900, 3600]
ADIT = dict(action_dim=14, hidden_dim=1024, num_heads=24, attn_head_dim=128,
            num_layers_double=5, num_layers_single=20, mlp_ratio=4.0,
            max_action_horizon=64, use_gradient_checkpointing=False)
PAD, HEADER_H, LABEL_W = 6, 34, 150


def mse(a, b):
    return float(((a.float() - b.float()) ** 2).mean().cpu())


def to_img(t):
    x = t.detach().float().clamp(-1, 1).cpu().numpy()
    x = ((x + 1.0) * 127.5).astype(np.uint8)
    return np.transpose(x, (1, 2, 0))


def _font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            continue
    return ImageFont.load_default()


def gather_samples(ds):
    out = []
    for idx in SAMPLE_IDX:
        s = ds[idx % len(ds)]
        video = s["video"]
        cur, gt = video[:, 0], video[:, -1]
        prompt = s.get("instruction", s.get("prompt", "task"))
        if isinstance(prompt, (list, tuple)):
            prompt = prompt[0]
        task = s.get("task_name", "")
        if isinstance(task, (list, tuple)):
            task = task[0]
        proprio = s.get("proprio")
        if proprio is not None:
            proprio = torch.as_tensor(proprio)
            proprio = proprio[0:1] if proprio.ndim == 2 else proprio.unsqueeze(0) if proprio.ndim == 1 else proprio
        out.append(dict(idx=idx, cur=cur, gt=gt, prompt=str(prompt), task=str(task) if task else str(prompt)[:20], proprio=proprio))
    return out


def run_model(model, samples):
    preds = []
    for s in samples:
        with torch.no_grad():
            o = model.infer_video_flux2(prompt=s["prompt"], input_image=s["cur"].unsqueeze(0),
                                        proprio=s["proprio"], num_inference_steps=20, seed=0)
        preds.append(o["image"].detach().float().cpu())
    return preds


def main():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="train", overrides=["task=robotwin_flux2_klein_4b_actionpatch_full"])
    ds = instantiate(cfg.data.val)
    print("val dataset size:", len(ds), flush=True)
    samples = gather_samples(ds)

    print(">>> loading OFFICIAL baseline (MoT)", flush=True)
    from imagewam.models.backbones.imagewam import ImageWAM
    base = ImageWAM.from_flux2_klein_pretrained(
        flux2_model_path=FLUX2_MODEL, ae_model_path=AE_MODEL, flux2_src_path=FLUX2_SRC,
        action_dit_config=ADIT, action_dit_pretrained_path=None, variant="klein-base-4b",
        proprio_dim=14, load_text_encoder=True, device="cuda", torch_dtype=torch.bfloat16,
        mot_checkpoint_mixed_attn=True, qwen3_model_spec=QWEN3, qwen_context_len=128, concept_k=0)
    base.load_checkpoint(BASE_CKPT)
    base.eval()
    preds_b = run_model(base, samples)
    del base
    torch.cuda.empty_cache()
    print(">>> baseline done, freed", flush=True)

    print(">>> loading PATCH 90k", flush=True)
    from exploration.action_img_patch.model import ImageWAMActionPatch
    patch = ImageWAMActionPatch.from_flux2_klein_actionpatch_pretrained(
        flux2_model_path=FLUX2_MODEL, ae_model_path=AE_MODEL, flux2_src_path=FLUX2_SRC,
        qwen3_model_spec=QWEN3, qwen_context_len=128, proprio_dim=14, load_text_encoder=True,
        device="cuda", torch_dtype=torch.bfloat16, action_patch_dim=14)
    patch.load_checkpoint(PATCH_CKPT)
    patch.eval()
    preds_p = run_model(patch, samples)
    del patch
    torch.cuda.empty_cache()

    rows, mb_all, mp_all, mn_all = [], [], [], []
    for s, pb, pp in zip(samples, preds_b, preds_p):
        gt = s["gt"].float().cpu()
        mb, mp, mn = mse(pb, gt), mse(pp, gt), mse(s["cur"], gt)
        mb_all.append(mb); mp_all.append(mp); mn_all.append(mn)
        rows.append((to_img(s["cur"]), to_img(gt), to_img(pb), to_img(pp)))
        print(f"[{s['idx']}] {s['task']:22s} baseline={mb:.4f}  patch={mp:.4f}  naive={mn:.4f}  (patch vs base: {100*(1-mp/mb) if mb>0 else 0:+.0f}%)", flush=True)

    h, w, _ = rows[0][0].shape
    heads = ["current", "GT future", "MoT baseline pred", "PATCH 90k pred"]
    ncol = 4
    gw = LABEL_W + ncol * w + (ncol + 1) * PAD
    gh = HEADER_H + len(rows) * (h + PAD) + PAD
    cv = Image.new("RGB", (gw, gh), (245, 245, 245))
    dr = ImageDraw.Draw(cv)
    fh, fl = _font(19), _font(14)
    for j, hd in enumerate(heads):
        dr.text((LABEL_W + PAD + j * (w + PAD) + 6, 8), hd, fill=(20, 20, 20), font=fh)
    for i, (quad, s, mb, mp) in enumerate(zip(rows, samples, mb_all, mp_all)):
        y = HEADER_H + i * (h + PAD) + PAD
        dr.text((6, y + h // 2 - 16), s["task"][:20], fill=(30, 30, 30), font=fl)
        dr.text((6, y + h // 2 + 2), "base %.3f" % mb, fill=(0, 0, 150), font=fl)
        dr.text((6, y + h // 2 + 18), "patch %.3f" % mp, fill=(180, 0, 0), font=fl)
        for j, im in enumerate(quad):
            cv.paste(Image.fromarray(im), (LABEL_W + PAD + j * (w + PAD), y))
    cv.save(OUT)
    f = lambda x: sum(x) / len(x)
    print("=== SUMMARY: future-frame MSE(pred, GT), image space [-1,1] ===", flush=True)
    print("mean MoT baseline = %.4f" % f(mb_all), flush=True)
    print("mean PATCH 90k    = %.4f" % f(mp_all), flush=True)
    print("mean naive(copy)  = %.4f" % f(mn_all), flush=True)
    print("SAVED:", OUT, flush=True)


if __name__ == "__main__":
    main()
