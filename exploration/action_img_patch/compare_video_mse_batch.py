"""批量 video-MSE 对比:官方 MoT baseline vs patch 90k。
val dataset 无 task_id,只有自然语言 instruction → 按 dataset 均匀切 K_TASK 块,
每块取 M_EP 个相邻样本(近似同场景多 episode/帧窗口),label 取指令关键词。
串行加载防 OOM。输出 video_MSE/: per_sample.csv, summary.md, per_task_mse_bar.png, montages/*.png"""
import os, csv, re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
OUTDIR = "/storage/yukaichengLab/mazijian/wth/ImageWAM/exploration/action_img_patch/video_MSE"
ADIT = dict(action_dim=14, hidden_dim=1024, num_heads=24, attn_head_dim=128,
            num_layers_double=5, num_layers_single=20, mlp_ratio=4.0,
            max_action_horizon=64, use_gradient_checkpointing=False)
K_TASK = 15
M_EP = 5
PAD, HEADER_H, LABEL_W = 6, 30, 150


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


def sanit(s):
    return (re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:40]) or "block"


def gather(ds):
    N = len(ds)
    step = max(1, N // K_TASK)
    samples = []
    for b in range(K_TASK):
        base_idx = b * step
        label = None
        for e in range(M_EP):
            idx = base_idx + e
            if idx >= N:
                break
            try:
                s = ds[idx]
            except Exception:
                continue
            video = s["video"]
            cur, gt = video[:, 0], video[:, -1]
            prompt = s.get("instruction", s.get("prompt", "task"))
            if isinstance(prompt, (list, tuple)):
                prompt = prompt[0]
            prompt = str(prompt)
            if label is None:
                tail = prompt.split("instruction:")[-1].strip()
                label = f"b{b:02d}_" + (tail[:24] if tail else prompt[:24])
            proprio = s.get("proprio")
            if proprio is not None:
                proprio = torch.as_tensor(proprio)
                proprio = proprio[0:1] if proprio.ndim == 2 else proprio.unsqueeze(0) if proprio.ndim == 1 else proprio
            samples.append(dict(task=label, ep=e, idx=idx, cur=cur.cpu(), gt=gt.cpu(), prompt=prompt, proprio=proprio))
    print(f"gathered {len(samples)} samples over {K_TASK} blocks", flush=True)
    return samples


def run_model(model, samples):
    preds = []
    for s in samples:
        with torch.no_grad():
            o = model.infer_video_flux2(prompt=s["prompt"], input_image=s["cur"].unsqueeze(0),
                                        proprio=s["proprio"], num_inference_steps=20, seed=0)
        preds.append(o["image"].detach().float().cpu())
    return preds


def montage(task, subs, pbs, pps):
    h, w, _ = to_img(subs[0]["cur"]).shape
    ncol = 4
    heads = ["current", "GT future", "MoT baseline", "PATCH 90k"]
    gw = LABEL_W + ncol * w + (ncol + 1) * PAD
    gh = HEADER_H + len(subs) * (h + PAD) + PAD
    cv = Image.new("RGB", (gw, gh), (245, 245, 245))
    dr = ImageDraw.Draw(cv)
    fh, fl = _font(16), _font(12)
    for j, hd in enumerate(heads):
        dr.text((LABEL_W + PAD + j * (w + PAD) + 4, 6), hd, fill=(20, 20, 20), font=fh)
    for i, (s, pb, pp) in enumerate(zip(subs, pbs, pps)):
        y = HEADER_H + i * (h + PAD) + PAD
        gt = s["gt"]
        dr.text((4, y + 4), f"ep{s['ep']} idx{s['idx']}", fill=(30, 30, 30), font=fl)
        dr.text((4, y + 22), "base %.3f" % mse(pb, gt), fill=(0, 0, 150), font=fl)
        dr.text((4, y + 38), "patch %.3f" % mse(pp, gt), fill=(180, 0, 0), font=fl)
        for j, im in enumerate([to_img(s["cur"]), to_img(gt), to_img(pb), to_img(pp)]):
            cv.paste(Image.fromarray(im), (LABEL_W + PAD + j * (w + PAD), y))
    cv.save(os.path.join(OUTDIR, "montages", f"{sanit(task)}.png"))


def main():
    os.makedirs(os.path.join(OUTDIR, "montages"), exist_ok=True)
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="train", overrides=["task=robotwin_flux2_klein_4b_actionpatch_full"])
    ds = instantiate(cfg.data.val)
    print("val size:", len(ds), flush=True)
    samples = gather(ds)

    print(">>> loading OFFICIAL baseline (MoT)", flush=True)
    from imagewam.models.backbones.imagewam import ImageWAM
    base = ImageWAM.from_flux2_klein_pretrained(
        flux2_model_path=FLUX2_MODEL, ae_model_path=AE_MODEL, flux2_src_path=FLUX2_SRC,
        action_dit_config=ADIT, action_dit_pretrained_path=None, variant="klein-base-4b",
        proprio_dim=14, load_text_encoder=True, device="cuda", torch_dtype=torch.bfloat16,
        mot_checkpoint_mixed_attn=True, qwen3_model_spec=QWEN3, qwen_context_len=128, concept_k=0)
    base.load_checkpoint(BASE_CKPT); base.eval()
    preds_b = run_model(base, samples)
    del base; torch.cuda.empty_cache()
    print(">>> baseline done", flush=True)

    print(">>> loading PATCH 90k", flush=True)
    from exploration.action_img_patch.model import ImageWAMActionPatch
    patch = ImageWAMActionPatch.from_flux2_klein_actionpatch_pretrained(
        flux2_model_path=FLUX2_MODEL, ae_model_path=AE_MODEL, flux2_src_path=FLUX2_SRC,
        qwen3_model_spec=QWEN3, qwen_context_len=128, proprio_dim=14, load_text_encoder=True,
        device="cuda", torch_dtype=torch.bfloat16, action_patch_dim=14)
    patch.load_checkpoint(PATCH_CKPT); patch.eval()
    preds_p = run_model(patch, samples)
    del patch; torch.cuda.empty_cache()

    rows = []
    for s, pb, pp in zip(samples, preds_b, preds_p):
        gt = s["gt"]
        rows.append(dict(task=s["task"], ep=s["ep"], idx=s["idx"],
                         base=mse(pb, gt), patch=mse(pp, gt), naive=mse(s["cur"], gt)))
        print(f"{s['task']:30s} ep{s['ep']} base={rows[-1]['base']:.4f} patch={rows[-1]['patch']:.4f}", flush=True)

    with open(os.path.join(OUTDIR, "per_sample.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["block_label", "episode", "dataset_idx", "baseline_mse", "patch_mse", "naive_mse"])
        for r in rows:
            w.writerow([r["task"], r["ep"], r["idx"], f"{r['base']:.5f}", f"{r['patch']:.5f}", f"{r['naive']:.5f}"])

    tasks = []
    for r in rows:
        if r["task"] not in tasks:
            tasks.append(r["task"])
    per_task = {}
    for t in tasks:
        rs = [r for r in rows if r["task"] == t]
        per_task[t] = (float(np.mean([r["base"] for r in rs])), float(np.mean([r["patch"] for r in rs])), len(rs))
    ob = float(np.mean([r["base"] for r in rows])); op = float(np.mean([r["patch"] for r in rows])); on = float(np.mean([r["naive"] for r in rows]))
    win_s = sum(1 for r in rows if r["patch"] <= r["base"])

    L = ["# 批量 video-MSE 对比:官方 MoT baseline vs patch 90k", "",
         f"- 样本: **{len(rows)} 个** = dataset 均匀切 {len(tasks)} 块 × 每块 {M_EP} 个相邻样本(近似同场景多 episode/帧窗口)",
         "- val dataset 无 task_id,仅自然语言 instruction;块 label 取指令关键词(`bNN_...`)",
         "- 指标: 未来帧 MSE(pred, GT),图像空间 [-1,1],同 `infer_video_flux2` / 20 步 / seed=0",
         "- baseline: 官方 HF `ImageWAM-FLUX.2-4B-RoboTwin/model.pt`(MoT,step 69900);patch: step_090000",
         "",
         "| 总体(N={}) | MSE |".format(len(rows)), "|---|---:|",
         f"| **patch 90k** | **{op:.4f}** |",
         f"| **官方 MoT baseline** | **{ob:.4f}** |",
         f"| naive(copy-current) | {on:.4f} |",
         f"| patch vs baseline | {100*(1-op/ob):+.1f}% |",
         f"| 逐样本 patch ≤ baseline | {win_s}/{len(rows)} |", "",
         "## 逐块均值", "",
         "| block | n | baseline MSE | patch MSE | patch 优势 |",
         "|---|---:|---:|---:|---:|"]
    win_t = 0
    for t in tasks:
        b, p, n = per_task[t]
        if p <= b:
            win_t += 1
        L.append(f"| {t} | {n} | {b:.4f} | {p:.4f} | {100*(1-p/b) if b>0 else 0:+.0f}% |")
    L.append("")
    L.append(f"- 逐块 patch ≤ baseline: **{win_t}/{len(tasks)}**")
    open(os.path.join(OUTDIR, "summary.md"), "w").write("\n".join(L))

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(tasks)); ww = 0.4
    ax.bar(x - ww/2, [per_task[t][0] for t in tasks], ww, label="MoT baseline", color="#1f77b4")
    ax.bar(x + ww/2, [per_task[t][1] for t in tasks], ww, label="patch 90k", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("future-frame MSE(pred, GT)")
    ax.set_title(f"per-block video-MSE: patch {op:.4f} vs MoT baseline {ob:.4f} (N={len(rows)})")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "per_task_mse_bar.png"), dpi=150)

    for t in tasks:
        idxs = [i for i, s in enumerate(samples) if s["task"] == t]
        montage(t, [samples[i] for i in idxs], [preds_b[i] for i in idxs], [preds_p[i] for i in idxs])

    print("=== SUMMARY ===", flush=True)
    print(f"patch={op:.4f} baseline={ob:.4f} naive={on:.4f} sample_win={win_s}/{len(rows)} block_win={win_t}/{len(tasks)}", flush=True)
    print("SAVED to", OUTDIR, flush=True)


if __name__ == "__main__":
    main()
