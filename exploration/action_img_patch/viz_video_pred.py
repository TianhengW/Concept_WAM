"""Polished visualization of 90k patch ckpt future-frame prediction.

8 diverse RoboTwin samples, each row: current | GT future | predicted future,
with column headers and per-row task labels. Shows action-as-patch training
did not hurt the video (world-model) stream.
"""
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
CKPT = "/storage/yukaichengLab/mazijian/wth/ImageWAM/runs/robotwin_flux2_klein_4b_actionpatch_full/2026-08-10_19-24-58/checkpoints/weights/step_090000.pt"
CONFIG_DIR = "/storage/yukaichengLab/mazijian/wth/ImageWAM/configs"
OUT = "/storage/yukaichengLab/mazijian/wth/ImageWAM/exploration/action_img_patch/video_pred_90k.png"
SAMPLE_IDX = [0, 300, 700, 1100, 1600, 2200, 2900, 3600]
PAD = 6
HEADER_H = 34
LABEL_W = 150


def mse(a, b):  # [3,H,W] float in [-1,1]
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


def main():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="train",
                      overrides=["task=robotwin_flux2_klein_4b_actionpatch_full"])
    ds = instantiate(cfg.data.val)
    print("val dataset size:", len(ds), flush=True)

    from exploration.action_img_patch.model import ImageWAMActionPatch
    model = ImageWAMActionPatch.from_flux2_klein_actionpatch_pretrained(
        flux2_model_path=FLUX2_MODEL, ae_model_path=AE_MODEL, flux2_src_path=FLUX2_SRC,
        qwen3_model_spec=QWEN3, qwen_context_len=128, proprio_dim=14,
        load_text_encoder=True, device="cuda", torch_dtype=torch.bfloat16, action_patch_dim=14,
    )
    model.load_checkpoint(CKPT)
    model.eval()
    print("loaded 90k ckpt", flush=True)

    tiles = []      # list of (cur, gt, pred) uint8 HWC
    mses = []
    labels = []
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
        with torch.no_grad():
            out = model.infer_video_flux2(prompt=str(prompt), input_image=cur.unsqueeze(0),
                                          proprio=proprio, num_inference_steps=20, seed=0)
        pred = out["image"]
        m_pred = mse(pred, gt)
        m_base = mse(cur, gt)  # naive: copy current as future
        mses.append((m_pred, m_base))
        tiles.append((to_img(cur), to_img(gt), to_img(pred)))
        lab = str(task) if task else str(prompt)[:24]
        labels.append(lab)
        print(f"[{idx}] {lab}: MSE(pred,gt)=%.4f  MSE(cur,gt)=%.4f  reduction=%.0f%%" % (m_pred, m_base, 100*(1-m_pred/m_base) if m_base>0 else 0), flush=True)

    h, w, _ = tiles[0][0].shape
    ncol = 3
    grid_w = LABEL_W + ncol * w + (ncol + 1) * PAD
    grid_h = HEADER_H + len(tiles) * (h + PAD) + PAD
    canvas = Image.new("RGB", (grid_w, grid_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    fh, fl = _font(20), _font(15)
    heads = ["current frame", "GT future", "PRED future (90k)"]
    for j, head in enumerate(heads):
        x = LABEL_W + PAD + j * (w + PAD)
        draw.text((x + w // 2 - 55, 8), head, fill=(20, 20, 20), font=fh)
    for i, (trip, lab) in enumerate(zip(tiles, labels)):
        y = HEADER_H + i * (h + PAD) + PAD
        draw.text((6, y + h // 2 - 8), lab[:20], fill=(30, 30, 30), font=fl)
        for j, im in enumerate(trip):
            x = LABEL_W + PAD + j * (w + PAD)
            canvas.paste(Image.fromarray(im), (x, y))
    # per-row MSE annotation
    for i, (mp, mb) in enumerate(mses):
        y = HEADER_H + i * (h + PAD) + PAD
        draw.text((6, y + h // 2 + 12), "MSE %.3f" % mp, fill=(180, 0, 0), font=fl)
    ap = sum(m[0] for m in mses)/len(mses); ab = sum(m[1] for m in mses)/len(mses)
    canvas.save(OUT)
    print("=== MSE summary (image space, [-1,1]) ===", flush=True)
    print("mean MSE(pred,gt) = %.4f" % ap, flush=True)
    print("mean MSE(cur,gt)  = %.4f  (naive copy-current baseline)" % ab, flush=True)
    print("prediction reduces error by %.0f%% vs naive" % (100*(1-ap/ab)), flush=True)
    print("SAVED:", OUT, flush=True)


if __name__ == "__main__":
    main()
