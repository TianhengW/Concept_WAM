#!/usr/bin/env python3
"""Benchmark + PSNR/MAE vs source for candidate av1 configs. cpu-used max=8."""
import os, subprocess, time
import numpy as np
import imageio_ffmpeg, av

FF = imageio_ffmpeg.get_ffmpeg_exe()
BENCH = "/storage/yukaichengLab/mazijian/wth/ImageWAM/scripts/data/_enc_bench"
FPS = 50
frames = np.load(f"{BENCH}/cam_high.npy")  # source truth (N,H,W,3) uint8
N, H, W, _ = frames.shape
raw = frames.tobytes()

def encode(out_path, extra):
    cmd = [FF, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
           "-c:v", "libaom-av1"] + extra + [
           "-pix_fmt", "yuv420p", "-r", str(FPS), "-loglevel", "error", out_path]
    t = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(raw); p.stdin.close(); ret = p.wait()
    return time.time() - t, ret

def decode_all(path):
    c = av.open(path)
    out = np.empty((N, H, W, 3), np.uint8)
    for i, fr in enumerate(c.decode(video=0)):
        out[i] = fr.to_ndarray(format="rgb24")
    s = av.open(path).streams.video[0]
    return out, s

def metrics(dec):
    a = frames.astype(np.float64); b = dec.astype(np.float64)
    mae = np.abs(a - b).mean()
    mse = ((a - b) ** 2).mean()
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 99.0
    return psnr, mae

CONFIGS = [
    ("good_cpu8_crf30", ["-usage", "good", "-cpu-used", "8", "-crf", "30", "-row-mt", "1", "-threads", "16"]),
    ("realtime_cpu8_crf30", ["-usage", "realtime", "-cpu-used", "8", "-crf", "30", "-row-mt", "1", "-threads", "16"]),
    ("realtime_cpu8_crf23", ["-usage", "realtime", "-cpu-used", "8", "-crf", "23", "-row-mt", "1", "-threads", "16"]),
    ("realtime_cpu7_crf25", ["-usage", "realtime", "-cpu-used", "7", "-crf", "25", "-row-mt", "1", "-threads", "16"]),
    ("realtime_cpu6_crf25", ["-usage", "realtime", "-cpu-used", "6", "-crf", "25", "-row-mt", "1", "-threads", "16"]),
    ("good_cpu6_crf25", ["-usage", "good", "-cpu-used", "6", "-crf", "25", "-row-mt", "1", "-threads", "16"]),
]
os.makedirs("/tmp/encbench2", exist_ok=True)
print(f"N={N} {W}x{H}")
print(f"{'config':28s} {'time(s)':>8s} {'kb/s':>7s} {'PSNR':>7s} {'MAE':>6s} {'ok':>4s}")
for name, extra in CONFIGS:
    out = f"/tmp/encbench2/{name}.mp4"
    dt, ret = encode(out, extra)
    if ret != 0:
        print(f"{name:28s}  FAIL ret={ret}"); continue
    dec, s = decode_all(out)
    psnr, mae = metrics(dec)
    sz = os.path.getsize(out); kbps = (sz*8/1000)/(N/FPS)
    ok = (s.codec_context.pix_fmt=="yuv420p" and s.codec_context.width==W
          and s.codec_context.height==H and int(s.average_rate)==FPS and "av1" in s.codec_context.name)
    print(f"{name:28s} {dt:8.2f} {kbps:7.0f} {psnr:7.2f} {mae:6.3f} {str(ok):>4s}")
