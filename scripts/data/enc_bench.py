#!/usr/bin/env python3
"""Benchmark libaom-av1 encoding parameter sets on one cam (339 frames)."""
import os, subprocess, time, sys
import numpy as np
import imageio_ffmpeg, av

FF = imageio_ffmpeg.get_ffmpeg_exe()
BENCH = "/storage/yukaichengLab/mazijian/wth/ImageWAM/scripts/data/_enc_bench"
FPS = 50
frames = np.load(f"{BENCH}/cam_high.npy")
N, H, W, _ = frames.shape
raw = frames.tobytes()

def encode(out_path, extra):
    cmd = [FF, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
           "-c:v", "libaom-av1"] + extra + [
           "-pix_fmt", "yuv420p", "-r", str(FPS), "-loglevel", "error", out_path]
    t = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(raw); p.stdin.close()
    ret = p.wait()
    dt = time.time() - t
    if ret != 0:
        return dt, None, None
    sz = os.path.getsize(out_path)
    # verify decode
    c = av.open(out_path); s = c.streams.video[0]
    nd = sum(1 for _ in c.decode(video=0))
    ok = (s.codec_context.pix_fmt == "yuv420p" and s.codec_context.width == W
          and s.codec_context.height == H and int(s.average_rate) == FPS
          and nd == N and "av1" in s.codec_context.name)
    return dt, sz, ok

CONFIGS = [
    ("baseline_default", []),
    ("cpu6_crf30", ["-cpu-used", "6", "-crf", "30", "-row-mt", "1"]),
    ("cpu8_crf30", ["-cpu-used", "8", "-crf", "30", "-row-mt", "1"]),
    ("cpu8_crf30_thr16", ["-cpu-used", "8", "-crf", "30", "-row-mt", "1", "-threads", "16"]),
    ("realtime_cpu8_crf30", ["-usage", "realtime", "-cpu-used", "8", "-crf", "30", "-row-mt", "1", "-threads", "16"]),
    ("realtime_cpu8_crf23", ["-usage", "realtime", "-cpu-used", "8", "-crf", "23", "-row-mt", "1", "-threads", "16"]),
    ("realtime_cpu10_crf30", ["-usage", "realtime", "-cpu-used", "10", "-crf", "30", "-row-mt", "1", "-threads", "16"]),
    ("realtime_cpu8_crf30_thr8", ["-usage", "realtime", "-cpu-used", "8", "-crf", "30", "-row-mt", "1", "-threads", "8"]),
]
os.makedirs("/tmp/encbench", exist_ok=True)
print(f"N={N} {W}x{H}")
print(f"{'config':32s} {'time(s)':>9s} {'kb/s':>8s} {'ok':>4s}")
for name, extra in CONFIGS:
    out = f"/tmp/encbench/{name}.mp4"
    dt, sz, ok = encode(out, extra)
    kbps = (sz * 8 / 1000) / (N / FPS) if sz else 0
    print(f"{name:32s} {dt:9.2f} {kbps:8.0f} {str(ok):>4s}")
