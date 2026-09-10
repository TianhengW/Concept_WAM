#!/usr/bin/env python3
"""Encode ONE config (argv) and report time/kbps/PSNR/MAE/ok. Memory-safe (closes containers)."""
import os, subprocess, time, sys, gc
import numpy as np
import imageio_ffmpeg, av

FF = imageio_ffmpeg.get_ffmpeg_exe()
BENCH = "/storage/yukaichengLab/mazijian/wth/ImageWAM/scripts/data/_enc_bench"
FPS = 50
cam = sys.argv[1]
name = sys.argv[2]
extra = sys.argv[3:]

frames = np.load(f"{BENCH}/{cam}.npy")
N, H, W, _ = frames.shape

out = f"/tmp/encone_{name}.mp4"
cmd = [FF, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
       "-i", "pipe:0", "-c:v", "libaom-av1"] + extra + [
       "-pix_fmt", "yuv420p", "-r", str(FPS), "-loglevel", "error", out]
t = time.time()
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
p.stdin.write(frames.tobytes()); p.stdin.close(); ret = p.wait()
dt = time.time() - t
if ret != 0:
    print(f"{name} FAIL ret={ret}"); sys.exit(1)

# decode + metrics, streaming SSE to avoid second full buffer
c = av.open(out)
s = c.streams.video[0]
pix, cw, ch = s.codec_context.pix_fmt, s.codec_context.width, s.codec_context.height
rate = int(s.average_rate); codec = s.codec_context.name
sse = 0.0; abs_sum = 0.0; nd = 0
for i, fr in enumerate(c.decode(video=0)):
    d = fr.to_ndarray(format="rgb24").astype(np.float64)
    a = frames[i].astype(np.float64)
    diff = a - d
    sse += (diff * diff).sum(); abs_sum += np.abs(diff).sum(); nd += 1
c.close(); gc.collect()
npix = nd * H * W * 3
mse = sse / npix; mae = abs_sum / npix
psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 99.0
sz = os.path.getsize(out); kbps = (sz*8/1000)/(N/FPS)
ok = (pix=="yuv420p" and cw==W and ch==H and rate==FPS and "av1" in codec and nd==N)
print(f"{name:26s} t={dt:7.2f}s kbps={kbps:6.0f} PSNR={psnr:6.2f} MAE={mae:5.3f} "
      f"nframes={nd}/{N} pix={pix} {cw}x{ch} fps={rate} codec={codec} ok={ok}")
