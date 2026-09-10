"""Probe whether MuJoCo/robosuite EGL rendering on the selected device is stable.
Reproduces the 08-28 gnho006 failure path (SIGABRT in mjr_readPixels after an env re-create):
create a LIBERO env, reset, step N times with camera obs (renders), close, and repeat R times.
Exit 0 = stable. Any abort/exception -> non-zero. Usage: CUDA_VISIBLE_DEVICES=d MUJOCO_EGL_DEVICE_ID=d python egl_probe_device.py [suite]
"""
import os, sys, faulthandler
faulthandler.enable()
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite = sys.argv[1] if len(sys.argv) > 1 else "libero_spatial"
N, R = int(os.environ.get("EGL_PROBE_STEPS", 100)), int(os.environ.get("EGL_PROBE_ROUNDS", 3))
b = benchmark.get_benchmark_dict()[suite]()
task = b.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
for r in range(R):
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0); env.reset()
    for t in range(N):
        obs, *_ = env.step(np.zeros(7, dtype=np.float32))
        assert obs["agentview_image"].shape[0] == 256
    env.close()
    print(f"probe round {r+1}/{R} ok", flush=True)
print("EGL_PROBE_OK", flush=True)
