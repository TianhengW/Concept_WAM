#!/usr/bin/env python3
"""
Convert official RoboDojo hdf5 episodes (arx_x5 dual-arm) into the fastwam-style
LeRobot v2.1 layout that ImageWAM's RobotVideoDataset consumes directly.

Input (official RoboDojo v1.0 hdf5, see third_party/RoboDojo + XPolicyLab):
  <raw_root>/<task>/arx_x5/data/episode_XXXXXXX.hdf5
    /instruction                       scalar bytes
    /state/{left,right}_arm_joint_states   (T,6) f64   joint positions (rad)
    /state/{left,right}_ee_joint_states    (T,1) f64   gripper in [0,1] (1=open)
    /action/...                            same keys, action[t] == state[t+1]
    /vision/cam_{head,left_wrist,right_wrist}/colors  (T,) |S<n> JPEG bytes (480x640 RGB)

Output (matches fastwam_robotwin/robotwin2.0 so `configs/data/robotwin_omnigen2.yaml`
and the compact_288x256 three-camera layout are reused unchanged):
  data/chunk-{000..}/episode_{index:06d}.parquet
  videos/chunk-{000..}/observation.images.{cam_high,cam_left_wrist,cam_right_wrist}/episode_{index:06d}.mp4
  meta/{info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl}

state/action are 14-d float32 packed as [L_arm6, L_grip1, R_arm6, R_grip1]
(XPolicyLab.utils.process_data.pack_robot_state order). Frames are stored at the
native 480x640 (no resize), av1/yuv420p @ 25 fps.

Usage:
  # verify one episode
  python robodojo_raw_to_lerobot.py --single stack_bowls 0 --out-root /tmp/rd_single
  # batch (resumable; deterministic order: sorted task name -> episode number)
  python robodojo_raw_to_lerobot.py --batch --tasks complete --out-root <dir>
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time

import h5py
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# constants / conventions
# ---------------------------------------------------------------------------
FPS = 25                          # RoboDojo collect_freq (env_cfg/arx_x5.yml)
CHUNKS_SIZE = 1000
IMG_H, IMG_W = 480, 640           # native RoboDojo camera resolution (H, W)
MAX_IMG_STAT_SAMPLES = 100
EMBODIMENT = "arx_x5"

VIDEO_ENC_THREADS = int(os.environ.get("VIDEO_ENC_THREADS", "4"))
VIDEO_CPU_USED = int(os.environ.get("VIDEO_CPU_USED", "8"))
VIDEO_CRF = int(os.environ.get("VIDEO_CRF", "23"))

DEFAULT_RAW_ROOT = "/storage/yukaichengLab/share/datasets/RoboDojo"
DEFAULT_STATE_DIR = "/storage/yukaichengLab/share/datasets/.robodojo_state"
# non-benchmark extras that live next to the 35 benchmark task folders
EXCLUDED_TASKS = {"dlc"}

JOINT_NAMES = [[
    "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4",
    "left_joint_5", "left_joint_6", "left_gripper",
    "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4",
    "right_joint_5", "right_joint_6", "right_gripper",
]]

# raw hdf5 camera group -> fastwam video feature key
CAM_MAP = [
    ("cam_head", "observation.images.cam_high"),
    ("cam_left_wrist", "observation.images.cam_left_wrist"),
    ("cam_right_wrist", "observation.images.cam_right_wrist"),
]
VIDEO_KEYS = [vk for _, vk in CAM_MAP]


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------
def get_ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode_video(frames_uint8, out_path, ffmpeg):
    """frames_uint8: (N, H, W, 3) uint8 RGB. Encode av1/yuv420p at FPS via stdin pipe."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n, h, w, _ = frames_uint8.shape
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
        "-i", "pipe:0",
        "-c:v", "libaom-av1",
        "-usage", "realtime", "-cpu-used", str(VIDEO_CPU_USED), "-crf", str(VIDEO_CRF),
        "-row-mt", "1", "-threads", str(VIDEO_ENC_THREADS),
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-loglevel", "error",
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.stdin.write(frames_uint8.tobytes())
    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed ({ret}) for {out_path}")


# ---------------------------------------------------------------------------
# stats (LeRobot v2.1 episodes_stats.jsonl conventions)
# ---------------------------------------------------------------------------
def sample_indices(n, max_samples=MAX_IMG_STAT_SAMPLES):
    if n <= max_samples:
        return np.arange(n)
    return np.linspace(0, n - 1, max_samples).astype(int)


def vector_stats(arr):
    return {
        "min": arr.min(axis=0).astype(np.float64).tolist(),
        "max": arr.max(axis=0).astype(np.float64).tolist(),
        "mean": arr.mean(axis=0).astype(np.float64).tolist(),
        "std": arr.std(axis=0).astype(np.float64).tolist(),
        "count": [int(arr.shape[0])],
    }


def scalar_stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    return {
        "min": [float(a.min())], "max": [float(a.max())],
        "mean": [float(a.mean())], "std": [float(a.std())],
        "count": [int(a.shape[0])],
    }


def image_stats(frames_uint8):
    x = frames_uint8.astype(np.float64) / 255.0
    mn, mx = x.min(axis=(0, 1, 2)), x.max(axis=(0, 1, 2))
    mean, std = x.mean(axis=(0, 1, 2)), x.std(axis=(0, 1, 2))

    def r(v):
        return [[[float(c)]] for c in v]  # (3,1,1)

    return {"min": r(mn), "max": r(mx), "mean": r(mean), "std": r(std),
            "count": [int(frames_uint8.shape[0])]}


# ---------------------------------------------------------------------------
# hdf5 reading
# ---------------------------------------------------------------------------
def _bytes(v):
    if isinstance(v, np.ndarray):
        v = v.tobytes()
    if isinstance(v, bytes):
        return v
    return bytes(v)


def read_instruction(f):
    raw = f["instruction"][()]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return str(raw).strip()


def pack_14(group):
    """/state or /action group -> (T,14) float32 [L_arm6, L_grip1, R_arm6, R_grip1]."""
    parts = [
        group["left_arm_joint_states"][:], group["left_ee_joint_states"][:],
        group["right_arm_joint_states"][:], group["right_ee_joint_states"][:],
    ]
    parts = [np.asarray(p, dtype=np.float32).reshape(p.shape[0], -1) for p in parts]
    out = np.concatenate(parts, axis=1)
    if out.shape[1] != 14:
        raise ValueError(f"packed dim {out.shape[1]} != 14")
    return out


def decode_frames(raw_bytes_arr):
    """(N,) JPEG byte strings (NUL padded) -> (N, IMG_H, IMG_W, 3) uint8 RGB."""
    out = np.empty((len(raw_bytes_arr), IMG_H, IMG_W, 3), dtype=np.uint8)
    for i, b in enumerate(raw_bytes_arr):
        img = Image.open(io.BytesIO(_bytes(b))).convert("RGB")
        a = np.asarray(img)
        if a.shape != (IMG_H, IMG_W, 3):
            raise ValueError(f"frame {i} decoded to {a.shape}, expected {(IMG_H, IMG_W, 3)}")
        out[i] = a
    return out


# ---------------------------------------------------------------------------
# per-episode conversion
# ---------------------------------------------------------------------------
def convert_episode(hdf5_path, out_root, episode_index, global_index_start,
                    task_index, instruction, write_videos=True):
    with h5py.File(hdf5_path, "r") as f:
        state = pack_14(f["state"])
        action = pack_14(f["action"])
        cam_raw = {vk: f[f"vision/{raw}/colors"][:] for raw, vk in CAM_MAP}
    N = state.shape[0]
    if action.shape[0] != N:
        raise ValueError(f"state {N} vs action {action.shape[0]} length mismatch")
    for vk, arr in cam_raw.items():
        if arr.shape[0] != N:
            raise ValueError(f"{vk} has {arr.shape[0]} frames, expected {N}")

    chunk = episode_index // CHUNKS_SIZE
    ffmpeg = get_ffmpeg()
    cam_frames = {}
    for vk in VIDEO_KEYS:
        frames = decode_frames(cam_raw[vk])
        cam_frames[vk] = frames
        if write_videos:
            vpath = os.path.join(out_root, "videos", f"chunk-{chunk:03d}", vk,
                                 f"episode_{episode_index:06d}.mp4")
            encode_video(frames, vpath, ffmpeg)

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    frame_index = np.arange(N, dtype=np.int64)
    timestamp = (frame_index.astype(np.float32) / float(FPS)).astype(np.float32)
    index = np.arange(global_index_start, global_index_start + N, dtype=np.int64)
    episode_index_col = np.full(N, episode_index, dtype=np.int64)
    frame_task_index = np.full(N, task_index, dtype=np.int64)

    df = pd.DataFrame({
        "observation.state": list(state),
        "action": list(action),
        "timestamp": timestamp,
        "frame_index": frame_index,
        "episode_index": episode_index_col,
        "index": index,
        "task_index": frame_task_index,
    })
    schema = pa.schema([
        ("observation.state", pa.list_(pa.float32())),
        ("action", pa.list_(pa.float32())),
        ("timestamp", pa.float32()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("index", pa.int64()),
        ("task_index", pa.int64()),
    ])
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=True)
    ppath = os.path.join(out_root, "data", f"chunk-{chunk:03d}", f"episode_{episode_index:06d}.parquet")
    os.makedirs(os.path.dirname(ppath), exist_ok=True)
    pq.write_table(table, ppath)

    stats = {
        "observation.state": vector_stats(state),
        "action": vector_stats(action),
    }
    idxs = sample_indices(N)
    for vk in VIDEO_KEYS:
        stats[vk] = image_stats(cam_frames[vk][idxs])
    stats["timestamp"] = scalar_stats(timestamp)
    stats["frame_index"] = scalar_stats(frame_index)
    stats["episode_index"] = scalar_stats(episode_index_col)
    stats["index"] = scalar_stats(index)
    stats["task_index"] = scalar_stats(frame_task_index)

    return {
        "ep_meta": {"episode_index": episode_index, "tasks": [instruction], "length": int(N)},
        "ep_stats": {"episode_index": episode_index, "stats": stats},
        "n_frames": int(N),
        "next_index": global_index_start + N,
    }


# ---------------------------------------------------------------------------
# meta writers
# ---------------------------------------------------------------------------
def write_info(out_root, total_episodes, total_frames, total_tasks):
    features = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": [14], "names": JOINT_NAMES},
    }
    for vk in VIDEO_KEYS:
        features[vk] = {
            "dtype": "video", "shape": [IMG_H, IMG_W, 3],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.height": IMG_H, "video.width": IMG_W,
                "video.codec": "av1", "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False, "video.fps": FPS,
                "video.channels": 3, "has_audio": False,
            },
        }
    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    for k in ["frame_index", "episode_index", "index", "task_index"]:
        features[k] = {"dtype": "int64", "shape": [1], "names": None}
    total_chunks = (total_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE
    info = {
        "codebase_version": "v2.1",
        "robot_type": EMBODIMENT,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * len(VIDEO_KEYS),
        "total_chunks": total_chunks,
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    os.makedirs(os.path.join(out_root, "meta"), exist_ok=True)
    with open(os.path.join(out_root, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def write_meta_jsonl(out_root, episodes_meta, episodes_stats, task_index_map, source_manifest=None):
    meta = os.path.join(out_root, "meta")
    os.makedirs(meta, exist_ok=True)
    with open(os.path.join(meta, "episodes.jsonl"), "w") as f:
        for em in episodes_meta:
            f.write(json.dumps(em) + "\n")
    with open(os.path.join(meta, "episodes_stats.jsonl"), "w") as f:
        for es in episodes_stats:
            f.write(json.dumps(es) + "\n")
    with open(os.path.join(meta, "tasks.jsonl"), "w") as f:
        for task, ti in sorted(task_index_map.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": ti, "task": task}) + "\n")
    if source_manifest is not None:
        # episode_index -> (robodojo task, raw episode number); handy for per-task eval slices
        with open(os.path.join(meta, "robodojo_source.jsonl"), "w") as f:
            for row in source_manifest:
                f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def resolve_tasks(raw_root, tasks_arg, state_dir):
    if tasks_arg == "all":
        names = [d for d in os.listdir(raw_root)
                 if os.path.isdir(os.path.join(raw_root, d, EMBODIMENT, "data"))]
    elif tasks_arg == "complete":
        marker_dir = os.path.join(state_dir, "tasks")
        names = [fn[:-len(".complete")] for fn in os.listdir(marker_dir) if fn.endswith(".complete")]
        names = [n for n in names if os.path.isdir(os.path.join(raw_root, n, EMBODIMENT, "data"))]
    else:
        names = [t.strip() for t in tasks_arg.split(",") if t.strip()]
    names = sorted(n for n in names if n not in EXCLUDED_TASKS)
    if not names:
        raise SystemExit("no tasks resolved")
    return names


def episode_number(fn):
    # episode_0000012.hdf5 -> 12
    stem = fn[len("episode_"):-len(".hdf5")]
    return int(stem)


def enumerate_episodes(raw_root, tasks):
    eps = []
    for task in tasks:
        ddir = os.path.join(raw_root, task, EMBODIMENT, "data")
        files = [fn for fn in os.listdir(ddir) if fn.startswith("episode_") and fn.endswith(".hdf5")]
        for fn in sorted(files, key=episode_number):
            eps.append({"task": task, "ep": episode_number(fn), "hdf5": os.path.join(ddir, fn)})
    return eps


def scan_episodes(eps, cache_path):
    """Read frame count + instruction per episode (cached; resumes are free)."""
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path))
        except Exception:
            cache = {}
    changed = False
    for e in eps:
        p = e["hdf5"]
        if p not in cache:
            with h5py.File(p, "r") as f:
                cache[p] = {"n_frames": int(f["state/left_arm_joint_states"].shape[0]),
                            "instruction": read_instruction(f)}
            changed = True
        e["n_frames"] = cache[p]["n_frames"]
        e["instruction"] = cache[p]["instruction"]
    if changed:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        json.dump(cache, open(cache_path, "w"))
    return eps


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------
def _batch_worker(job):
    hdf5, out_root, gidx, gstart, task_index, instruction, sidecar, write_videos = job
    try:
        r = convert_episode(hdf5, out_root, gidx, gstart, task_index, instruction, write_videos)
        with open(sidecar, "w") as f:
            json.dump({"ep_meta": r["ep_meta"], "ep_stats": r["ep_stats"]}, f)
        return (gidx, True, r["n_frames"], "")
    except Exception as ex:  # noqa: BLE001
        return (gidx, False, 0, repr(ex))


def _episode_outputs(out_root, gidx):
    chunk = gidx // CHUNKS_SIZE
    pq_path = os.path.join(out_root, "data", f"chunk-{chunk:03d}", f"episode_{gidx:06d}.parquet")
    vids = [os.path.join(out_root, "videos", f"chunk-{chunk:03d}", vk, f"episode_{gidx:06d}.mp4")
            for vk in VIDEO_KEYS]
    return pq_path, vids


def run_batch(args):
    tasks = resolve_tasks(args.raw_root, args.tasks, args.state_dir)
    eps = enumerate_episodes(args.raw_root, tasks)
    if args.every > 1:
        eps = eps[::args.every]
    if args.limit:
        eps = eps[:args.limit]
    print(f"[batch] tasks={len(tasks)} episodes={len(eps)}", flush=True)
    print(f"[batch] tasks: {' '.join(tasks)}", flush=True)

    eps = scan_episodes(eps, os.path.join(args.out_root, "meta", ".scan_cache.json"))
    gstart = 0
    task_index_map = {}
    for gidx, e in enumerate(eps):
        e["gidx"], e["gstart"] = gidx, gstart
        gstart += e["n_frames"]
        ins = e["instruction"]
        if ins not in task_index_map:
            task_index_map[ins] = len(task_index_map)
        e["task_index"] = task_index_map[ins]
    total_frames_all = gstart
    print(f"[batch] unique instructions={len(task_index_map)} total_frames={total_frames_all}", flush=True)

    sidecar_dir = os.path.join(args.out_root, "meta", ".episodes")
    os.makedirs(sidecar_dir, exist_ok=True)
    jobs, skipped = [], 0
    for e in eps:
        sidecar = os.path.join(sidecar_dir, f"episode_{e['gidx']:06d}.json")
        e["sidecar"] = sidecar
        pq_path, vids = _episode_outputs(args.out_root, e["gidx"])
        vids_ok = all(os.path.exists(v) and os.path.getsize(v) > 0 for v in vids)
        if args.refresh_parquet:
            write_videos = not vids_ok
        else:
            if os.path.exists(sidecar) and os.path.exists(pq_path) and vids_ok:
                skipped += 1
                continue
            write_videos = True
        jobs.append((e["hdf5"], args.out_root, e["gidx"], e["gstart"], e["task_index"],
                     e["instruction"], sidecar, write_videos))
    print(f"[batch] to_convert={len(jobs)} skipped(resume)={skipped} refresh_parquet={args.refresh_parquet}",
          flush=True)

    fails = []
    if jobs:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        t0, done = time.time(), 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_batch_worker, j) for j in jobs]
            for fu in as_completed(futs):
                gidx, ok, nf, err = fu.result()
                done += 1
                if not ok:
                    fails.append((gidx, err))
                    print(f"[FAIL] gidx={gidx} {err}", flush=True)
                if done % 20 == 0 or done == len(jobs):
                    el = time.time() - t0
                    eta = el / done * (len(jobs) - done)
                    print(f"[batch] {done}/{len(jobs)} elapsed={el:.0f}s eta={eta:.0f}s", flush=True)

    if fails:
        fp = os.path.join(args.out_root, "meta", "convert_failures.json")
        with open(fp, "w") as f:
            json.dump([{"gidx": g, "err": e} for g, e in fails], f, indent=2)
        raise SystemExit(f"[batch] {len(fails)} failures; final meta NOT written. See {fp}; re-run to resume.")

    episodes_meta, episodes_stats, manifest = [], [], []
    for e in eps:
        with open(e["sidecar"]) as f:
            d = json.load(f)
        episodes_meta.append(d["ep_meta"])
        episodes_stats.append(d["ep_stats"])
        manifest.append({"episode_index": e["gidx"], "task": e["task"], "raw_episode": e["ep"],
                         "length": e["n_frames"], "hdf5": e["hdf5"]})
    write_info(args.out_root, len(eps), total_frames_all, len(task_index_map))
    write_meta_jsonl(args.out_root, episodes_meta, episodes_stats, task_index_map, manifest)
    print(f"[batch DONE] tasks={len(tasks)} episodes={len(eps)} frames={total_frames_all} "
          f"instructions={len(task_index_map)}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    ap.add_argument("--state-dir", default=DEFAULT_STATE_DIR,
                    help="where <task>.complete markers live (for --tasks complete)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tasks", default="complete",
                    help="'complete' (tasks with a .complete marker), 'all', or comma list")
    ap.add_argument("--single", nargs=2, metavar=("TASK", "EP"), help="convert one episode")
    ap.add_argument("--no-videos", action="store_true")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--refresh-parquet", action="store_true")
    args = ap.parse_args()

    if args.batch:
        run_batch(args)
        return

    if args.single:
        task, ep = args.single[0], int(args.single[1])
        hdf5 = os.path.join(args.raw_root, task, EMBODIMENT, "data", f"episode_{ep:07d}.hdf5")
        with h5py.File(hdf5, "r") as f:
            instruction = read_instruction(f)
        print(f"[info] converting {hdf5}\n[info] instruction={instruction!r}")
        r = convert_episode(hdf5, args.out_root, episode_index=0, global_index_start=0,
                            task_index=0, instruction=instruction, write_videos=not args.no_videos)
        write_info(args.out_root, 1, r["n_frames"], 1)
        write_meta_jsonl(args.out_root, [r["ep_meta"]], [r["ep_stats"]], {instruction: 0},
                         [{"episode_index": 0, "task": task, "raw_episode": ep,
                           "length": r["n_frames"], "hdf5": hdf5}])
        print(f"[done] frames={r['n_frames']}")
        return

    raise SystemExit("No mode selected. Use --single or --batch.")


if __name__ == "__main__":
    main()
