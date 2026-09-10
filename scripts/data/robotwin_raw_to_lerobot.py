#!/usr/bin/env python3
"""
Convert raw RoboTwin hdf5 episodes into fastwam LeRobot v2.1 format.

Output layout (matches fastwam ground truth):
  data/chunk-{000..}/episode_{index:06d}.parquet
  videos/chunk-{000..}/observation.images.{cam_high,cam_left_wrist,cam_right_wrist}/episode_{index:06d}.mp4
  meta/{info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl}

This script supports batch conversion but is invoked for a single episode during
format verification. See main() / build_dataset().
"""
import argparse
import io
import json
import os
import subprocess
import sys
import random
import time

import h5py
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# constants / fastwam conventions
# ---------------------------------------------------------------------------
FPS = 50
CHUNKS_SIZE = 1000
# Native RoboTwin raw JPEG resolution is 240x320 (H,W). We store frames at native
# resolution (no upscaling): the RobotVideoDataset dataloader does an unconditional
# aspect-preserving resize into its compact layout, and base_lerobot_dataset.py:362
# image raw_shape check is commented out, so resolution is not validated on load.
IMG_H, IMG_W = 240, 320           # native raw resolution (H, W)
MAX_IMG_STAT_SAMPLES = 100        # LeRobot samples up to 100 frames for image stats

# libaom-av1 realtime encoding on native 240x320 (~1/4 pixels of 640x480).
# Per-cam ffmpeg thread budget; keep modest so many episodes can run in parallel.
VIDEO_ENC_THREADS = int(os.environ.get("VIDEO_ENC_THREADS", "4"))
VIDEO_CPU_USED = int(os.environ.get("VIDEO_CPU_USED", "8"))
VIDEO_CRF = int(os.environ.get("VIDEO_CRF", "23"))

JOINT_NAMES = [[
    "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
    "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll",
    "right_wrist_angle", "right_wrist_rotate", "right_gripper",
]]

# raw hdf5 camera -> fastwam video feature key
CAM_MAP = [
    ("head_camera", "observation.images.cam_high"),
    ("left_camera", "observation.images.cam_left_wrist"),
    ("right_camera", "observation.images.cam_right_wrist"),
]
VIDEO_KEYS = [vk for _, vk in CAM_MAP]

# path to RoboTwin instruction generator (imported dynamically)
ROBOTWIN_DESC_DIR = os.environ.get(
    "ROBOTWIN_DESC_DIR",
    "/storage/yukaichengLab/mazijian/wth/ImageWAM/third_party/RoboTwin/description",
)


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------
def get_ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def get_ffprobe(ffmpeg):
    cand = os.path.join(os.path.dirname(ffmpeg), "ffprobe")
    if os.path.exists(cand):
        return cand
    return "ffprobe"


def encode_video(frames_uint8, out_path, ffmpeg):
    """frames_uint8: (N, H, W, 3) uint8 RGB at native IMG_H x IMG_W (240x320).
    Encode to av1/yuv420p at FPS via rawvideo stdin pipe."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n, h, w, _ = frames_uint8.shape
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(FPS),
        "-i", "pipe:0",
        "-c:v", "libaom-av1",
        # --- speed: libaom realtime mode + row multithreading. On native 240x320
        #     (~1/4 pixels of 640x480) single-cam ~0.5-0.7s. Container/codec identical
        #     to fastwam GT (av1 / yuv420p / fps50).
        "-usage", "realtime",
        "-cpu-used", str(VIDEO_CPU_USED),
        "-crf", str(VIDEO_CRF),
        "-row-mt", "1",
        "-threads", str(VIDEO_ENC_THREADS),
        "-pix_fmt", "yuv420p",
        "-r", str(FPS),
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
# instruction generation (reuse RoboTwin official generator)
# ---------------------------------------------------------------------------
def generate_instructions_for_episode(task_name, episode_params, seed=None):
    """Return a de-duplicated list of English instruction strings for one episode.
    Falls back to full_description on any failure. Never raises."""
    try:
        if ROBOTWIN_DESC_DIR not in sys.path:
            sys.path.insert(0, os.path.join(ROBOTWIN_DESC_DIR, "utils"))
        import importlib
        gen = importlib.import_module("generate_episode_instructions")
        if seed is not None:
            random.seed(seed)
        # generate_episode_descriptions expects a list of episode param dicts
        results = gen.generate_episode_descriptions(task_name, [episode_params], max_descriptions=100)
        seen = results[0].get("seen", []) if results else []
        # de-dup preserving order
        out, s = [], set()
        for x in seen:
            if x not in s:
                s.add(x)
                out.append(x)
        if out:
            return out
    except Exception as e:
        print(f"[warn] instruction gen failed for {task_name}: {e}", file=sys.stderr)
    # fallback: full_description
    try:
        td = json.load(open(os.path.join(ROBOTWIN_DESC_DIR, "task_instruction", f"{task_name}.json")))
        fd = td.get("full_description")
        if fd:
            return [fd]
    except Exception:
        pass
    return [f"Perform the {task_name.replace('_', ' ')} task."]


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
def sample_indices(n, max_samples=MAX_IMG_STAT_SAMPLES):
    if n <= max_samples:
        return np.arange(n)
    return np.linspace(0, n - 1, max_samples).astype(int)


def vector_stats(arr):
    """arr: (N, D) -> min/max/mean/std shape (D,), count shape (1,)."""
    return {
        "min": arr.min(axis=0).astype(np.float64).tolist(),
        "max": arr.max(axis=0).astype(np.float64).tolist(),
        "mean": arr.mean(axis=0).astype(np.float64).tolist(),
        "std": arr.std(axis=0).astype(np.float64).tolist(),
        "count": [int(arr.shape[0])],
    }


def scalar_stats(arr):
    """arr: (N,) -> min/max/mean/std/count each shape (1,)."""
    a = np.asarray(arr, dtype=np.float64)
    return {
        "min": [float(a.min())],
        "max": [float(a.max())],
        "mean": [float(a.mean())],
        "std": [float(a.std())],
        "count": [int(a.shape[0])],
    }


def image_stats(frames_uint8):
    """frames_uint8: (M, H, W, 3) uint8 (already sampled). Per-channel stats in [0,1],
    shape [3,1,1] for min/max/mean/std, count [M]."""
    x = frames_uint8.astype(np.float64) / 255.0  # (M,H,W,3)
    # per channel over M,H,W
    mn = x.min(axis=(0, 1, 2))
    mx = x.max(axis=(0, 1, 2))
    mean = x.mean(axis=(0, 1, 2))
    std = x.std(axis=(0, 1, 2))

    def r(v):
        return [[[float(c)]] for c in v]  # -> (3,1,1)

    return {
        "min": r(mn), "max": r(mx), "mean": r(mean), "std": r(std),
        "count": [int(frames_uint8.shape[0])],
    }


# ---------------------------------------------------------------------------
# per-episode conversion
# ---------------------------------------------------------------------------
def decode_frames(raw_bytes_arr):
    """raw_bytes_arr: (N,) of JPEG byte strings. Decode at NATIVE resolution
    (no resize). Returns (N, IMG_H, IMG_W, 3) uint8. Raw RoboTwin JPEGs decode
    to 240x320 (H,W); we assert this so any off-spec source is caught."""
    out = np.empty((len(raw_bytes_arr), IMG_H, IMG_W, 3), dtype=np.uint8)
    for i, b in enumerate(raw_bytes_arr):
        if isinstance(b, np.ndarray):
            b = b.tobytes()
        img = Image.open(io.BytesIO(b)).convert("RGB")
        a = np.asarray(img)
        if a.shape != (IMG_H, IMG_W, 3):
            raise ValueError(
                f"frame {i} decoded to {a.shape}, expected {(IMG_H, IMG_W, 3)}"
            )
        out[i] = a
    return out


def convert_episode(hdf5_path, task_name, episode_params, out_root,
                    episode_index, global_index_start, task_index_map,
                    instr_seed=None, write_videos=True,
                    instructions=None, ep_task_indices=None):
    """Convert one hdf5 episode. Returns dict with per-episode meta + stats + next index.

    In batch mode, `instructions` + `ep_task_indices` are precomputed in the main
    process (global task map must be built before parquet task_index is written);
    workers receive them ready-made and never touch task_index_map."""
    with h5py.File(hdf5_path, "r") as f:
        vec = f["joint_action/vector"][:]  # (N,14) float64
        N = vec.shape[0]
        cam_raw = {}
        for raw_name, vk in CAM_MAP:
            cam_raw[vk] = f[f"observation/{raw_name}/rgb"][:]

    state = vec.astype(np.float32)   # (N,14) joint position at t
    # fastwam GT convention (verified exact on GT parquets): action[t] = state[t+1]
    # (next-step position target), and the last action repeats the final position.
    # NOTE: state must NOT equal action — the nonidle filter and the policy's
    # action semantics both rely on the one-step shift.
    action = np.concatenate([vec[1:], vec[-1:]], axis=0).astype(np.float32)

    # ---- instructions ----
    if instructions is None:
        instructions = generate_instructions_for_episode(task_name, episode_params, seed=instr_seed)
        # register in global task map, get per-instruction task_index
        ep_task_indices = []
        for ins in instructions:
            if ins not in task_index_map:
                task_index_map[ins] = len(task_index_map)
            ep_task_indices.append(task_index_map[ins])

    # per-frame task_index: fastwam assigns each frame a random instruction from the
    # episode's list (all instructions appear; counts vary). Seeded for reproducibility.
    rng = random.Random(instr_seed if instr_seed is not None else episode_index)
    frame_task_index = np.array(
        [ep_task_indices[rng.randrange(len(ep_task_indices))] for _ in range(N)],
        dtype=np.int64,
    )

    # ---- videos ----
    chunk = episode_index // CHUNKS_SIZE
    ffmpeg = get_ffmpeg()
    cam_frames = {}
    for vk in VIDEO_KEYS:
        frames = decode_frames(cam_raw[vk])  # (N, IMG_H, IMG_W, 3) native 240x320
        cam_frames[vk] = frames
        if write_videos:
            vpath = os.path.join(
                out_root, "videos", f"chunk-{chunk:03d}", vk,
                f"episode_{episode_index:06d}.mp4",
            )
            encode_video(frames, vpath, ffmpeg)

    # ---- parquet ----
    # Write via pandas.to_parquet so the emitted file carries the same pandas
    # schema metadata (range index, list[float32] columns) as fastwam GT, which
    # is produced by LeRobot's pandas-based writer.
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    frame_index = np.arange(N, dtype=np.int64)
    timestamp = (frame_index.astype(np.float32) / float(FPS)).astype(np.float32)
    index = np.arange(global_index_start, global_index_start + N, dtype=np.int64)
    episode_index_col = np.full(N, episode_index, dtype=np.int64)

    df = pd.DataFrame({
        "observation.state": list(state),   # each row a (14,) float32 ndarray
        "action": list(action),
        "timestamp": timestamp,
        "frame_index": frame_index,
        "episode_index": episode_index_col,
        "index": index,
        "task_index": frame_task_index,
    })
    # Force list<float32> for the two vector columns (matches GT schema exactly).
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
    ppath = os.path.join(
        out_root, "data", f"chunk-{chunk:03d}", f"episode_{episode_index:06d}.parquet"
    )
    os.makedirs(os.path.dirname(ppath), exist_ok=True)
    pq.write_table(table, ppath)

    # ---- stats ----
    stats = {}
    stats["observation.state"] = vector_stats(state)
    stats["action"] = vector_stats(action)
    idxs = sample_indices(N)
    for vk in VIDEO_KEYS:
        sampled = cam_frames[vk][idxs]
        stats[vk] = image_stats(sampled)
    stats["timestamp"] = scalar_stats(timestamp)
    stats["frame_index"] = scalar_stats(frame_index)
    stats["episode_index"] = scalar_stats(episode_index_col)
    stats["index"] = scalar_stats(index)
    stats["task_index"] = scalar_stats(frame_task_index)

    ep_meta = {"episode_index": episode_index, "tasks": instructions, "length": int(N)}
    ep_stats = {"episode_index": episode_index, "stats": stats}
    return {
        "ep_meta": ep_meta,
        "ep_stats": ep_stats,
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
            "dtype": "video",
            "shape": [IMG_H, IMG_W, 3],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.height": IMG_H, "video.width": IMG_W,
                "video.codec": "av1", "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False, "video.fps": 50,
                "video.channels": 3, "has_audio": False,
            },
        }
    for k in ["timestamp"]:
        features[k] = {"dtype": "float32", "shape": [1], "names": None}
    for k in ["frame_index", "episode_index", "index", "task_index"]:
        features[k] = {"dtype": "int64", "shape": [1], "names": None}

    total_chunks = (total_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE
    info = {
        "codebase_version": "v2.1",
        "robot_type": "aloha",
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


def write_meta_jsonl(out_root, episodes_meta, episodes_stats, task_index_map):
    meta = os.path.join(out_root, "meta")
    os.makedirs(meta, exist_ok=True)
    with open(os.path.join(meta, "episodes.jsonl"), "w") as f:
        for em in episodes_meta:
            f.write(json.dumps(em) + "\n")
    with open(os.path.join(meta, "episodes_stats.jsonl"), "w") as f:
        for es in episodes_stats:
            f.write(json.dumps(es) + "\n")
    tasks_sorted = sorted(task_index_map.items(), key=lambda kv: kv[1])
    with open(os.path.join(meta, "tasks.jsonl"), "w") as f:
        for task, ti in tasks_sorted:
            f.write(json.dumps({"task_index": ti, "task": task}) + "\n")


# ---------------------------------------------------------------------------
# discovery of episodes across the 3 sub-dataset directories
# ---------------------------------------------------------------------------
SUBDIRS = [
    ("data_clean_task_specified", "demo_clean"),
    ("data_random_task_specified", "demo_randomized"),
    ("data_random_task_specified_20", "demo_randomized_2"),
]


def load_scene_params(demo_dir):
    """Return list of per-episode param dicts, indexed by episode number from scene_info.json."""
    sip = os.path.join(demo_dir, "scene_info.json")
    params = {}
    try:
        scene = json.load(open(sip))
        for k, v in scene.items():
            # key like episode_0
            try:
                ei = int(k.split("_")[-1])
            except Exception:
                continue
            params[ei] = v.get("info", {}) if isinstance(v, dict) else {}
    except Exception as e:
        print(f"[warn] scene_info load failed {sip}: {e}", file=sys.stderr)
    return params


# ---------------------------------------------------------------------------
# batch conversion
# ---------------------------------------------------------------------------
TASKS_ORDER = [
    "hanging_mug", "move_stapler_pad", "pick_diverse_bottles",
    "place_can_basket", "place_object_basket", "stack_bowls_three",
    "turn_switch",
]


def enumerate_episodes(raw_root):
    """Deterministic enumeration: SUBDIRS order -> TASKS_ORDER -> numeric episode order."""
    eps = []
    for subdir, demo_sub in SUBDIRS:
        for task in TASKS_ORDER:
            demo_dir = os.path.join(raw_root, subdir, task, demo_sub)
            ddir = os.path.join(demo_dir, "data")
            if not os.path.isdir(ddir):
                print(f"[warn] missing {ddir}", file=sys.stderr)
                continue
            nums = []
            for fn in os.listdir(ddir):
                if fn.startswith("episode") and fn.endswith(".hdf5"):
                    try:
                        nums.append(int(fn[len("episode"):-len(".hdf5")]))
                    except ValueError:
                        pass
            for n in sorted(nums):
                eps.append({
                    "subdir": subdir, "task": task, "ep": n,
                    "hdf5": os.path.join(ddir, f"episode{n}.hdf5"),
                    "demo_dir": demo_dir,
                })
    return eps


def scan_frames(eps, cache_path):
    """Read per-episode frame count (hdf5 shape only). Cached: rescans are free on resume."""
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
                cache[p] = int(f["joint_action/vector"].shape[0])
            changed = True
        e["n_frames"] = cache[p]
    if changed:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        json.dump(cache, open(cache_path, "w"))
    return eps


def _batch_worker(job):
    """Runs in a subprocess. Instructions/task indices arrive precomputed."""
    (hdf5, task, ep_params, out_root, gidx, gstart,
     instructions, ep_task_indices, sidecar, write_videos) = job
    try:
        r = convert_episode(
            hdf5, task, ep_params, out_root,
            episode_index=gidx, global_index_start=gstart,
            task_index_map=None, instr_seed=gidx, write_videos=write_videos,
            instructions=instructions, ep_task_indices=ep_task_indices,
        )
        with open(sidecar, "w") as f:
            json.dump({"ep_meta": r["ep_meta"], "ep_stats": r["ep_stats"]}, f)
        return (gidx, True, r["n_frames"], "")
    except Exception as ex:
        return (gidx, False, 0, repr(ex))


def _episode_outputs(out_root, gidx):
    chunk = gidx // CHUNKS_SIZE
    pq_path = os.path.join(out_root, "data", f"chunk-{chunk:03d}",
                           f"episode_{gidx:06d}.parquet")
    vids = [os.path.join(out_root, "videos", f"chunk-{chunk:03d}", vk,
                         f"episode_{gidx:06d}.mp4") for vk in VIDEO_KEYS]
    return pq_path, vids


def run_batch(args):
    eps = enumerate_episodes(args.raw_root)
    if args.every > 1:
        eps = eps[::args.every]
    if args.limit:
        eps = eps[:args.limit]
    print(f"[batch] episodes to process: {len(eps)}", flush=True)

    eps = scan_frames(eps, os.path.join(args.out_root, "meta", ".scan_cache.json"))
    gstart = 0
    for gidx, e in enumerate(eps):
        e["gidx"], e["gstart"] = gidx, gstart
        gstart += e["n_frames"]
    total_frames_all = gstart

    # instructions + global task map in main process (deterministic: seed=gidx)
    task_index_map = {}
    scene_cache = {}
    for e in eps:
        if e["demo_dir"] not in scene_cache:
            scene_cache[e["demo_dir"]] = load_scene_params(e["demo_dir"])
        ep_params = scene_cache[e["demo_dir"]].get(e["ep"], {})
        ins = generate_instructions_for_episode(e["task"], ep_params, seed=e["gidx"])
        idxs = []
        for s in ins:
            if s not in task_index_map:
                task_index_map[s] = len(task_index_map)
            idxs.append(task_index_map[s])
        e["params"], e["instructions"], e["task_indices"] = ep_params, ins, idxs
    print(f"[batch] global tasks={len(task_index_map)} total_frames={total_frames_all}", flush=True)

    sidecar_dir = os.path.join(args.out_root, "meta", ".episodes")
    os.makedirs(sidecar_dir, exist_ok=True)
    jobs, skipped = [], 0
    for e in eps:
        sidecar = os.path.join(sidecar_dir, f"episode_{e['gidx']:06d}.json")
        e["sidecar"] = sidecar
        pq_path, vids = _episode_outputs(args.out_root, e["gidx"])
        vids_ok = all(os.path.exists(v) and os.path.getsize(v) > 0 for v in vids)
        if args.refresh_parquet:
            # regenerate parquet + stats sidecar; keep videos that already exist
            write_videos = not vids_ok
        else:
            if os.path.exists(sidecar) and os.path.exists(pq_path) and vids_ok:
                skipped += 1
                continue
            write_videos = True
        jobs.append((e["hdf5"], e["task"], e["params"], args.out_root,
                     e["gidx"], e["gstart"], e["instructions"], e["task_indices"],
                     sidecar, write_videos))
    print(f"[batch] to_convert={len(jobs)} skipped(resume)={skipped} "
          f"refresh_parquet={args.refresh_parquet}", flush=True)

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
                if done % 10 == 0 or done == len(jobs):
                    el = time.time() - t0
                    eta = el / done * (len(jobs) - done)
                    print(f"[batch] {done}/{len(jobs)} elapsed={el:.0f}s eta={eta:.0f}s",
                          flush=True)

    if fails:
        fp = os.path.join(args.out_root, "meta", "convert_failures.json")
        with open(fp, "w") as f:
            json.dump([{"gidx": g, "err": e} for g, e in fails], f, indent=2)
        raise SystemExit(
            f"[batch] {len(fails)} failures; final meta NOT written. "
            f"See {fp}; re-run same command to resume.")

    episodes_meta, episodes_stats = [], []
    for e in eps:
        with open(e["sidecar"]) as f:
            d = json.load(f)
        episodes_meta.append(d["ep_meta"])
        episodes_stats.append(d["ep_stats"])
    write_info(args.out_root, total_episodes=len(eps),
               total_frames=total_frames_all, total_tasks=len(task_index_map))
    write_meta_jsonl(args.out_root, episodes_meta, episodes_stats, task_index_map)
    print(f"[batch DONE] episodes={len(eps)} frames={total_frames_all} "
          f"tasks={len(task_index_map)}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="/storage/yukaichengLab/mazijian/wth/Robotwin_aug")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--single", nargs=3, metavar=("SUBDIR", "TASK", "EP"),
                    help="Convert a single episode: subdir_key task_name episode_number")
    ap.add_argument("--no-videos", action="store_true")
    ap.add_argument("--batch", action="store_true",
                    help="Convert all episodes under raw-root (deterministic order)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0,
                    help="Batch: only first K episodes of the enumeration")
    ap.add_argument("--every", type=int, default=1,
                    help="Batch: take every M-th episode (smoke tests across subdirs)")
    ap.add_argument("--refresh-parquet", action="store_true",
                    help="Batch: rewrite parquet+stats for every episode, reuse existing videos")
    args = ap.parse_args()

    if args.batch:
        run_batch(args)
        return

    task_index_map = {}
    episodes_meta = []
    episodes_stats = []
    total_frames = 0

    if args.single:
        subdir_key, task_name, ep = args.single
        ep = int(ep)
        demo_sub = dict(SUBDIRS)[subdir_key]
        demo_dir = os.path.join(args.raw_root, subdir_key, task_name, demo_sub)
        hdf5 = os.path.join(demo_dir, "data", f"episode{ep}.hdf5")
        params = load_scene_params(demo_dir)
        ep_params = params.get(ep, {})
        print(f"[info] converting {hdf5}\n[info] episode_params={ep_params}")
        r = convert_episode(
            hdf5, task_name, ep_params, args.out_root,
            episode_index=0, global_index_start=0,
            task_index_map=task_index_map, instr_seed=ep,
            write_videos=not args.no_videos,
        )
        episodes_meta.append(r["ep_meta"])
        episodes_stats.append(r["ep_stats"])
        total_frames = r["n_frames"]
        write_info(args.out_root, total_episodes=1, total_frames=total_frames,
                   total_tasks=len(task_index_map))
        write_meta_jsonl(args.out_root, episodes_meta, episodes_stats, task_index_map)
        print(f"[done] frames={total_frames} tasks={len(task_index_map)}")
        return

    raise SystemExit("No mode selected. Use --single or --batch.")


if __name__ == "__main__":
    main()
