#!/usr/bin/env python3
"""Auto-eval watchdog for patch FULL-data training.

Whenever a checkpoint at a multiple of STRIDE (10000) steps is written and
fully flushed, submit a 10-episode 50-task {clean,random} eval on a healthy
node. Serial (one eval at a time, no pileup), timeout-guarded (curobo poison
tasks can't hang it for hours), no dupes. step_010000 already evaluated (65.8).
"""
import glob
import os
import re
import subprocess
import time

import yaml

R = "/storage/yukaichengLab/mazijian/wth/ImageWAM"
RUNS = R + "/runs/robotwin_flux2_klein_4b_actionpatch_full"
SUB = "/tmp/apfull_eval_done.txt"
STRIDE = 10000
BASE = 10000          # step_010000 already evaluated
MIN_SIZE = 7_000_000_000
# FULL 50-task metric. curobo/warp deadlock tasks are handled by the manager's
# HARD_TIMEOUT (patched): a hung worker is SIGKILLed and scored 0.0, so all 50
# tasks x {clean,random} complete without hanging the eval.
# Widened pool: exclude only known-bad nodes + the 4 training nodes (033-036).
# H100 alone gets starved (other users hog it); a bigger healthy pool frees up
# faster. Poison tasks are safe now (manager HARD_TIMEOUT SIGKILLs them).
EXCLUDE = ",".join([
    "gnho006", "gnho008", "gnho011", "gnho016", "gnho020", "gnho031", "gnho032",
    "gnho041", "gnho033", "gnho034", "gnho035", "gnho036",
])
os.environ["PATH"] = "/soft/slurm/bin:" + os.environ.get("PATH", "")


def ckpts():
    out = {}
    for pt in glob.glob(RUNS + "/*/checkpoints/weights/step_*.pt"):
        m = re.search(r"step_(\d+)\.pt", pt)
        if m:
            out[int(m.group(1))] = pt
    return out


def submitted():
    if not os.path.exists(SUB):
        return set()
    return set(int(x) for x in open(SUB).read().split())


def mark(s):
    open(SUB, "a").write("%d\n" % s)


def eval_active():
    out = subprocess.run(["squeue", "-u", "mazijian", "-h", "-o", "%j"],
                         capture_output=True, text=True).stdout
    return any("eval_apfull" in l for l in out.splitlines())


def stable_size(path):
    try:
        s1 = os.path.getsize(path)
        time.sleep(20)
        return os.path.getsize(path) == s1 and s1 >= MIN_SIZE
    except OSError:
        return False


def log(*a):
    print(time.strftime("%m-%d %H:%M"), *a, flush=True)


log("apfull auto-eval watchdog start; stride", STRIDE, "base", BASE, "excl", EXCLUDE)
if not os.path.exists(SUB):
    mark(BASE)  # 10000 already done

while True:
    try:
        cand = sorted(s for s in ckpts()
                      if s > BASE and s % STRIDE == 0 and s not in submitted())
        if cand and not eval_active():
            step = cand[0]                       # oldest un-evaluated 10k ckpt, in order
            pt = ckpts()[step]
            if not stable_size(pt):
                log("step %d not fully written yet, wait" % step)
                time.sleep(120)
                continue
            exp = pt.split("/checkpoints/")[0]
            env = dict(os.environ)
            env.update({
                "EXP_PATH": exp,
                "EVAL_TRAIN_STEP": "%06d" % step,
                "EVAL_NUM_EPISODES": "10",
                "ROBOTWIN_EPISODE_TIMEOUT_S": "300",
                "ROBOTWIN_TASK_HARD_TIMEOUT_S": "3600",  # SIGKILL only past 60min (give poison tasks time to finish real)
            })
            cmd = ["sbatch", "--parsable", "--time=03:30:00", "--exclude=" + EXCLUDE,
                   "-J", "eval_apfull_%dk" % (step // 1000),
                   "exploration/action_img_patch/sbatch_eval_ap_patch.sh"]
            r = subprocess.run(cmd, cwd=R, env=env, capture_output=True, text=True)
            log("submitted eval step %d job=%s %s"
                % (step, r.stdout.strip(), r.stderr.strip()[:120]))
            mark(step)
        time.sleep(300)
    except Exception as e:
        log("err", repr(e))
        time.sleep(300)
