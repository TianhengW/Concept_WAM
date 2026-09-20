#!/usr/bin/env python3
"""Merge summary_worker*.json of one eval run into summary.json and print a per-task table."""
import json, sys
from pathlib import Path

run_dir = Path(sys.argv[1])
parts = sorted(run_dir.glob("summary_worker*.json"))
tasks, meta = {}, {}
for p in parts:
    s = json.loads(p.read_text())
    tasks.update(s["tasks"])
    meta = {k: s[k] for k in ("split", "task_set", "replan_steps", "seed")}
n_ep = sum(t["num_episodes"] for t in tasks.values())
n_ok = sum(t["successes"] for t in tasks.values())
summary = dict(meta, num_workers=len(parts), num_tasks=len(tasks), num_episodes=n_ep, successes=n_ok,
               episode_success_rate=(n_ok / n_ep) if n_ep else None,
               mean_task_success_rate=(sum(t["success_rate"] for t in tasks.values()) / len(tasks)) if tasks else None,
               tasks=tasks)
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"{'task':40s} {'succ':>5s}/{'n':<4s} {'SR':>6s}  horizon  aborted")
for name, t in sorted(tasks.items()):
    print(f"{name:40s} {t['successes']:5d}/{t['num_episodes']:<4d} {100*t['success_rate']:5.1f}%  {t['horizon']:7d}  {t.get('aborted', 0)}")
print(f"\nepisodes {n_ok}/{n_ep} = {100*(n_ok/n_ep if n_ep else 0):.1f}%   mean task SR = {100*(summary['mean_task_success_rate'] or 0):.1f}%   ({len(tasks)} tasks, {len(parts)} workers)")
