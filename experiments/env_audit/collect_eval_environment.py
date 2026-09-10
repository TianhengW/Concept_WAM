#!/usr/bin/env python3
"""Create an additive, comparable JSON snapshot of an evaluation environment.

The collector intentionally records provenance and evidence without modifying the
environment being inspected.  It supports Kubernetes-backed runs now and keeps
the same top-level schema usable for a later Slurm/H800 snapshot.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "eval-environment-snapshot/v1"
SENSITIVE_NAME = re.compile(
    r"(?:TOKEN|PASSWORD|PASSWD|SECRET|PRIVATE|CREDENTIAL|AUTH|COOKIE|KEY)$",
    re.IGNORECASE,
)
ENV_ALLOWLIST = {
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HYDRA_FULL_ERROR",
    "LD_LIBRARY_PATH",
    "LIBERO_CONFIG_PATH",
    "MUJOCO_GL",
    "NCCL_DEBUG",
    "NCCL_IB_DISABLE",
    "NCCL_SOCKET_IFNAME",
    "PATH",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "TRANSFORMERS_OFFLINE",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run(command: list[str], timeout: int = 30, cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except FileNotFoundError as exc:
        return {"command": command, "error": f"not_found: {exc}"}
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "error": f"timeout_after_{timeout}s",
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, hash_max_bytes: int) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path)}
    try:
        stat = path.lstat()
    except OSError as exc:
        record.update({"exists": False, "error": str(exc)})
        return record

    record.update(
        {
            "exists": True,
            "type": "symlink" if path.is_symlink() else "directory" if path.is_dir() else "file",
            "size_bytes": stat.st_size,
            "mtime_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
            "mode_octal": oct(stat.st_mode & 0o7777),
        }
    )
    if path.is_symlink():
        try:
            record["symlink_target"] = os.readlink(path)
        except OSError as exc:
            record["symlink_error"] = str(exc)
    if path.is_file():
        if stat.st_size <= hash_max_bytes:
            try:
                record["sha256"] = sha256_file(path)
                record["sha256_status"] = "computed_now"
            except OSError as exc:
                record["sha256_status"] = "error"
                record["sha256_error"] = str(exc)
        else:
            record["sha256_status"] = "skipped_size_limit"
            record["hash_max_bytes"] = hash_max_bytes
    return record


def directory_summary(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_dir()}
    if not path.is_dir():
        return result
    counts: collections.Counter[str] = collections.Counter()
    files = 0
    dirs = 0
    symlinks = 0
    total_bytes = 0
    errors: list[str] = []
    try:
        for root, dirnames, filenames in os.walk(path):
            dirs += len(dirnames)
            for name in filenames:
                candidate = Path(root) / name
                try:
                    stat = candidate.lstat()
                    files += 1
                    total_bytes += stat.st_size
                    symlinks += int(candidate.is_symlink())
                    counts[candidate.suffix.lower() or "<no_extension>"] += 1
                except OSError as exc:
                    errors.append(f"{candidate}: {exc}")
    except OSError as exc:
        errors.append(str(exc))
    result.update(
        {
            "file_count": files,
            "directory_count": dirs,
            "symlink_count": symlinks,
            "total_file_bytes": total_bytes,
            "extension_counts": dict(sorted(counts.items())),
            "errors": errors[:100],
        }
    )
    return result


def parse_json_command(command: list[str], timeout: int = 30) -> dict[str, Any]:
    evidence = run(command, timeout=timeout)
    if evidence.get("returncode") == 0:
        try:
            evidence["json"] = json.loads(evidence.get("stdout", ""))
            evidence.pop("stdout", None)
        except json.JSONDecodeError as exc:
            evidence["json_error"] = str(exc)
    return evidence


def clean_metadata(obj: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(obj.get("metadata") or {})
    metadata.pop("managedFields", None)
    return metadata


def kubernetes_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    if not args.kube_job:
        return {"status": "not_requested"}
    base = ["kubectl", "-n", args.kube_namespace, "get"]
    job = parse_json_command(base + ["vcjob", args.kube_job, "-o", "json"], timeout=45)
    pod_name = args.kube_pod or f"{args.kube_job}-evaluator-0"
    pod = parse_json_command(base + ["pod", pod_name, "-o", "json"], timeout=45)

    result: dict[str, Any] = {
        "namespace": args.kube_namespace,
        "job_name": args.kube_job,
        "pod_name": pod_name,
        "job_query": job,
        "pod_query": pod,
        "configmaps": {},
    }
    pod_json = pod.get("json") or {}
    node_name = (pod_json.get("spec") or {}).get("nodeName")
    result["node_name"] = node_name
    if node_name:
        node = parse_json_command(["kubectl", "get", "node", node_name, "-o", "json"], timeout=45)
        node_json = node.get("json") or {}
        if node_json:
            labels = node_json.get("metadata", {}).get("labels", {})
            stable_labels = {
                key: value
                for key, value in labels.items()
                if key.startswith("nvidia.com/")
                or key.startswith("feature.node.kubernetes.io/system-os")
                or key.startswith("feature.node.kubernetes.io/kernel-version")
                or key in {
                    "cloud.d-robotics.cc/queue-name",
                    "cloud.d-robotics.cc/node-type",
                    "kubernetes.io/arch",
                    "kubernetes.io/os",
                }
            }
            result["node"] = {
                "metadata": {"name": node_json.get("metadata", {}).get("name")},
                "labels": stable_labels,
                "capacity": node_json.get("status", {}).get("capacity", {}),
                "allocatable": node_json.get("status", {}).get("allocatable", {}),
                "node_info": node_json.get("status", {}).get("nodeInfo", {}),
            }
        else:
            result["node_query"] = node

    for name in args.kube_configmap:
        cm = parse_json_command(base + ["configmap", name, "-o", "json"], timeout=45)
        cm_json = cm.get("json") or {}
        if cm_json:
            result["configmaps"][name] = {
                "metadata": clean_metadata(cm_json),
                "data": cm_json.get("data", {}),
                "binaryData_keys": sorted((cm_json.get("binaryData") or {}).keys()),
            }
        else:
            result["configmaps"][name] = cm

    # Remove noisy server-managed fields while preserving the exact executable spec/status.
    for key in ("job_query", "pod_query"):
        obj = result[key].get("json")
        if obj:
            result[key]["json"] = {
                "apiVersion": obj.get("apiVersion"),
                "kind": obj.get("kind"),
                "metadata": clean_metadata(obj),
                "spec": obj.get("spec"),
                "status": obj.get("status"),
            }
    return result


def slurm_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    if not args.slurm_job:
        return {"status": "not_requested"}
    return {
        "job_id": args.slurm_job,
        "scontrol_job": run(["scontrol", "show", "job", "-dd", args.slurm_job], timeout=30),
        "sacct": run(
            [
                "sacct",
                "-j",
                args.slurm_job,
                "--parsable2",
                "--format=JobID,JobName,Partition,State,Elapsed,AllocTRES,NodeList,ExitCode",
            ],
            timeout=30,
        ),
    }


def git_snapshot(repo: Path | None) -> dict[str, Any]:
    if repo is None:
        return {"status": "not_requested"}
    return {
        "path": str(repo),
        "head": run(["git", "rev-parse", "HEAD"], cwd=repo),
        "branch": run(["git", "branch", "--show-current"], cwd=repo),
        "status_short": run(["git", "status", "--short"], cwd=repo),
        "remote_get_url": run(["git", "remote", "get-url", "origin"], cwd=repo),
    }


def python_snapshot(python_executable: str | None) -> dict[str, Any]:
    if not python_executable:
        return {"status": "not_requested"}
    probe = (
        "import importlib, json, platform, sys; "
        "names=['torch','transformers','safetensors','mujoco','numpy','hydra','omegaconf',"
        "'accelerate','deepspeed','libero','robosuite']; out={}; "
        "\nfor n in names:\n"
        " try:\n  m=importlib.import_module(n); out[n]={'version':getattr(m,'__version__',None),"
        "'file':getattr(m,'__file__',None),'path':list(getattr(m,'__path__',[]))}\n"
        " except Exception as e: out[n]={'error':repr(e)}\n"
        "print(json.dumps({'python':sys.version,'executable':sys.executable,'platform':platform.platform(),"
        "'modules':out},ensure_ascii=False))"
    )
    result = parse_json_command([python_executable, "-c", probe], timeout=120)
    result["pip_freeze"] = run([python_executable, "-m", "pip", "freeze", "--all"], timeout=120)
    return result


def log_snapshot(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_requested"}
    result = file_record(path, hash_max_bytes=512 * 1024 * 1024)
    if not path.is_file():
        return result
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        result["line_count"] = len(lines)
        result["head"] = "".join(lines[:160])
        result["tail"] = "".join(lines[-160:])
        patterns = re.compile(
            r"(?:NVIDIA-SMI|Driver Version|CUDA Version|mujoco=|torch=|transformers=|"
            r"safetensors=|libero=|checkpoint|dataset stats|action dim|Loaded MoT|"
            r"success rate|success=|completed)",
            re.IGNORECASE,
        )
        matches = [line.rstrip("\n") for line in lines if patterns.search(line)]
        result["selected_evidence_lines"] = matches[:2000]
    except OSError as exc:
        result["excerpt_error"] = str(exc)
    return result


def parse_facts(values: list[str]) -> dict[str, str]:
    facts: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--fact must be KEY=VALUE, got {value!r}")
        key, item = value.split("=", 1)
        facts[key] = item
    return facts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--platform-label", required=True)
    parser.add_argument("--repo")
    parser.add_argument("--python")
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--dir", action="append", default=[])
    parser.add_argument("--log")
    parser.add_argument("--fact", action="append", default=[])
    parser.add_argument("--hash-max-bytes", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--kube-namespace", default="dreamzero-training")
    parser.add_argument("--kube-job")
    parser.add_argument("--kube-pod")
    parser.add_argument("--kube-configmap", action="append", default=[])
    parser.add_argument("--slurm-job")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    filtered_env = {
        key: value
        for key, value in os.environ.items()
        if key in ENV_ALLOWLIST and not SENSITIVE_NAME.search(key)
    }
    repo = Path(args.repo).expanduser().resolve() if args.repo else None
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "capture": {
            "created_at_utc": utc_now(),
            "platform_label": args.platform_label,
            "collector_path": str(Path(__file__).resolve()),
            "collector_sha256": sha256_file(Path(__file__).resolve()),
            "capture_host": socket.gethostname(),
            "capture_user": os.environ.get("USER") or os.environ.get("USERNAME"),
            "python": sys.version,
            "limitations": [
                "A completed container cannot be entered; runtime package evidence may come from immutable image/config and its recorded launch log.",
                "Files larger than hash_max_bytes are recorded by path, size and mtime unless an existing checksum manifest is supplied separately.",
                "Only allowlisted non-secret environment variables are recorded.",
            ],
        },
        "evaluation_facts": parse_facts(args.fact),
        "capture_host_environment": {
            "platform": platform.platform(),
            "uname": platform.uname()._asdict(),
            "allowlisted_environment": filtered_env,
            "os_release": run(["sh", "-c", "cat /etc/os-release 2>/dev/null || true"]),
        },
        "git": git_snapshot(repo),
        "python_environment": python_snapshot(args.python),
        "scheduler": {
            "kubernetes": kubernetes_snapshot(args),
            "slurm": slurm_snapshot(args),
        },
        "files": [file_record(Path(item), args.hash_max_bytes) for item in args.path],
        "directories": [directory_summary(Path(item)) for item in args.dir],
        "evaluation_log": log_snapshot(Path(args.log) if args.log else None),
    }

    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(snapshot, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output)
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size, "sha256": sha256_file(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
