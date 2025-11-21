#!/usr/bin/env python3
"""
Prime-RL multi-node launcher (SLURM) with optional verifiers-style mode.

Default (no --train-cmd, i.e., your current usage):
  - Leader: inference + orchestrator
  - Followers: trainer (torch.distributed.run)
    ✅ Works with your same srun line.

Verifiers-style (when --train-cmd is provided):
  - Leader: inference ONLY, wait for /v1/models
  - Followers: export OPENAI_*/VLLM_* env, wait for readiness, run --train-cmd template
    Placeholders: {MODEL} {PORT} {BASE_URL} {HOST} {BASE_HOST}
"""

import argparse
import os
import shlex
import socket
import subprocess
import sys
import time
from typing import List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


# ------------------------- Utilities ------------------------- #

def which(cmd: str) -> Optional[str]:
    from shutil import which as _which
    return _which(cmd)

def run_cmd(cmd: List[str], env=None, cwd=None, check=False, capture=True) -> subprocess.CompletedProcess:
    print(f"[run] {' '.join(shlex.quote(x) for x in cmd)}")
    return subprocess.run(cmd, text=True, capture_output=capture, env=env, cwd=cwd, check=check)

def expand_slurm_nodelist(nodelist_expr: str) -> List[str]:
    if which("scontrol"):
        out = run_cmd(["scontrol", "show", "hostnames", nodelist_expr])
        if out.returncode == 0:
            nodes = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
            if nodes:
                return nodes
    # Fallback for patterns like node[01-03,05]
    import re
    m = re.match(r"^([^\[]+)\[([^\]]+)\]$", nodelist_expr)
    if not m:
        return [nodelist_expr.strip()]
    prefix, ranges = m.groups()
    out_nodes = []
    for part in ranges.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            width = max(len(a), len(b))
            for i in range(int(a), int(b) + 1):
                out_nodes.append(f"{prefix}{i:0{width}d}")
        else:
            out_nodes.append(f"{prefix}{part}")
    return out_nodes

def get_slurm_nodes() -> List[str]:
    nodelist_expr = os.environ.get("SLURM_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist_expr:
        raise RuntimeError("SLURM_NODELIST/SLURM_JOB_NODELIST not found. Are you running under SLURM?")
    nodes = expand_slurm_nodelist(nodelist_expr)
    if not nodes:
        raise RuntimeError(f"Failed to expand SLURM node list: {nodelist_expr}")
    return nodes

def short_hostname(name: str) -> str:
    return name.split(".", 1)[0]

def _get_ip_via_getent(host: str) -> Optional[str]:
    try:
        res = run_cmd(["getent", "ahosts", host])
        if res.returncode == 0 and res.stdout.strip():
            for line in res.stdout.splitlines():
                parts = line.split()
                if not parts:
                    continue
                ip = parts[0]
                if ip not in ("127.0.0.1", "::1"):
                    return ip
    except Exception:
        pass
    return None

def resolve_ip(host: str) -> str:
    # Honor override first
    override = os.environ.get("INFERENCE_SERVER_IP")
    if override:
        return override
    ip = _get_ip_via_getent(host)
    if ip:
        return ip
    try:
        return socket.gethostbyname(host)
    except socket.gaierror:
        return host  # last resort

def url_host(ip_or_host: str) -> str:
    # bracket IPv6 literal for http://[addr]:port
    return f"[{ip_or_host}]" if (":" in ip_or_host and not ip_or_host.startswith("[")) else ip_or_host

def ensure_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {var}")
    return val

def wait_for_v1_models(base_url: str, timeout_s: int = 900, interval_s: int = 3, tag: str = ""):
    url = f"{base_url}/models"
    start = time.time()
    while True:
        elapsed = int(time.time() - start)
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=5) as resp:
                status = getattr(resp, "status", 200)
                if status == 200:
                    print(f"[ok]{tag} {url} is ready after {elapsed}s")
                    return
        except (URLError, HTTPError):
            pass
        print(f"[wait]{tag} {url} not ready... {elapsed}s")
        if elapsed >= timeout_s:
            print(f"[timeout]{tag} exceeded {timeout_s}s waiting for {url}", file=sys.stderr)
            sys.exit(1)
        time.sleep(interval_s)


# ------------------------- Launchers ------------------------- #

def launch_inference(api_key_env: str, port: int, extra: str) -> subprocess.Popen:
    api_key = ensure_env(api_key_env)
    cmd = [
        sys.executable, "-m", "prime_rl.inference.server",
        "--api-key", api_key,
        "--port", str(port),
    ] + shlex.split(extra)

    print("[leader] inference launched", flush=True)
    return subprocess.Popen(cmd)

def launch_orchestrator(base_url: str, api_key_env: str, output_dir: str, extra: str) -> subprocess.Popen:
    _ = ensure_env(api_key_env)  # orchestrator reads it itself
    cmd = [
        sys.executable, "-m", "prime_rl.orchestrator.orchestrator",
        "--client.base-url", base_url,
        "--client.api-key-var", api_key_env,
        "--output-dir", output_dir,
    ] + shlex.split(extra)
    print(f"[leader] Starting orchestrator: {' '.join(shlex.quote(x) for x in cmd)}")

    print("[leader] orchestrator launched", flush=True)
    return subprocess.Popen(cmd)

def launch_trainer(nproc_per_node: int, output_dir: str, extra: str, local_rank_filter: Optional[str]) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc-per-node", str(int(nproc_per_node)),
        "--module", "prime_rl.trainer.rl.train",
    ]
    if local_rank_filter:
        cmd += ["--local-rank-filter", local_rank_filter]
    cmd += shlex.split(extra)
    cmd += ["--output-dir", output_dir]
    print(f"[trainer] Starting trainer: {' '.join(shlex.quote(x) for x in cmd)}", flush=True)
    return subprocess.Popen(cmd)


# ------------------------- Main ------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Prime-RL multi-node SLURM launcher")
    parser.add_argument("--port", type=int, default=8000, help="Inference server port (default: 8000)")
    parser.add_argument("--api-key-env", type=str, default="INFERENCE_SERVER_API_KEY",
                        help="Env var holding the inference API key")
    parser.add_argument("--output-dir-env", type=str, default="OUTPUT_DIR",
                        help="Env var holding OUTPUT_DIR path")

    # Extra CLI for components (default/legacy path)
    parser.add_argument("--inference-extra", type=str, default="", help="Extra args for inference server")
    parser.add_argument("--orchestrator-extra", type=str, default="", help="Extra args for orchestrator")
    parser.add_argument("--trainer-extra", type=str, default="", help="Extra args for trainer module")
    parser.add_argument("--nproc-per-node", type=int, default=8, help="--nproc-per-node for trainer")
    parser.add_argument("--local-rank-filter", type=str, default="0",
                        help="Forwarded to trainer; use '' to omit")

    # Verifiers-style toggles
    parser.add_argument("--train-cmd", type=str, default="",
                        help="If set, use verifiers-style: leader runs inference only; "
                             "followers wait for readiness then run this template. "
                             "Placeholders: {MODEL} {PORT} {BASE_URL} {HOST} {BASE_HOST}")
    parser.add_argument("--leader-host", type=str, default=None,
                        help="Override leader hostname/IP if needed")

    # Advanced node control
    parser.add_argument("--nodes", type=str, default="", help="Comma-separated node list")
    parser.add_argument("--leader-index", type=int, default=0, help="Leader index in node list")
    args = parser.parse_args()

    # Env checks
    output_dir = ensure_env(args.output_dir_env)
    ensure_env(args.api_key_env)

    # Nodes / roles
    if args.nodes:
        nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    else:
        nodes = get_slurm_nodes()
    if not nodes:
        raise RuntimeError("No nodes detected.")
    if args.leader_index < 0 or args.leader_index >= len(nodes):
        raise RuntimeError(f"leader-index {args.leader_index} out of range for {len(nodes)} nodes")

    leader = nodes[args.leader_index]
    me = short_hostname(socket.gethostname())
    nodes_short = [short_hostname(n) for n in nodes]
    leader_short = short_hostname(leader)

    # Allow explicit override (useful on some SLURM fabrics)
    leader_host_raw = args.leader_host or leader_short
    leader_ip = resolve_ip(leader_host_raw)
    base_host = url_host(leader_ip)                 # bracketed if IPv6
    base_root = f"http://{base_host}:{args.port}"
    base_url  = f"{base_root}/v1"

    print(f"[info] Nodes: {nodes_short}")
    print(f"[info] Leader: {leader_short} (resolved {leader_ip})")
    print(f"[info] Leader URL: {base_url}")
    print(f"[info] This host: {me}")

    is_leader = (me == leader_short)

    procs: List[subprocess.Popen] = []
    try:

            # -------- Default legacy mode (your current behavior) --------
            if is_leader:
                procs.append(launch_inference(args.api_key_env, args.port, args.inference_extra))
                time.sleep(3)  # small cushion
                procs.append(launch_orchestrator(base_url, args.api_key_env, output_dir, args.orchestrator_extra))
                print("############################################", flush=True)
            else:
                lrf = args.local_rank_filter if args.local_rank_filter != "" else None
                procs.append(launch_trainer(args.nproc_per_node, output_dir, args.trainer_extra, lrf))
                print("############################################", flush=True)

            exit_code = 0
            for p in procs:
                rc = p.wait()
                if rc != 0:
                    exit_code = rc
            sys.exit(exit_code)

    except KeyboardInterrupt:
        print("[info] KeyboardInterrupt: terminating child processes...")
        for p in procs:
            try: p.terminate()
            except Exception: pass
        for p in procs:
            try: p.wait(timeout=10)
            except Exception:
                try: p.kill()
                except Exception: pass
        sys.exit(130)


if __name__ == "__main__":
    main()
