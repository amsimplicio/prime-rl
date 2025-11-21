import argparse
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def strip_ansi(s: str) -> str:
    """Remove ANSI color codes from a string."""
    return ANSI_ESCAPE_RE.sub("", s)


def parse_metrics_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single log line of the form:
      '... Step 1500 | Time: 134.09s | Reward: 0.8529 | Throughput: 1859.1 tokens/s | Seq. Length: 1455.6 tokens/sample'
      '... Step 0 | Time: 348.53s | Loss: 0.0019 | Entropy: 0.2105 | Mismatch KL: 0.0006 | Grad. Norm: 0.0680 | ...'
    Returns dict with at least 'step' and any numeric metrics found.
    """
    clean = strip_ansi(line)

    # Require a "Step X" to even consider it
    # Use [Ss]tep just in case the capitalization changes
    m_step = re.search(r"[Ss]tep\s+(\d+)", clean)
    if not m_step:
        return None

    step = float(m_step.group(1))
    item: Dict[str, Any] = {"step": step}

    # Generic "Name: value" matcher for numeric values
    # Name can have spaces or dots ("Seq. Length", "Grad. Norm", etc.)
    kv_pairs = re.findall(
        r"([A-Za-z.\s]+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|[-+]?\d+)",
        clean,
    )

    # Optional aliasing for nicer / more standard column names
    alias_map = {
        "mismatch_kl": "kl",          # "Mismatch KL" -> kl
        "seq_length": "seq_length",   # keep as is, but defined explicitly for clarity
        "grad_norm": "grad_norm",
        "entropy": "entropy",
        "reward": "reward",
        "loss": "loss",
    }

    for raw_name, val_str in kv_pairs:
        name_norm = raw_name.strip().lower()
        # turn "Seq. Length" -> "seq_length", "Grad. Norm" -> "grad_norm"
        auto_key = re.sub(r"[\s\.]+", "_", name_norm)

        key = alias_map.get(auto_key, auto_key)

        try:
            v = float(val_str)
        except ValueError:
            continue

        item[key] = v

    # If we only have step and nothing else, skip
    if len(item) <= 1:
        return None
    return item


def parse_log_file(path: Path) -> List[Dict[str, Any]]:
    """Parse a single log file and return a list of metric dicts."""
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        print(f"[WARN] Log file not found: {path}")
        return rows

    print(f"[INFO] Parsing {path}")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            item = parse_metrics_line(line)
            if item is not None:
                rows.append(item)
    print(f"[INFO] Parsed {len(rows)} metric lines from {path}")
    return rows


def combine_logs(logdir: Path) -> pd.DataFrame:
    """
    Parse orchestrator.log and trainer/rank_0.log from the given directory,
    and merge them by step, combining metrics from both.
    """
    orch_path = logdir / "orchestrator.log"
    trainer_path = logdir / "trainer" / "rank_0.log"

    rows: List[Dict[str, Any]] = []
    rows.extend(parse_log_file(orch_path))
    rows.extend(parse_log_file(trainer_path))

    if not rows:
        raise RuntimeError(
            f"No metrics found in logs under {logdir}. "
            f"Expected at least one of {orch_path} or {trainer_path}."
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("step").reset_index(drop=True)

    # Group by step and keep the last non-NaN value for each metric
    def last_non_nan(series: pd.Series):
        non_nan = series.dropna()
        return non_nan.iloc[-1] if not non_nan.empty else float("nan")

    df = (
        df.groupby("step", as_index=False)
        .agg(last_non_nan)
        .sort_values("step")
        .reset_index(drop=True)
    )

    # Ensure numeric types where possible
    for col in df.columns:
        if col == "step":
            df[col] = df[col].astype(float)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def rolling(series: pd.Series, window: int = 50) -> pd.Series:
    """Rolling mean with some safety."""
    try:
        return series.rolling(
            window=window,
            min_periods=max(5, window // 5),
            center=False
        ).mean()
    except Exception:
        return series


def plot_series(x, y, y_smooth, title: str, ylabel: str, out_path: Path, ylim=None):
    plt.figure()
    plt.plot(x, y, label="raw", alpha=0.6)
    if y_smooth is not None:
        plt.plot(x, y_smooth, label="rolling mean", linewidth=2)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_dashboard(
    df: pd.DataFrame,
    metrics: List[str],
    window: int,
    out_path: Path,
):
    """
    Create a multi-panel dashboard of arbitrary metrics.
    - metrics: list of column names to plot (e.g. ['reward', 'loss', 'kl', 'seq_length'])
    The layout is 2 columns, rows computed automatically.
    """
    metric_configs = {
        "reward": {
            "title": "Reward over Steps",
            "ylabel": "reward",
            "ylim": None,
        },
        "loss": {
            "title": "Loss over Steps",
            "ylabel": "loss",
            "ylim": None,
        },
        "kl": {
            "title": "KL Divergence over Steps",
            "ylabel": "KL",
            "ylim": None,
        },
        "seq_length": {
            "title": "Sequence Length over Steps",
            "ylabel": "seq length (tokens/sample)",
            "ylim": None,
        },
        "grad_norm": {
            "title": "Grad Norm over Steps",
            "ylabel": "grad norm",
            "ylim": None,
        },
        "entropy": {
            "title": "Entropy over Steps",
            "ylabel": "entropy",
            "ylim": None,
        },
    }

    # Filter out metrics that don't exist in df or are all NaN
    metrics = [
        m for m in metrics
        if m in df.columns and df[m].notna().any()
    ]
    if not metrics:
        print("[WARN] No valid metrics for dashboard.")
        return

    n = len(metrics)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in row]

    for i, metric in enumerate(metrics):
        ax = axes_flat[i]
        y = df[metric]
        y_smooth = rolling(y, window)

        cfg = metric_configs.get(
            metric,
            {
                "title": f"{metric} over Steps",
                "ylabel": metric,
                "ylim": None,
            },
        )

        ax.plot(df["step"], y, label="raw", alpha=0.6)
        if y_smooth is not None:
            ax.plot(df["step"], y_smooth, label="rolling mean", linewidth=2)
        ax.set_title(cfg["title"])
        ax.set_xlabel("step")
        ax.set_ylabel(cfg["ylabel"])
        if cfg["ylim"] is not None:
            ax.set_ylim(cfg["ylim"])
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Turn off any extra axes
    for j in range(len(metrics), len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout(pad=3.0)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Parse orchestrator.log and trainer/rank_0.log from a folder, "
            "merge metrics by step, and plot dashboard."
        )
    )
    ap.add_argument(
        "--logdir",
        required=True,
        help="Folder containing orchestrator.log and trainer/rank_0.log",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory for CSV and plots (default: <logdir>/plots)",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=50,
        help="Rolling window size for smoothing",
    )
    ap.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Maximum step to include in plots (e.g. 1500)",
    )
    ap.add_argument(
        "--metrics",
        type=str,
        default="reward,loss,kl,seq_length",
        help=(
            "Comma-separated list of metrics for the DASHBOARD only. "
            "Single-metric PNGs for reward,loss,kl,seq_length,grad_norm,entropy "
            "are always generated when present. "
            "Defaults to: reward,loss,kl,seq_length."
        ),
    )

    args = ap.parse_args()

    logdir = Path(args.logdir)
    outdir = Path(args.outdir) if args.outdir is not None else logdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    df = combine_logs(logdir)

    # Optional filtering by step
    if args.max_step is not None:
        original_len = len(df)
        df = df[df["step"] <= args.max_step].copy()
        print(
            f"[INFO] Filtered data to steps <= {args.max_step}: "
            f"{original_len} -> {len(df)} rows"
        )

    # Save CSV with all metrics
    csv_path = outdir / "parsed_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved metrics CSV to {csv_path}")

    # --- Single-metric PNGs for the 6 core metrics (if columns exist & not all NaN) ---
    single_metrics = {
        "reward": {
            "title": "Reward over Steps",
            "ylabel": "reward",
            "filename": "reward.png",
            "ylim": None,
        },
        "loss": {
            "title": "Loss over Steps",
            "ylabel": "loss",
            "filename": "loss.png",
            "ylim": None,
        },
        "kl": {
            "title": "KL Divergence over Steps",
            "ylabel": "KL",
            "filename": "kl.png",
            "ylim": None,
        },
        "seq_length": {
            "title": "Sequence Length over Steps",
            "ylabel": "seq length (tokens/sample)",
            "filename": "seq_length.png",
            "ylim": None,
        },
        "grad_norm": {
            "title": "Grad Norm over Steps",
            "ylabel": "grad norm",
            "filename": "grad_norm.png",
            "ylim": None,
        },
        "entropy": {
            "title": "Entropy over Steps",
            "ylabel": "entropy",
            "filename": "entropy.png",
            "ylim": None,
        },
    }

    for metric, cfg in single_metrics.items():
        if metric in df and df[metric].notna().any():
            out_path = outdir / cfg["filename"]
            plot_series(
                df["step"],
                df[metric],
                rolling(df[metric], args.window),
                cfg["title"],
                cfg["ylabel"],
                out_path,
                ylim=cfg["ylim"],
            )
            print(f"[INFO] Saved {metric} plot to {out_path}")

    # --- Dashboard (multi-panel) with configurable metrics (ONLY controlled by --metrics) ---
    metrics_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
    dashboard_path = outdir / "dashboard.png"
    plot_dashboard(df, metrics_list, args.window, dashboard_path)

    print(f"[INFO] Saved plots to {outdir}")
    print(f"- CSV: {csv_path}")
    print(f"- Dashboard: {dashboard_path}")
    print(f"- Individual metric plots for available metrics (reward, loss, kl, seq_length, grad_norm, entropy).")


if __name__ == "__main__":
    main()
