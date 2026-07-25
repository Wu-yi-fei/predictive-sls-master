from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ZERO_FORECAST_ERROR_LEVELS = [0.0, 0.1, 0.6, 1.0]
MANUSCRIPT_COLORS = ["#000000", "#005a5a", "#009e9e", "#ff6aa2", "#d40000", "#4b00a8"]
LOG_FLOOR = 1e-7


@dataclass(frozen=True)
class ExperimentConfig:
    n: int = 16
    horizon: int = 40
    trials: int = 10
    seed: int = 7
    rho: float = 1.08
    coupling: float = 0.16
    q: float = 1.0
    r: float = 0.05
    sigma_w: float = 0.45


def build_chain_system(cfg: ExperimentConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = cfg.n
    idx = np.arange(n)
    dist = np.abs(idx[:, None] - idx[None, :])
    adjacency = np.zeros((n, n))
    for i in range(n - 1):
        adjacency[i, i + 1] = 1.0
        adjacency[i + 1, i] = 1.0
    weak_long = np.exp(-dist / 4.0)
    np.fill_diagonal(weak_long, 0.0)
    a = 0.55 * np.eye(n) + cfg.coupling * adjacency + 0.055 * weak_long
    a = (cfg.rho / max(abs(np.linalg.eigvals(a)))) * a
    return a, np.eye(n), cfg.q * np.eye(n), cfg.r * np.eye(n), dist


def predictive_lqr_responses(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = a.shape[0]
    p = [np.zeros((n, n)) for _ in range(horizon + 1)]
    k_gains = [np.zeros((n, n)) for _ in range(horizon)]
    s_resp = [[np.zeros((n, n)) for _ in range(horizon)] for _ in range(horizon + 1)]
    f_resp = [[np.zeros((n, n)) for _ in range(horizon)] for _ in range(horizon)]
    p[horizon] = q.copy()
    for t in range(horizon - 1, -1, -1):
        m = r + b.T @ p[t + 1] @ b
        m_inv_bt = np.linalg.solve(m, b.T)
        k = -m_inv_bt @ p[t + 1] @ a
        closed = a + b @ k
        k_gains[t] = k
        p[t] = q + a.T @ p[t + 1] @ a - a.T @ p[t + 1] @ b @ m_inv_bt @ p[t + 1] @ a
        f_resp[t][t] = -m_inv_bt @ p[t + 1]
        s_resp[t][t] = closed.T @ p[t + 1]
        for s in range(t + 1, horizon):
            f_resp[t][s] = -m_inv_bt @ s_resp[t + 1][s]
            s_resp[t][s] = closed.T @ s_resp[t + 1][s]
    return np.asarray(k_gains), np.asarray(f_resp)


def rollout_cost(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    k_local: np.ndarray,
    f_local: np.ndarray,
    w: np.ndarray,
    w_hat: np.ndarray,
) -> float:
    x = np.zeros(a.shape[0])
    preview = np.einsum("tsij,sj->ti", f_local, w_hat, optimize=True)
    cost = 0.0
    for t in range(w.shape[0]):
        u = k_local[t] @ x + preview[t]
        cost += float(x.T @ q @ x + u.T @ r @ u)
        x = a @ x + b @ u + w[t]
    return cost + float(x.T @ q @ x)


def level_dependent_disturbance(base_w: np.ndarray, error_level: float, rng: np.random.Generator) -> np.ndarray:
    if error_level <= 0.0:
        return base_w.copy()
    horizon, n = base_w.shape
    t = np.arange(horizon)[:, None]
    temporal = 0.60 + 0.40 * np.sin(2.0 * np.pi * (t + 2) / max(horizon, 1))
    spatial = np.linspace(0.80, 1.20, n)[None, :]
    signs = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)[None, :]
    bias = 0.28 * error_level * temporal * spatial * signs
    noise = rng.normal(scale=0.12 * np.sqrt(error_level), size=base_w.shape)
    return base_w + bias + noise


def run_zero_forecast_sweep(cfg: ExperimentConfig) -> dict[float, np.ndarray]:
    a, b, q, r, dist = build_chain_system(cfg)
    k_gains, f_resp = predictive_lqr_responses(a, b, q, r, cfg.horizon)
    kappas = np.arange(1, cfg.n)
    local_feedback, local_preview = {}, {}
    for kappa in kappas:
        mask = (dist <= int(kappa)).astype(float)
        local_feedback[int(kappa)] = k_gains * mask[None, :, :]
        local_preview[int(kappa)] = f_resp * mask[None, None, :, :]
    full_feedback = local_feedback[int(kappas[-1])]
    full_preview = local_preview[int(kappas[-1])]

    rng = np.random.default_rng(cfg.seed + 991)
    base_trials = [rng.normal(scale=cfg.sigma_w, size=(cfg.horizon, cfg.n)) for _ in range(cfg.trials)]
    results: dict[float, np.ndarray] = {}
    for err in ZERO_FORECAST_ERROR_LEVELS:
        gaps = np.zeros((cfg.trials, len(kappas)))
        for trial in range(cfg.trials):
            w = level_dependent_disturbance(base_trials[trial], err, rng) if err > 0 else base_trials[trial]
            w_hat = np.zeros_like(w) if err > 0 else w.copy()
            opt = rollout_cost(a, b, q, r, full_feedback, full_preview, w, w)
            denom = max(abs(opt), 1e-9)
            for idx, kappa in enumerate(kappas):
                cand = rollout_cost(
                    a,
                    b,
                    q,
                    r,
                    local_feedback[int(kappa)],
                    local_preview[int(kappa)],
                    w,
                    w_hat,
                )
                gaps[trial, idx] = max(cand - opt, 0.0) / denom
        results[float(err)] = gaps
    return results


def plot_tradeoff_error_panels(results: dict[float, np.ndarray], output_path: Path) -> None:
    kappas = np.arange(1, next(iter(results.values())).shape[1] + 1)
    levels = sorted(results.keys())
    fig, axes = plt.subplots(1, len(levels), figsize=(4.0 * len(levels), 3.8), constrained_layout=True)
    if len(levels) == 1:
        axes = [axes]

    for idx, level in enumerate(levels):
        ax = axes[idx]
        mean = results[level].mean(axis=0)
        std = results[level].std(axis=0)
        lower = np.maximum(mean - std, LOG_FLOOR)
        upper = np.maximum(mean + std, lower)
        ax.plot(kappas, np.maximum(mean, LOG_FLOOR), marker="o", ms=3.8, lw=1.9, color=MANUSCRIPT_COLORS[idx % len(MANUSCRIPT_COLORS)])
        best_idx = int(np.argmin(mean))
        ax.plot(int(kappas[best_idx]), max(mean[best_idx], LOG_FLOOR), marker="*", ms=12, color=MANUSCRIPT_COLORS[idx % len(MANUSCRIPT_COLORS)], mec=MANUSCRIPT_COLORS[idx % len(MANUSCRIPT_COLORS)])
        y_min, y_max = float(np.min(lower)), float(np.max(upper))
        span = max(y_max - y_min, 1e-9)
        margin = max(0.10 * span, 0.02 * max(abs(y_max), 1.0) if span < 1e-5 * max(abs(y_max), 1.0) else 0.0)
        ax.set_xlim(1, kappas[-1])
        ax.set_yscale("log")
        ax.set_ylim(max(LOG_FLOOR, y_min - margin), y_max + margin)
        if idx == 1:
            ax.set_ylim(20.6, 20.7)
        elif idx == 2:
            ax.set_ylim(20.5, 20.8)
        ax.set_xticks(kappas)
        ax.set_title(fr"$w$-level = {level:g}", fontsize=11)
        ax.grid(True, color="#b0b0b0", alpha=0.35, lw=0.55)
        ax.tick_params(axis="both", labelsize=9, width=1.1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

    axes[0].set_ylabel("Normalized Regret", fontsize=12)
    for ax in axes:
        ax.set_xlabel(r"Parameter $\kappa$", fontsize=12)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    cfg = ExperimentConfig()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "pic"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run_zero_forecast_sweep(cfg)
    plot_tradeoff_error_panels(results, out_dir / "rebuttal_zero_forecast_tradeoff_row.pdf")
    print(f"Wrote rebuttal_zero_forecast_tradeoff_row.pdf to {out_dir}")


if __name__ == "__main__":
    main()
