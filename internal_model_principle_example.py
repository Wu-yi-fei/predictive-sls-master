from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HORIZON = 180
A = 0.9
KP = 0.8
OMEGA_TRUE = 0.16
OMEGA_PRED = 0.16
PRED_PHASE_BIAS = 0.08
DISTURBANCE_LEVELS = (0.1, 0.3, 0.6, 1.0)


def exosystem_step(omega: float) -> np.ndarray:
    c, s = np.cos(omega), np.sin(omega)
    return np.array([[c, -s], [s, c]])


def generate_disturbance(amplitude: float, omega: float, phase_bias: float = 0.0) -> np.ndarray:
    """d_t = [1,0] z_t, with z_{t+1} = S z_t."""
    s_mat = exosystem_step(omega)
    z = np.array([phase_bias * amplitude, amplitude], dtype=float)
    d = np.zeros(HORIZON)
    for t in range(HORIZON):
        d[t] = z[0]
        z = s_mat @ z
    return d


def simulate_no_prediction(disturbance: np.ndarray) -> np.ndarray:
    x = np.zeros(HORIZON + 1)
    for t in range(HORIZON):
        u = -KP * x[t]
        x[t + 1] = A * x[t] + u + disturbance[t]
    return x


def simulate_with_prediction(disturbance: np.ndarray, disturbance_pred: np.ndarray) -> np.ndarray:
    x = np.zeros(HORIZON + 1)
    for t in range(HORIZON):
        u = -KP * x[t] - disturbance_pred[t]
        x[t + 1] = A * x[t] + u + disturbance[t]
    return x


def main() -> None:
    t = np.arange(HORIZON + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), constrained_layout=True)

    # Group 1: trajectories at one representative disturbance level.
    amp_ref = DISTURBANCE_LEVELS[-1]
    d_true = generate_disturbance(amp_ref, OMEGA_TRUE)
    d_pred = generate_disturbance(amp_ref, OMEGA_PRED, phase_bias=PRED_PHASE_BIAS)
    x_no_pred = simulate_no_prediction(d_true)
    x_pred = simulate_with_prediction(d_true, d_pred)
    axes[0].plot(t, x_no_pred, lw=2.0, color="#d62728", label="No-prediction policy")
    axes[0].plot(t, x_pred, lw=2.0, color="#1f77b4", label="Prediction policy")
    axes[0].axhline(0.0, color="black", lw=1.0, alpha=0.5)
    axes[0].set_xlabel("time step")
    axes[0].set_ylabel("state x_t")
    axes[0].set_title(f"IMP disturbance response (amplitude={amp_ref:g})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=True, fontsize=9)

    # Group 2: performance vs disturbance level.
    tail = slice(int(0.6 * HORIZON), None)
    no_pred_rms, pred_rms = [], []
    for amp in DISTURBANCE_LEVELS:
        d_true = generate_disturbance(amp, OMEGA_TRUE)
        d_pred = generate_disturbance(amp, OMEGA_PRED, phase_bias=PRED_PHASE_BIAS)
        x_n = simulate_no_prediction(d_true)
        x_p = simulate_with_prediction(d_true, d_pred)
        no_pred_rms.append(float(np.sqrt(np.mean(x_n[tail] ** 2))))
        pred_rms.append(float(np.sqrt(np.mean(x_p[tail] ** 2))))
    axes[1].plot(DISTURBANCE_LEVELS, no_pred_rms, marker="o", lw=2.0, color="#d62728", label="No prediction")
    axes[1].plot(DISTURBANCE_LEVELS, pred_rms, marker="o", lw=2.0, color="#1f77b4", label="With prediction")
    axes[1].set_xlabel("IMP disturbance amplitude")
    axes[1].set_ylabel(r"tail RMS of |x_t|")
    axes[1].set_title("Performance vs IMP-generated disturbance")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=True, fontsize=9)

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "pic" / "review_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "internal_model_principle_case_demo.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
