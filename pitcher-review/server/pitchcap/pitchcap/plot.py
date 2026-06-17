"""Render the kinematic-sequence time-series plot."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_sequence(result, out_path):
    fps = result["fps"]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"pelvis": "tab:blue", "trunk": "tab:green", "shoulder": "tab:red", "elbow": "tab:orange"}
    for name, seg in result["segments"].items():
        series = np.array(seg["series_degps"])
        t = np.arange(len(series)) / fps
        ax.plot(t, series, label=name, color=colors.get(name))
        ax.axvline(seg["peak_time_s"], color=colors.get(name), ls="--", alpha=0.5)
        ax.scatter([seg["peak_time_s"]], [seg["peak_degps"]],
                   color=colors.get(name), zorder=5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angular velocity (deg/s)")
    ax.set_title("Kinematic sequence: " + " -> ".join(result["sequence_order"]))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
