# python speed_estimation/plots/comparison/comparison_plot_30_80.py \
#   --data_dir speed_estimation/final_output_smooth \
#   --output comparative_AB_vs_C_30_80.png \
#   --ylim 0 180

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import math

def load_triplet(data_dir: Path, a_file: str, b_file: str, c_file: str, distance: str):
    df_a = pd.read_csv(data_dir / a_file); df_a["series"] = "A"
    df_b = pd.read_csv(data_dir / b_file); df_b["series"] = "B"
    df_c = pd.read_csv(data_dir / c_file); df_c["series"] = "C"
    out = pd.concat([df_a, df_b, df_c], ignore_index=True)
    out["distance"] = distance
    return out

def pick_grid(n):
    if n <= 3:
        return 1, n
    elif n <= 6:
        return 2, 3
    else:
        cols = 3
        rows = math.ceil(n / cols)
        return rows, cols

def main():
    parser = argparse.ArgumentParser(
        description="Boxplots of speed (km/h) for Series A, B, C at 30/40/50/60/70/80 m."
    )
    parser.add_argument("--data_dir", required=True, type=Path,
                        help="Folder with *_final_speeds.csv files.")
    parser.add_argument("--output", default="comparative_AB_vs_C_30_80.png",
                        help="Output image filename (PNG).")
    parser.add_argument("--ylim", type=float, nargs=2, default=None,
                        help="Optional y-limits, e.g. --ylim 0 180")
    args = parser.parse_args()

    file_map = {
        "30m": ("A_30m_final_speeds.csv", "B_30m_final_speeds.csv", "C_135m_final_speeds.csv"),
        "40m": ("A_40m_final_speeds.csv", "B_40m_final_speeds.csv", "C_145m_final_speeds.csv"),
        "50m": ("A_50m_final_speeds.csv", "B_50m_final_speeds.csv", "C_170m_final_speeds.csv"),
        "60m": ("A_60m_final_speeds.csv", "B_60m_final_speeds.csv", "C_190m_final_speeds.csv"),
        "70m": ("A_70m_final_speeds.csv", "B_70m_final_speeds.csv", "C_220m_final_speeds.csv"),
        "80m": ("A_80m_final_speeds.csv", "B_80m_final_speeds.csv", "C_260m_final_speeds.csv"),
    }

    loaded = []
    for dist, (fa, fb, fc) in file_map.items():
        missing = [f for f in (fa, fb, fc) if not (args.data_dir / f).exists()]
        if missing:
            print(f"[WARN] {dist}: missing {missing} — skipping this distance.")
            continue
        loaded.append(load_triplet(args.data_dir, fa, fb, fc, dist))

    if not loaded:
        print("No data loaded. Check --data_dir and filenames.")
        return

    df = pd.concat(loaded, ignore_index=True)
    distances_sorted = [d for d in ["30m","40m","50m","60m","70m","80m"]
                        if d in df["distance"].unique()]

    sns.set_theme(context="talk", style="whitegrid")
    order = ["A", "B", "C"]
    palette = ["#5DA5DA", "#B276B2", "#60BD68"]

    n = len(distances_sorted)
    rows, cols = pick_grid(n)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5.8*rows), sharey=True)
    # Flatten axes for easy indexing
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    fig.suptitle("Speed distributions (A/B/C) at 30–80 m", y=0.98, fontsize=22)

    for ax, dist in zip(axes, distances_sorted):
        sub = df[df["distance"] == dist]
        if sub.empty:
            ax.set_visible(False)
            continue

        sns.boxplot(
            x="series", y="speed_kph", data=sub,
            order=order, palette=palette, ax=ax, width=0.5, fliersize=2
        )

        stats = (sub.groupby("series")["speed_kph"]
                   .agg(median="median", n="count")).reindex(order)

        for i, s in enumerate(order):
            med = stats.loc[s, "median"]
            n_s = stats.loc[s, "n"]
            if pd.isna(med) or pd.isna(n_s):
                continue
            ax.axhline(med, xmin=(i+0.05)/3, xmax=(i+0.95)/3, color="k", lw=1, alpha=0.25)
            ax.text(i, med, f"med={med:.1f}\n(n={int(n_s)})",
                    ha="center", va="bottom", fontsize=11, color="#444",
                    bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="0.85", alpha=0.95))

        ax.set_title(f"Distance: {dist}", fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel("Speed (km/h)")

        if args.ylim:
            ax.set_ylim(args.ylim)

    for j in range(len(distances_sorted), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(args.output, dpi=200)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()