import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
CSV_FILES = [
    ("seed_13", "topk_retrain/trial_54/seed_13/epoch_log.csv"),
    ("seed_37", "topk_retrain/trial_54/seed_37/epoch_log.csv"),
    ("seed_71", "topk_retrain/trial_54/seed_71/epoch_log.csv"),
]

BASE_DIR = Path("seg_phase5")
OUT_DIR = BASE_DIR / "plots"

OUT_NAME_PNG = "Figure_S4_training_dynamics_loss_only.png"
OUT_NAME_PDF = "Figure_S4_training_dynamics_loss_only.pdf"

DPI = 300

# Global figure style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
})

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def load_and_tag(csv_path: Path, seed_label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"epoch", "train_loss", "val_loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["seed"] = seed_label
    return df[["epoch", "train_loss", "val_loss", "seed"]]


def summarize_across_seeds(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = (
        df.groupby("epoch")[value_col]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": f"{value_col}_mean", "std": f"{value_col}_std"})
    )
    out[f"{value_col}_std"] = out[f"{value_col}_std"].fillna(0.0)
    return out


def style_axis(ax, title: str, ylabel: str = "Loss") -> None:
    ax.set_title(title, pad=8, fontweight="bold", fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.tick_params(axis="both", width=1.0, length=5)


def plot_seed_lines(ax, df: pd.DataFrame, value_col: str) -> None:
    for seed_name, sdf in df.groupby("seed"):
        sdf = sdf.sort_values("epoch")
        ax.plot(
            sdf["epoch"].to_numpy(),
            sdf[value_col].to_numpy(),
            linewidth=1.0,
            alpha=0.35,
        )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for seed_label, rel_path in CSV_FILES:
        path = BASE_DIR / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Could not find: {path}")
        frames.append(load_and_tag(path, seed_label))

    all_df = pd.concat(frames, ignore_index=True)

    train_stats = summarize_across_seeds(all_df, "train_loss")
    val_stats = summarize_across_seeds(all_df, "val_loss")

    fig, axes = plt.subplots(2, 1, figsize=(6, 6))

    # (a) Training loss
    ax = axes[0]
    plot_seed_lines(ax, all_df, "train_loss")

    x = train_stats["epoch"].to_numpy()
    y = train_stats["train_loss_mean"].to_numpy()
    s = train_stats["train_loss_std"].to_numpy()

    ax.plot(x, y, linewidth=2.4, label="Mean across seeds")
    ax.fill_between(x, y - s, y + s, alpha=0.22, label="± SD")
    style_axis(ax, "(a) Training loss")
    ax.legend(frameon=False, loc="upper right")

    # (b) Validation loss
    ax = axes[1]
    plot_seed_lines(ax, all_df, "val_loss")

    x = val_stats["epoch"].to_numpy()
    y = val_stats["val_loss_mean"].to_numpy()
    s = val_stats["val_loss_std"].to_numpy()

    ax.plot(x, y, linewidth=2.4, label="Mean across seeds")
    ax.fill_between(x, y - s, y + s, alpha=0.22, label="± SD")
    style_axis(ax, "(b) Validation loss")
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()

    png_path = OUT_DIR / OUT_NAME_PNG
    pdf_path = OUT_DIR / OUT_NAME_PDF

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()