"""Plot training loss curves from TensorBoard event files."""

import glob
import os
from collections import defaultdict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("tensorboard not found, trying tbparse...")

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def extract_tb_data(logdir):
    """Extract scalar data from TensorBoard logs."""
    # Find event files
    event_files = glob.glob(
        os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True
    )

    if not event_files:
        print(f"No event files found in {logdir}")
        return {}

    print(f"Found {len(event_files)} event file(s)")

    all_data = defaultdict(lambda: {"steps": [], "values": []})

    for ef_path in event_files:
        ef_dir = os.path.dirname(ef_path)
        print(f"Reading: {ef_dir}")

        ea = EventAccumulator(ef_dir)
        ea.Reload()

        tags = ea.Tags().get("scalars", [])
        print(f"  Tags: {tags}")

        for tag in tags:
            events = ea.Scalars(tag)
            for e in events:
                all_data[tag]["steps"].append(e.step)
                all_data[tag]["values"].append(e.value)

    return dict(all_data)


def plot_losses(data, output_path):
    """Create a multi-panel loss plot."""
    # Group metrics - include val metrics too
    loss_tags = [
        t
        for t in data
        if "loss" in t.lower() and ("epoch" in t.lower() or t.startswith("val_"))
    ]
    acc_tags = [
        t
        for t in data
        if "acc" in t.lower() and ("epoch" in t.lower() or t.startswith("val_"))
    ]

    # Remove non-interesting tags
    loss_tags = [t for t in loss_tags if t not in ("hp_metric",)]

    # Fallback to step-level if no epoch-level
    if not loss_tags:
        loss_tags = [t for t in data if "loss" in t.lower()]
    if not acc_tags:
        acc_tags = [t for t in data if "acc" in t.lower()]

    print(f"\nLoss tags: {loss_tags}")
    print(f"Acc tags: {acc_tags}")

    n_plots = 0
    if loss_tags:
        n_plots += 1
    if acc_tags:
        n_plots += 1
    if n_plots == 0:
        print("No metrics found!")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 5 * n_plots))
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0

    if loss_tags:
        ax = axes[ax_idx]
        for tag in sorted(loss_tags):
            d = data[tag]
            label = (
                tag.replace("_epoch", "")
                .replace("train_", "train ")
                .replace("val_", "val ")
            )
            ax.plot(d["steps"], d["values"], label=label, alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Losses")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    if acc_tags:
        ax = axes[ax_idx]
        for tag in sorted(acc_tags):
            d = data[tag]
            label = (
                tag.replace("_epoch", "")
                .replace("train_", "train ")
                .replace("val_", "val ")
            )
            ax.plot(d["steps"], d["values"], label=label, alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Training & Validation Accuracy")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="random (50%)")
        ax_idx += 1

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    logdir = "output/tensorboard_logs"
    data = extract_tb_data(logdir)

    if data:
        # Print summary
        print("\n--- Summary ---")
        for tag, d in sorted(data.items()):
            vals = d["values"]
            print(
                f"{tag}: {len(vals)} points, range [{min(vals):.4g}, {max(vals):.4g}]"
            )

        plot_losses(data, "output/loss_plot.png")
    else:
        print("No data found!")
