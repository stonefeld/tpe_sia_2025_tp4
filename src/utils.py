import os


def save_plot(fig, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {filepath}")
