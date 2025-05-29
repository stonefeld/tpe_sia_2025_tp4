import matplotlib.pyplot as plt

from src.utils import save_plot


def plot_pattern(pattern, title="", ax=None):
    matrix = pattern.reshape((5, 5))

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(matrix, cmap="gray_r", vmin=-1, vmax=1)  # 1 = negro, -1 = blanco
    ax.set_title(title)
    ax.axis("off")


def plot_recall_steps(history, title_prefix="Paso", filepath="results/hopfield_pasos.png", vertical=False):
    steps = len(history)
    size = (steps * 2, 3) if not vertical else (2, steps * 3)
    fig, axes = plt.subplots(steps if vertical else 1, 1 if vertical else steps, figsize=size)

    for i in range(steps):
        plot_pattern(history[i][0], title=f"{title_prefix}: {i}\nEnergía: {history[i][1]:.3f}", ax=axes[i])

    save_plot(fig, filepath)
    plt.show()


def plot_comparison(original, noisy, recalled):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    plot_pattern(original, "Original", ax=axes[0])
    plot_pattern(noisy, "Ruidoso", ax=axes[1])
    plot_pattern(recalled, "Resultado", ax=axes[2])

    save_plot(fig, "results/hopfield_comparacion.png")
    plt.show()
