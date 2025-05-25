import matplotlib.pyplot as plt

from src.utils import save_plot


def plot_pattern(pattern, title="", ax=None):
    matrix = pattern.reshape((5, 5))

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(matrix, cmap="gray_r", vmin=-1, vmax=1)  # 1 = negro, -1 = blanco
    ax.set_title(title)
    ax.axis("off")


def plot_recall_steps(history, title_prefix="Paso"):
    steps = history.shape[0]
    fig, axes = plt.subplots(1, steps, figsize=(steps * 2, 2))

    for i in range(steps):
        plot_pattern(history[i], title=f"{title_prefix} {i}", ax=axes[i])

    save_plot(fig, "results/hopfield_pasos.png")
    plt.show()


def plot_comparison(original, noisy, recalled):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    plot_pattern(original, "Original", ax=axes[0])
    plot_pattern(noisy, "Ruidoso", ax=axes[1])
    plot_pattern(recalled, "Resultado", ax=axes[2])

    save_plot(fig, "results/hopfield_comparacion.png")
    plt.show()
