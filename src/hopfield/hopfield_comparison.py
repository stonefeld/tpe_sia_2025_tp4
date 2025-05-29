import matplotlib.pyplot as plt
import numpy as np

from src.hopfield.hopfield_network import Hopfield


def get_patterns(more=False):
    patterns = []
    letters = [
        # J
        np.array(
            [
                [1, 1, 1, 1, 1],
                [-1, -1, -1, 1, -1],
                [-1, -1, -1, 1, -1],
                [1, -1, -1, 1, -1],
                [1, 1, 1, -1, -1],
            ]
        ),
        # A
        np.array(
            [
                [-1, 1, 1, 1, -1],
                [1, -1, -1, -1, 1],
                [1, 1, 1, 1, 1],
                [1, -1, -1, -1, 1],
                [1, -1, -1, -1, 1],
            ]
        ),
        # E
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, -1, -1, -1, -1],
                [1, 1, 1, -1, -1],
                [1, -1, -1, -1, -1],
                [1, 1, 1, 1, 1],
            ]
        ),
        # X
        np.array(
            [
                [1, -1, -1, -1, 1],
                [-1, 1, -1, 1, -1],
                [-1, -1, 1, -1, -1],
                [-1, 1, -1, 1, -1],
                [1, -1, -1, -1, 1],
            ]
        ),
    ]

    if more:
        letters.extend(
            [
                # O
                np.array(
                    [
                        [1, 1, 1, 1, 1],
                        [1, -1, -1, -1, 1],
                        [1, -1, -1, -1, 1],
                        [1, -1, -1, -1, 1],
                        [1, 1, 1, 1, 1],
                    ]
                ),
                # W
                np.array(
                    [
                        [1, -1, -1, -1, 1],
                        [1, -1, -1, -1, 1],
                        [1, -1, 1, -1, 1],
                        [1, 1, -1, 1, 1],
                        [1, -1, -1, -1, 1],
                    ]
                ),
                # Z
                np.array(
                    [
                        [1, 1, 1, 1, 1],
                        [-1, -1, -1, 1, -1],
                        [-1, -1, 1, -1, -1],
                        [-1, 1, -1, -1, -1],
                        [1, 1, 1, 1, 1],
                    ]
                ),
            ]
        )

    for letter in letters:
        patterns.append(letter.flatten())

    return patterns


def hopfield_comparison(n_trials=100, noise_levels=None):
    if noise_levels is None:
        noise_levels = [0, 1, 2, 3, 4, 5, 7, 10, 13, 16, 20]

    patterns = get_patterns()
    pattern_size = patterns[0].size
    n_patterns = len(patterns)

    avg_success = []
    std_success = []
    all_success = []

    for n_noise in noise_levels:
        trial_success = []
        for trial in range(n_trials):
            net = Hopfield(size=pattern_size)
            net.train(np.array(patterns))

            correct = 0
            for orig in patterns:
                noisy = orig.copy()
                if n_noise > 0:
                    flip_idx = np.random.choice(pattern_size, n_noise, replace=False)
                    noisy[flip_idx] *= -1
                recalled = net.recall(noisy, steps=10)[-1]
                if np.all(recalled == orig):
                    correct += 1
            tasa = correct / n_patterns
            trial_success.append(tasa)
        avg_success.append(np.mean(trial_success))
        std_success.append(np.std(trial_success))
        all_success.append(trial_success)

    plt.figure(figsize=(8, 5))
    plt.errorbar(noise_levels, avg_success, yerr=std_success, marker="o", capsize=5, label="Tasa de acierto")
    plt.title("Tasa de acierto promedio vs. cantidad de bits ruidosos")
    plt.xlabel("Bits alterados")
    plt.ylabel("Tasa de acierto promedio")
    plt.ylim(-0.05, 1.05)
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.boxplot(all_success, positions=noise_levels)
    plt.title("Distribución de tasa de acierto (boxplot) por nivel de ruido")
    plt.xlabel("Bits alterados")
    plt.ylabel("Tasa de acierto")
    plt.ylim(-0.05, 1.05)
    plt.grid()
    plt.show()


def hopfield_capacity_comparison(n_trials=50, max_patterns=None, noise=0.2):
    patterns = get_patterns(more=True)
    pattern_size = patterns[0].size
    total_patterns = len(patterns) if max_patterns is None else min(max_patterns, len(patterns))

    target_pattern = patterns[0]
    n_patterns_range = list(range(1, total_patterns + 1))
    avg_success = []
    std_success = []
    all_success = []

    n_noisy_bits = int(pattern_size * noise)

    for n_patterns in n_patterns_range:
        trial_success = []
        for trial in range(n_trials):
            net = Hopfield(size=pattern_size)
            net.train(np.array(patterns[:n_patterns]))

            noisy = target_pattern.copy()
            if n_noisy_bits > 0:
                flip_idx = np.random.choice(pattern_size, n_noisy_bits, replace=False)
                noisy[flip_idx] *= -1
            recalled = net.recall(noisy, steps=10)[-1][0]
            success = int(np.all(recalled == target_pattern))
            trial_success.append(success)
        avg_success.append(np.mean(trial_success))
        std_success.append(np.std(trial_success))
        all_success.append(trial_success)

    plt.figure(figsize=(8, 5))
    plt.errorbar(n_patterns_range, avg_success, yerr=std_success, marker="o", capsize=5)
    plt.title(f"Reconocimiento de la letra 'A' con ruido {noise}%")
    plt.xlabel("Cantidad de patrones almacenados")
    plt.ylabel("Tasa de acierto promedio")
    plt.ylim(-0.05, 1.05)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "capacity":
        hopfield_capacity_comparison()
    else:
        hopfield_comparison()
