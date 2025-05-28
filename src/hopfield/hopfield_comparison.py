import numpy as np
import matplotlib.pyplot as plt
from src.hopfield.hopfield_network import Hopfield

def get_patterns():
    # Definí patrones simples (ejemplo: 4 letras como en tu main)
    patterns = []
    letters = [
        np.array([
            [-1, 1, 1, 1, -1],
            [1, -1, -1, -1, 1],
            [1, 1, 1, 1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1]
        ]),
        np.array([
            [1, 1, 1, 1, 1],
            [1, -1, -1, -1, -1],
            [1, 1, 1, -1, -1],
            [1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1]
        ]),
        np.array([
            [ 1, -1, -1, -1,  1],
            [-1,  1, -1,  1, -1],
            [-1, -1,  1, -1, -1],
            [-1,  1, -1,  1, -1],
            [ 1, -1, -1, -1,  1]
        ]),
        np.array([
            [1, 1, 1, 1, 1],
            [-1, -1, -1, 1, -1],
            [-1, -1, -1, 1, -1],
            [1, -1, -1, 1, -1],
            [1, 1, 1, -1, -1]
        ]),
    ]
    for l in letters:
        patterns.append(l.flatten())
    return patterns

def hopfield_comparison(n_trials=100, noise_levels=None):
    if noise_levels is None:
        noise_levels = [0, 1, 2, 3, 4, 5, 7, 10, 13, 16, 20]  # hasta 25 bits posibles

    patterns = get_patterns()
    pattern_size = patterns[0].size
    n_patterns = len(patterns)

    avg_success = []
    std_success = []
    all_success = []  # Para boxplot

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

    # Gráfico de tasa promedio con barras de error
    plt.figure(figsize=(8, 5))
    plt.errorbar(noise_levels, avg_success, yerr=std_success, marker='o', capsize=5, label='Tasa de acierto')
    plt.title('Tasa de acierto promedio vs. cantidad de bits ruidosos')
    plt.xlabel('Bits alterados')
    plt.ylabel('Tasa de acierto promedio')
    plt.ylim(-0.05, 1.05)
    plt.grid()
    plt.legend()
    plt.show()

    # Boxplot de dispersión por nivel de ruido
    plt.figure(figsize=(8, 5))
    plt.boxplot(all_success, positions=noise_levels)
    plt.title('Distribución de tasa de acierto (boxplot) por nivel de ruido')
    plt.xlabel('Bits alterados')
    plt.ylabel('Tasa de acierto')
    plt.ylim(-0.05, 1.05)
    plt.grid()
    plt.show()

    # (Opcional) Imprimir tabla resumen
    print("Bits Alterados | Tasa promedio | STD")
    for n, avg, stdev in zip(noise_levels, avg_success, std_success):
        print(f"{n:<14} {avg:.2f}         {stdev:.2f}")

if __name__ == "__main__":
    hopfield_comparison()
