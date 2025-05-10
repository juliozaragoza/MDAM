import matplotlib.pyplot as plt
import seaborn as sns


def plot_MLP(mlp_accuracys):
    plt.plot(range(1, len(mlp_accuracys) + 1), mlp_accuracys, label="MLP Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Repetition number")
    plt.show()


def plot_knn_f1(f1_scores):
    plt.figure(figsize=(10, 6))
    # Grafica los valores de F1 para cada valor de k
    for i, f1_score in enumerate(f1_scores):
        plt.plot(
            range(1, len(f1_score) + 1),
            f1_score,
            label=f"Repetition {i + 1}",
            color=colors[i],
        )
    # Calcula la media de los valores de F1 para cada valor de k
    mean_f1_scores = [sum(scores) / len(scores) for scores in zip(*f1_scores)]
    # Grafica la media de los valores de F1
    plt.plot(
        range(1, len(mean_f1_scores) + 1),
        mean_f1_scores,
        label="Media",
        linestyle="--",
        color="black",
    )
    # Muestra la gráfica
    plt.legend(loc="lower right")
    plt.xlabel("K")
    plt.ylabel("F1")
    plt.title("Combined F1 Scores")
    plt.show()
