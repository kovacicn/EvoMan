import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_ind

DIR_PATH = os.path.dirname(os.path.realpath("__file__"))
experiment_data = os.path.join(DIR_PATH, "ga_experiment")


def perform_statistical_test(data, enemy):
    # data is a list of two lists: [ga1_gains, ga2_gains]
    gains_ga1 = data[0]
    gains_ga2 = data[1]

    # t-test between GA1 and GA2 gains
    t_stat, p_value = ttest_ind(gains_ga1, gains_ga2)

    print(f"T-test result for enemy {enemy}: t-statistic = {t_stat}, p-value = {p_value}")

    # interpreting the results
    if p_value < 0.05:
        print(f"The difference in mean individual gain between GA1 and GA2 for enemy {enemy} is statistically significant (p < 0.05).")
    else:
        print(f"No significant difference in mean individual gain between GA1 and GA2 for enemy {enemy} (p >= 0.05).")


def plot_line_plots(data, ga, enemy):
    n_runs, n_gens, pop_size = data.shape

    maxes = data.max(axis=2)
    means = data.mean(axis=2)

    means_of_maxes = maxes.mean(axis=0)
    means_of_means = means.mean(axis=0)
    stdvs_of_maxes = maxes.std(axis=0)
    stdvs_of_means = means.std(axis=0)
    generations = np.arange(n_gens) + 1

    plt.clf()
    plt.plot(generations, means_of_maxes, label="Mean max fitness", color="red")
    plt.plot(generations, means_of_means, label="Mean mean fitness", color="blue")
    plt.fill_between(generations, means_of_maxes - stdvs_of_maxes, means_of_maxes + stdvs_of_maxes, color="red",
                     alpha=0.2)
    plt.fill_between(generations, means_of_means - stdvs_of_means, means_of_means + stdvs_of_means, color="blue",
                     alpha=0.2)
    
    plt.ylim(-20, 100)
    plt.title(f"GA{ga}: Means and maxes of fitnesses per generation for enemy {enemy}")
    plt.xlabel("Generation")
    plt.xticks(np.arange(1, n_gens + 1, 1))
    plt.ylabel("Fitness")
    plt.legend()

    plt.savefig(f"ea{ga}_line_plot_{enemy}.png")


def plot_box_plots(data, enemy):
    gains_ga1 = data[0]
    gains_ga2 = data[1]

    plt.clf()

    # box plot for the mean individual gains of GA1 and GA2
    plt.boxplot([gains_ga1, gains_ga2], labels=["GA1", "GA2"])
    plt.title(f"Mean Individual Gain for Enemy {enemy}")
    plt.ylabel("Mean Individual Gain (Player Energy - Enemy Energy)")
    plt.savefig(f"box_plot_{enemy}.png")


def main():
    enemies = [4, 6, 7]
    gas = [1, 2]
    for ga in gas:
        for enemy in enemies:
            fpath = os.path.join(experiment_data, f"generation_statistics_enemy_{'GA1' if ga == 1 else 'GA2'}_{enemy}.npy")
            result_data = np.load(fpath)
            plot_line_plots(result_data, ga, enemy)

    for enemy in enemies:
        # loading GA1 and GA2 data
        with open(os.path.join(experiment_data, f"best_individual_enemy_GA1_{enemy}.pkl"), "rb") as f:
            ga1_results = pickle.load(f)
        with open(os.path.join(experiment_data, f"best_individual_enemy_GA2_{enemy}.pkl"), "rb") as f:
            ga2_results = pickle.load(f)

        # extracting the mean individual gains across runs
        ga1_gains = ga1_results["mean_individual_gains"]
        ga2_gains = ga2_results["mean_individual_gains"]

        # saving the gains for plotting and statistical test
        winner_data = [ga1_gains, ga2_gains]
        plot_box_plots(winner_data, enemy)
        perform_statistical_test(winner_data, enemy)


if __name__ == "__main__":
    main()
