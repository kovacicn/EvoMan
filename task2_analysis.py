import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_ind
from evoman.environment import Environment
from demo_controller import player_controller

DIR_PATH = os.path.dirname(os.path.realpath("__file__"))
experiment_data = os.path.join(DIR_PATH, "task2_ga_experiment")

##################################################################################################################################################################################

def combined_line_plots(list_enemy_group):
    """
    It will plot combined fitness progress over generations for GA1 and GA2 for each enemy group in a list_enemy_group.
    """
    for enemy_group in list_enemy_group:
        # Loading data collected during the evolution
        # Data is stored in the dictionary
        data_ga1 = np.load(os.path.join(experiment_data, f"generation_statistics_enemy_group_GA1_{tuple(enemy_group)}.npy"), allow_pickle=True).item()
        data_ga2 = np.load(os.path.join(experiment_data, f"generation_statistics_enemy_group_GA2_{tuple(enemy_group)}.npy"), allow_pickle=True).item()
        
        # Passing the data to the plotting function
        line_plots(data_ga1, data_ga2, enemy_group)

def line_plots(data_ga1, data_ga2, enemy_group):
    """
    It will plot fitness progress for a specific enemy group.
    I am calculating the average of the mean and max fitness across 10 runs.
    """
    # Extracting data from the dictionaries
    mean_fitness_ga1 = data_ga1['mean_fitness']
    max_fitness_ga1 = data_ga1['max_fitness']
    
    mean_fitness_ga2 = data_ga2['mean_fitness']
    max_fitness_ga2 = data_ga2['max_fitness']
    
    n_runs, n_gens = mean_fitness_ga1.shape
    
    # Calculating mean and max fitness over generations for GA1 + std
    mean_fitness_ga1_avg = mean_fitness_ga1.mean(axis=0)  
    max_fitness_ga1_avg = max_fitness_ga1.max(axis=0)    
    std_mean_fitness_ga1 = mean_fitness_ga1.std(axis=0)  
    std_max_fitness_ga1 = max_fitness_ga1.std(axis=0)  

    # Calculating mean and max fitness over generations for GA2 + std
    mean_fitness_ga2_avg = mean_fitness_ga2.mean(axis=0)  
    max_fitness_ga2_avg = max_fitness_ga2.max(axis=0)    
    std_mean_fitness_ga2 = mean_fitness_ga2.std(axis=0)  
    std_max_fitness_ga2 = max_fitness_ga2.std(axis=0)  
    
    generations = np.arange(1, n_gens + 1)
    
    plt.clf()
    # Plotting mean and fitness for GA1
    plt.plot(generations, mean_fitness_ga1_avg, label="GA1 Mean Fitness", color="blue")
    plt.fill_between(generations, mean_fitness_ga1_avg - std_mean_fitness_ga1, mean_fitness_ga1_avg + std_mean_fitness_ga1, color="blue", alpha=0.2)
    
    plt.plot(generations, max_fitness_ga1_avg, label="GA1 Max Fitness", color="grey")
    plt.fill_between(generations, max_fitness_ga1_avg - std_max_fitness_ga1, max_fitness_ga1_avg + std_max_fitness_ga1, color="grey", alpha=0.2)

    # Plotting mean and max fitness for GA2
    plt.plot(generations, mean_fitness_ga2_avg, label="GA2 Mean Fitness", color="purple")
    plt.fill_between(generations, mean_fitness_ga2_avg - std_mean_fitness_ga2, mean_fitness_ga2_avg + std_mean_fitness_ga2, color="purple", alpha=0.2)
    
    plt.plot(generations, max_fitness_ga2_avg, label="GA2 Max Fitness", color="orange")
    plt.fill_between(generations, max_fitness_ga2_avg - std_max_fitness_ga2, max_fitness_ga2_avg + std_max_fitness_ga2, color="orange", alpha=0.2)

    ####
    plt.title(f"Fitness Progress for Enemy Group {enemy_group} (GA1 vs GA2)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(f"fitness_progress_enemy_group_{tuple(enemy_group)}.png")
    plt.show()


##################################################################################################################################################################################

def calculate_win_rate(individual, enemies, n_hidden_neurons=10):
    """
    I am calculating  win rate by testing the individual against all enemies.
    """
    wins = 0
    for enemy in enemies:
        _, player_energy, enemy_energy = individual_vs_enemy(individual, enemy, n_hidden_neurons)
        
        if player_energy > 0 and enemy_energy == 0:
            wins += 1
    return wins / len(enemies)   


##################################################################################################################################################################################

def individual_vs_enemy(individual, enemy, n_hidden_neurons = 10):
    """
    This is where our individual is fighting against the specific enemy 
    and where we calculate the gain.
    """
    
    env = Environment(
        enemies=[enemy],
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        speed="fastest",
        experiment_name="./task2_ga_experiment"
    )
    
    energies = env.play(pcont=individual)
    player_energy, enemy_energy = energies[0], energies[1]
    individual_gain = player_energy - enemy_energy
    return individual_gain, player_energy, enemy_energy

    

def test_individual_vs_all_enemies(individual, enemy_group, n_hidden_neurons=10):
    """
    Individual vs all enemies in the given enemy group,
    and i will return the average gain"""
    gains = []
    for enemy in enemy_group:
        gain = individual_vs_enemy(individual, enemy, n_hidden_neurons)[0]
        gains.append(gain)

    return np.mean(gains)

def test_best_individuals_vs_all_enemies(list_enemy_group, experiment_data):
    """
    Comparing GA1 and GA2 best solutions by testing them against all enemies
    + presenting the resulting gains in box plots
    + stat tests
    + save the very best individual of both GAs
    """
    all_ga1_gains = []
    all_ga2_gains = []
    all_ga1_win_rates = []  
    all_ga2_win_rates = []  
    all_ga1_individuals = []
    all_ga2_individuals = []
    
    for enemy_group in list_enemy_group:

        ga1_best_individuals = np.load(os.path.join(experiment_data,
                                                 f"best_individuals_enemy_group_GA1_{tuple(enemy_group)}.npy"))
        ga2_best_individuals = np.load(os.path.join(experiment_data,
                                                 f"best_individuals_enemy_group_GA2_{tuple(enemy_group)}.npy"))

        ga1_gains = []
        ga2_gains = []
        ga1_win_rates = []  
        ga2_win_rates = []  

        # Testing each of the 10 best individuals against all enemies
        for run in range(len(ga1_best_individuals)):
            ga1_individual = ga1_best_individuals[run]
            ga2_individual = ga2_best_individuals[run]

            # Evaluating GA1 and GA2 individuals against all enemies in the group
            ga1_gain = test_individual_vs_all_enemies(ga1_individual, enemy_group)
            ga2_gain = test_individual_vs_all_enemies(ga2_individual, enemy_group)

            ga1_gains.append(ga1_gain)
            ga2_gains.append(ga2_gain)
            
            ga1_win_rate = calculate_win_rate(ga1_individual, enemy_group)
            ga2_win_rate = calculate_win_rate(ga2_individual, enemy_group)

            ga1_win_rates.append(ga1_win_rate)
            ga2_win_rates.append(ga2_win_rate)

        all_ga1_gains.append(ga1_gains)
        all_ga2_gains.append(ga2_gains)
        all_ga1_win_rates.append(ga1_win_rates)  
        all_ga2_win_rates.append(ga2_win_rates)  
        all_ga1_individuals.append(ga1_best_individuals)
        all_ga2_individuals.append(ga2_best_individuals)

        # Box plots
        plt.figure(figsize=(8, 5))
        plt.boxplot([ga1_gains, ga2_gains], labels=["GA1", "GA2"])
        plt.scatter(x=[1]*len(ga1_gains), y=ga1_gains)
        plt.scatter(x=[2]*len(ga2_gains), y=ga2_gains)
        plt.title(f"Mean Individual Gain for enemy group {enemy_group}")
        plt.ylabel("Mean Individual Gain (Player Energy - Enemy Energy)")
        plt.savefig(f"box_plot_{tuple(enemy_group)}.png")
        plt.show()
        

        # Statistical test between GA1 and GA2 gains for this enemy group
        perform_statistical_test([ga1_gains, ga2_gains], enemy_group)
        perform_statistical_test([ga1_win_rates, ga2_win_rates], enemy_group, metric="win rate")

    # fInding the best individual of both GAs
    finding_best_of_best(all_ga1_gains, all_ga2_gains, all_ga1_individuals, all_ga2_individuals)


def finding_best_of_best(all_ga1_gains, all_ga2_gains, all_ga1_individuals, all_ga2_individuals):
    """
    Finding the best individual of both GAs based on the highest average individual gain,
    + saving it to a file for competition
    """
    ga1_gains_flat = np.concatenate(all_ga1_gains)
    ga2_gains_flat = np.concatenate(all_ga2_gains)
    
    ga1_individuals_flat = np.concatenate(all_ga1_individuals)
    ga2_individuals_flat = np.concatenate(all_ga2_individuals)

    # Finding the index of the best individual 
    best_ga1_idx = np.argmax(ga1_gains_flat)
    best_ga2_idx = np.argmax(ga2_gains_flat)

    best_ga1_individual = ga1_individuals_flat[best_ga1_idx]
    best_ga2_individual = ga2_individuals_flat[best_ga2_idx]

    # Comparing the best individual from both GAs
    if ga1_gains_flat[best_ga1_idx] > ga2_gains_flat[best_ga2_idx]:
        best_individual = best_ga1_individual
        best_ga = "GA1"
    else:
        best_individual = best_ga2_individual
        best_ga = "GA2"

    # Saving the best individual to a file
    best_solution_file = os.path.join(experiment_data, f"best_solution.npy")
    np.save(best_solution_file, best_individual)
    print(f"Best individual from {best_ga} saved to {best_solution_file}")
    
    # I created another file to save the best individual for the competition
    # Have to check if it is the right format
    best_solution_competition = os.path.join(experiment_data, f"competition_best_solution.txt")
    with open(best_solution_competition, 'w') as f:
        f.write(f"Best individual from {best_ga}:\n")
        np.savetxt(f, best_individual, fmt='%f')



def perform_statistical_test(data, list_enemy_group, metric="gain"):
    """
    T-test between GA1 and GA2, comparing either gains or win rates.
    """
    ga1_values = data[0]
    ga2_values = data[1]

    t_stat, p_value = ttest_ind(ga1_values, ga2_values)

    print(f"T-test result for {metric} in enemy group {list_enemy_group}: t-statistic = {t_stat}, p-value = {p_value}")
    if p_value < 0.05:
        print(f"The difference in {metric} between GA1 and GA2 for enemy group {list_enemy_group} is statistically significant (p < 0.05).")
    else:
        print(f"No significant difference in {metric} between GA1 and GA2 for enemy group {list_enemy_group} (p >= 0.05).")


def main():
    list_enemy_group = [[1,2,7], [4, 6, 8]]
    combined_line_plots(list_enemy_group)
    test_best_individuals_vs_all_enemies(list_enemy_group, experiment_data)



if __name__ == "__main__":
    main()
    
