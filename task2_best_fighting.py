import numpy as np
import pandas as pd
import os
from evoman.environment import Environment
from demo_controller import player_controller

DIR_PATH = os.path.dirname(os.path.realpath("__file__"))
experiment_data = os.path.join(DIR_PATH, "task2_ga_experiment")


def evaluate_individual_vs_enemy(individual, enemy, n_hidden_neurons=10):
    """
    Best individual vs enemy.
    Returns player and enemy energy.
    """
    # Set up the environment with the specified enemy
    env = Environment(
        enemies=[enemy],
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        speed="fastest",
        experiment_name="./task2_ga_experiment"
    )
    
    results = env.play(pcont=individual)
    player_energy, enemy_energy, _, _ = results

    return player_energy, enemy_energy

def simulate_best_solution_fights(best_individual_file, enemies, n_hidden_neurons=10):
    """
    Best individual vs all enemies
    + table with individual gain for each enemy
    """

    best_individual = np.load(best_individual_file)

    results = []
    
    for enemy in enemies:
        print(f"Fighting against enemy {enemy}...")
        player_energy, enemy_energy = evaluate_individual_vs_enemy(best_individual, enemy, n_hidden_neurons)
        
        
        individual_gain = player_energy - enemy_energy
        

        results.append({
            "Enemy": enemy,
            "Player Energy": player_energy,
            "Enemy Energy": enemy_energy,
            "Gain (Player - Enemy)": individual_gain
        })
    

    df = pd.DataFrame(results)
    print(df)

    table_file = os.path.join(experiment_data, "best_vs_enemies.csv")
    df.to_csv(table_file, index=False)
    print(f"Saved table to {table_file}")

def main():
    
    best_individual_file = os.path.join(experiment_data, "best_solution.npy")
    
    
    enemies = [1, 2, 3, 4, 5, 6, 7, 8]
    
    
    simulate_best_solution_fights(best_individual_file, enemies, n_hidden_neurons=10)

if __name__ == "__main__":
    main()
