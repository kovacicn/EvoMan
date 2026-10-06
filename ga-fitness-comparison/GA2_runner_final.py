import numpy as np
from numpy import inf
import pickle
import os
from evoman.environment import Environment
from demo_controller import player_controller

class GARunner:
    def __init__(self, experiment_name=None, enemies=None, population_size=100, generations=100,
                 mutation_rate=0.1, tournament_size=5, elitism_size=2, n_hidden_neurons=10, runs=10):
        self.experiment_name = experiment_name
        self.enemies = enemies
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.n_hidden_neurons = n_hidden_neurons
        self.runs = runs
        self.results = {}  # storing best individuals and gains per enemy
        self.generation_statistics = {}  # for stats analysis at the end
      
    #  to initialize population
    def initialize_population(self):
        return [np.random.uniform(-1.0, 1.0, 265) for _ in range(self.population_size)]

    # fitness function
    def fitness_function(self, individual, env):
        fitness, player_energy, enemy_energy, time_alive = env.play(pcont=individual)
        fitnessV2 = 100 - enemy_energy - 0.05 * time_alive
        return fitnessV2, player_energy, enemy_energy, time_alive

    # tournament selection
    def tournament_selection(self, population, fitness):
        candidates = []
        for _ in range(len(population)):
            candidate_idx = np.random.choice(len(population), self.tournament_size, replace=False)
            best_idx = max(candidate_idx, key=lambda x: fitness[x])
            candidates.append(population[best_idx])
        return candidates

    # single point crossover
    def crossover(self, parent1, parent2):
        cross_point = np.random.randint(0, 265)
        child1 = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
        child2 = np.concatenate((parent2[:cross_point], parent1[cross_point:]))
        return child1, child2

    # mutation with random step size
    def mutate(self, individual):
        i = 0
        while i < len(individual):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.uniform(-0.1, 0.1)
            step_size = np.random.randint(1, 5)
            i += step_size
        return individual

    def run(self, speed="fastest", visuals=False):
        for enemy in self.enemies:
            print(f"\nStarting GA evolution against enemy {enemy}\n")

            self.results[enemy] = {
                "best_individuals": [],
                "mean_individual_gains": []
            }
            all_fitnesses_per_run = []

            for run in range(self.runs):
                print(f"Run {run} of GA evolution against enemy {enemy}")

                # setting the environment
                env = Environment(experiment_name=self.experiment_name,
                                  enemies=[enemy],
                                  playermode="ai",
                                  enemymode="static",
                                  speed=speed,
                                  contacthurt="player",
                                  player_controller=player_controller(self.n_hidden_neurons),
                                  level=2,
                                  visuals=visuals)

                population = self.initialize_population()
                best_individual = None
                best_fitness = float(-inf)
                fitness_data_per_generation = []

                for generation in range(self.generations):
                    fitness_scores = [self.fitness_function(individual, env)[0] for individual in population]
                    fitness_data_per_generation.append(fitness_scores)

                    max_fitness = np.max(fitness_scores)
                    if max_fitness > best_fitness:
                        best_fitness = max_fitness
                        best_individual_idx = np.argmax(fitness_scores)
                        best_individual = population[best_individual_idx]

                    # selection + elitisim
                    parents = self.tournament_selection(population, fitness_scores)
                    elite_idx = np.argsort(fitness_scores)[-self.elitism_size:]
                    elites = [population[i] for i in elite_idx]

                    next_population = []
                    for i in range(0, len(parents), 2):
                        if i + 1 >= len(parents):
                            next_population.append(parents[i])
                            continue
                        parent1, parent2 = parents[i], parents[i + 1]
                        child1, child2 = self.crossover(parent1, parent2)
                        child1 = self.mutate(child1)
                        child2 = self.mutate(child2)
                        next_population.extend([child1, child2])

                    next_population[:self.elitism_size] = elites
                    population = next_population

                # testing the individual 5x and calculating the gain
                gains = []
                for _ in range(5):
                    _, player_energy, enemy_energy, _ = self.fitness_function(best_individual, env)
                    gains.append(player_energy - enemy_energy)

                mean_individual_gain = np.mean(gains)

                # Store best individual and mean gain for this run
                self.results[enemy]["best_individuals"].append(best_individual)
                self.results[enemy]["mean_individual_gains"].append(mean_individual_gain)
                
                # saving fitness for each run
                all_fitnesses_per_run.append(np.array(fitness_data_per_generation))  # Store fitnesses for each run

            # saving fitness per run and generation
            all_fitness_data = np.array(all_fitnesses_per_run)  
            np.save(os.path.join(self.experiment_name, f"generation_statistics_enemy_GA2_{enemy}.npy"), all_fitness_data)
            print(f"Saved generation statistics for enemy {enemy} to file")

        # save results for all enemies
        self.save_best_individuals()

    def save_best_individuals(self):
        for enemy, result_data in self.results.items():
            with open(os.path.join(self.experiment_name, f"best_individual_enemy_GA2_{enemy}.pkl"), "wb") as f:
                pickle.dump(result_data, f)
            print(f"Saved best individuals and mean individual gains for enemy {enemy} to file")


def main():
    enemies = [4, 6, 7]
    ga_runner = GARunner(
        experiment_name="ga_experiment",
        enemies=enemies,
        population_size=150,
        generations=20,
        mutation_rate=0.2,
        tournament_size=5,
        elitism_size=2,
        n_hidden_neurons=10,
        runs=10  
    )
    ga_runner.run(speed="fastest", visuals=False)


if __name__ == "__main__":
    main()
