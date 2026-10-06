# %%
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
        self.enemies = enemies  # list of enemy groups where each group contains a list of enemies
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_size = elitism_size
        self.n_hidden_neurons = n_hidden_neurons
        self.runs = runs
        self.results = {}
        self.generation_statistics = {}

        
        os.makedirs(self.experiment_name, exist_ok=True)

    # Initializing population
    def initialize_population(self):
        return [np.random.uniform(-1.0, 1.0, 265) for _ in range(self.population_size)]

    # Fitness function
    def fitness_function(self, individual, environments):
        fitness_values = []
        for env in environments:
            fitness, player_energy, enemy_energy, time_alive = env.play(pcont=individual)
            fitness_values.append(fitness)
        return np.mean(fitness_values)

    # Tournament selection
    def tournament_selection(self, population, fitness):
        candidates = []
        for _ in range(len(population)):
            candidate_idx = np.random.choice(len(population), self.tournament_size, replace=False)
            best_idx = max(candidate_idx, key=lambda x: fitness[x])
            candidates.append(population[best_idx])
        return candidates

    # Single point crossover
    def crossover(self, parent1, parent2):
        cross_point = np.random.randint(0, 265)
        child1 = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
        child2 = np.concatenate((parent2[:cross_point], parent1[cross_point:]))
        return child1, child2

    # Mutation with random step size
    def mutate(self, individual):
        i = 0
        while i < len(individual):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.uniform(-0.1, 0.1)
            step_size = np.random.randint(1, 5)
            i += step_size
        return individual

    def run(self, speed="fastest", visuals=False):
        for list_enemy_group in self.enemies:
            print(f"Starting GA evolution for enemy group {list_enemy_group}")
            group_key = tuple(list_enemy_group)
            self.results[group_key] = {
                "best_individuals": [],
                "max_fitness_per_generation": [[] for _ in range(self.runs)],
                "mean_fitness_per_generation": [[] for _ in range(self.runs)]
            }
            
            all_fitnesses_per_run = []

            for run in range(self.runs):
                print(f"-- Run {run + 1}/{self.runs} for enemy group {list_enemy_group}")

                # Environments for each enemy in the group
                environments = [Environment(
                    experiment_name=self.experiment_name,
                    enemies=[enemy],
                    playermode="ai",
                    enemymode="static",
                    speed=speed,
                    contacthurt="player",
                    player_controller=player_controller(self.n_hidden_neurons),
                    level=2,
                    visuals=visuals
                ) for enemy in list_enemy_group]

                population = self.initialize_population()
                best_individual = None
                best_fitness = float(-inf)

                for generation in range(self.generations):
                    fitness_scores = [self.fitness_function(individual, environments) for individual in population]

                    max_fitness = np.max(fitness_scores)
                    mean_fitness = np.mean(fitness_scores)

                    if max_fitness > best_fitness:
                        best_fitness = max_fitness
                        best_individual_idx = np.argmax(fitness_scores)
                        best_individual = population[best_individual_idx]

                    self.results[group_key]["max_fitness_per_generation"][run].append(max_fitness)
                    self.results[group_key]["mean_fitness_per_generation"][run].append(mean_fitness)

                    # Selection + elitism
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

                # Saving best individual for this run
                self.results[group_key]["best_individuals"].append(best_individual)

                # Saving fitness data for this run
                all_fitnesses_per_run.append(self.results[group_key]["mean_fitness_per_generation"][run])

            # Save generation statistics for all runs
            max_fitness_data = np.array(self.results[group_key]["max_fitness_per_generation"])
            mean_fitness_data = np.array(self.results[group_key]["mean_fitness_per_generation"])
            
            generation_stats = {
                'mean_fitness': mean_fitness_data,
                'max_fitness': max_fitness_data
            }
            
            
            np.save(os.path.join(self.experiment_name, f"generation_statistics_enemy_group_GA1_{group_key}.npy"),
                    generation_stats)
            print(f"Saved generation statistics for enemy group {list_enemy_group} to file.")

            # Saving best individuals
            all_best_individuals = np.array(self.results[group_key]["best_individuals"])
            np.save(os.path.join(self.experiment_name, f"best_individuals_enemy_group_GA1_{group_key}.npy"),
                    all_best_individuals)
            print(f"Saved best individuals for enemy group {list_enemy_group} to file.")


def main():
    ga_runner = GARunner(
        experiment_name="task2_ga_experiment",
        enemies=[[3,6,7], [4, 7, 8]],
        population_size=100,
        generations=30,
        mutation_rate=0.2,
        tournament_size=5,
        elitism_size=2,
        n_hidden_neurons=10,
        runs=10
    )
    ga_runner.run(speed="fastest", visuals=False)


if __name__ == "__main__":
    main()
