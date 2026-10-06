# EvoMan

## Dependency

This project is built using the EvoMan framework:

https://github.com/karinemiras/evoman_framework

EvoMan provides the game environment and neural-network controller
infrastructure used to evaluate the genetic algorithms in this project.

Clone the EvoMan repository and ensure it is available in your Python
environment before running the experiments.

## EvoMan Genetic Algorithm Fitness Comparison

This project investigates how fitness function design affects the evolution
and performance of neural-network-controlled agents in the EvoMan framework.

Two genetic algorithms are compared:

- **GA1** – uses EvoMan's default fitness function.
- **GA2** – uses a custom fitness function with a stronger time penalty.

Both algorithms evolve specialist agents against enemies 4, 6, and 7.

## Methods

Both GAs use:

- Population size: 150
- 20 generations
- 10 independent runs
- Tournament selection
- Single-point crossover
- Mutation rate: 0.2
- Elitism: top 2 individuals

## Fitness Functions

### GA1

Uses the default EvoMan fitness function.

### GA2

Uses a custom fitness function:

fitness = 100 - enemy_energy - 0.05 * time_alive

The stronger time penalty encourages faster enemy elimination.


## Running

Run GA1:

python GA1_runner.py

Run GA2:

python GA2_runner.py

Analyze the results:

python ga_analysis.py
