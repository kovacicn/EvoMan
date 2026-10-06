# Evolutionary Computing with EvoMan

This repository contains two projects developed using the
[EvoMan framework](https://github.com/karinemiras/evoman_framework)
for the Evolutionary Computing course at Vrije Universiteit Amsterdam.

The projects explore how Genetic Algorithms can be used to evolve
neural-network-controlled agents for video game playing.

## Projects

### 1. Specialist Agent – Fitness Function Comparison

Folder: `ga-fitness-comparison`

The first project investigates how different fitness functions influence
the evolution and performance of specialist agents.

Two Genetic Algorithms are compared:

- **GA1** uses EvoMan's default fitness function.
- **GA2** uses a custom fitness function with a stronger penalty for time.

The agents are evolved separately against enemies 4, 6, and 7.

### 2. Generalist Agent Evolution

Folder: `generalist-agent-evolution`

The second project extends the approach to generalist agents trained
against groups of enemies.

Again, two Genetic Algorithms are compared using different fitness
functions, but the goal is now to evolve agents that can perform well
across multiple enemies rather than specialize against a single opponent.

The final agents are evaluated using fitness progression, individual gain,
win rate, and performance against all EvoMan enemies.

## EvoMan Framework

Both projects depend on the EvoMan framework:

https://github.com/karinemiras/evoman_framework

EvoMan provides the game environment, enemy simulations, and neural-network
controller infrastructure used to evaluate the evolved agents.

The framework should be installed separately before running the experiments.
