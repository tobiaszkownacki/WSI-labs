import numpy as np
import random
from cec2017.functions import f2, f13
import copy

DIMENSIONALITY = 10
MAX_X = 100
BUDGET = 10000


def find_the_best(
        score_list, population: np.ndarray
) -> tuple[np.ndarray, float]:

    min_index = np.argmin(score_list)
    min_point = population[min_index]
    min_value = score_list[min_index]
    return (min_point, min_value)


def reproduction(
        population: list[np.ndarray], score_list: list[float]
) -> list[np.ndarray]:

    new_population = []
    for _ in range(len(population)):
        player1_index = random.randint(0, len(population) - 1)
        player2_index = random.randint(0, len(population) - 1)
        if score_list[player1_index] < score_list[player2_index]:
            new_population.append(copy.deepcopy(population[player1_index]))
        else:
            new_population.append(copy.deepcopy(population[player2_index]))
    return new_population


def mutation(population: list[np.ndarray], sigma: float) -> list[np.ndarray]:
    for i in range(len(population)):
        population[i] += np.random.normal(0, sigma, size=DIMENSIONALITY)
        population[i] = np.clip(population[i], -MAX_X, MAX_X)


def evolutionary_algorithm(
    f, population_size: int, sigma: float, budget: int = BUDGET
) -> tuple[np.ndarray, float]:

    population = [
        np.random.uniform(-MAX_X, MAX_X, size=DIMENSIONALITY)
        for _ in range(population_size)
    ]
    iteration_max = budget // population_size
    score_list = [f(x) for x in population]
    min_point, min_value = find_the_best(score_list, population)

    for _ in range(iteration_max):
        population = reproduction(population, score_list)
        mutation(population, sigma)
        score_list = [f(x) for x in population]
        new_min_point, new_min_value = find_the_best(
            score_list, population
        )
        if new_min_value < min_value:
            min_point = new_min_point
            min_value = new_min_value

    return (min_point, min_value)


def main():
    results = []
    population_size = 10
    for _ in range(30):
        results.append(evolutionary_algorithm(f2, population_size, 0.5)[1])
    avg = np.mean(results)
    std = np.std(results)
    min_value = np.min(results)
    max_value = np.max(results)
    print(results)
    print(f"Average: {avg:.2f}")
    print(f"Standard deviation: {std:.2f}")
    print(f"Min: {min_value:.2f}")
    print(f"Max: {max_value:.2f}")
    print(f"Iterations: {BUDGET // population_size}")


if __name__ == "__main__":
    main()
