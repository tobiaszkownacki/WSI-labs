import numpy as np
import random
from cec2017.functions import f2, f13
DIMENSIONALITY = 10
MAX_X = 100
BUDGET = 10000


def find_the_best(score_list, population) -> tuple[np.array, float]:
    min_index = np.argmin(score_list)
    min_point = population[min_index]
    min_value = score_list[min_index]
    return min_point, min_value


def reproduction(
        population: list[np.array], score_list: list[float]
) -> list[np.array]:
    new_population = []
    for _ in range(len(population)):
        player1_index = random.randint(0, len(population) - 1)
        player2_index = random.randint(0, len(population) - 1)
        if score_list[player1_index] < score_list[player2_index]:
            new_population.append(population[player1_index])
        else:
            new_population.append(population[player2_index])
    return new_population


def mutation(population: list[np.array], sigma: float) -> list[np.array]:
    for i in range(len(population)):
        population[i] += np.random.normal(0, sigma, size=DIMENSIONALITY)
        population[i] = np.clip(population[i], -MAX_X, MAX_X)
    return population


def evolutionary_algorithm(
    f, population_size: int, sigma: float, budget: int = BUDGET
) -> tuple[np.array, float]:

    population = [
        np.random.uniform(-MAX_X, MAX_X, size=DIMENSIONALITY)
        for _ in range(population_size)
    ]
    iteration_max = budget // population_size
    score_list = [f(x) for x in population]
    min_point, min_value = find_the_best(score_list, population)

    for _ in range(iteration_max):
        population = reproduction(population, score_list)
        population = mutation(population, sigma)
        score_list = [f(x) for x in population]
        new_min_point, new_min_value = find_the_best(
            score_list, population
        )
        if new_min_value < min_value:
            min_point = new_min_point
            min_value = new_min_value

    return min_point, min_value


def main():
    min_point, min_value = evolutionary_algorithm(f2, 20, 2)
    print(f"Minimum point: {min_point}")
    print(f"Minimum value: {min_value}")


if __name__ == "__main__":
    main()
