from knapsack import bruteForce, heuristic
import time
import random


def main():
    weights = []
    values = []
    max_capacity = 0
    total_time = 0

    while total_time <= 45:
        weights.append(random.randint(1, 15))
        values.append(random.randint(1, 15))
        max_capacity = sum(weights) / 2
        start = time.process_time()
        _ = bruteForce(weights, values, max_capacity)
        end = time.process_time()
        total_time = end - start
    print(f"BruteForce method can handle {len(weights)} items in "
          f"{total_time:.4f} seconds")

    start = time.process_time()
    _ = heuristic(weights, values, max_capacity)
    end = time.process_time()
    total_time = end - start
    print(f"Heuristic method can handle {len(weights)} items in "
          f"{total_time:.4f} seconds")

    weights.append(random.randint(1, 15))
    values.append(random.randint(1, 15))
    max_capacity = sum(weights) / 2

    start = time.process_time()
    _ = bruteForce(weights, values, max_capacity)
    end = time.process_time()
    total_time = end - start
    print(f"After adding one more item, BruteForce method takes "
          f"{total_time:.4f} seconds")

    start = time.process_time()
    _ = heuristic(weights, values, max_capacity)
    end = time.process_time()
    total_time = end - start
    print(f"After adding one more item, Heuristic method takes "
          f"{total_time:.4f} seconds")


if __name__ == "__main__":
    main()
