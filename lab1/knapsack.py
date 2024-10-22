def bruteForce(
    weights: list[float], values: list[float], max_capacity: float
) -> tuple[list[int], float]:
    best_backpack = [0] * len(weights)
    best_value = 0
    num_items = len(weights)

    for i in range(1 << num_items):
        backpack = list(map(int, bin(i)[2:].zfill(num_items)))
        weight = sum(weight for weight, item_status in zip(weights, backpack)
                     if item_status == 1)
        value = sum(value for value, item_status in zip(values, backpack)
                    if item_status == 1)

        if weight <= max_capacity and value >= best_value:
            best_backpack = backpack.copy()
            best_value = value

    return (best_backpack, best_value)


def heuristic(
    weights: list[float], values: list[float], capacity: float
) -> tuple[list[int], float]:
    value_to_weight_ratio = [
        (value / weight, index)
        for index, (value, weight) in enumerate(zip(values, weights))
    ]
    value_to_weight_ratio.sort(key=lambda x: x[0], reverse=True)

    current_weight = 0
    current_value = 0
    backpack = [0] * len(weights)
    for _, item_index in value_to_weight_ratio:
        if current_weight + weights[item_index] > capacity:
            continue
        current_weight += weights[item_index]
        current_value += values[item_index]
        backpack[item_index] = 1
        if current_weight == capacity:
            return (backpack, current_value)

    return (backpack, current_value)


def main() -> None:
    weights = [8, 3, 5, 2]
    values = [16, 8, 9, 6]
    max_capacity = sum(weights)/2

    result_bF = bruteForce(weights.copy(), values.copy(), max_capacity)
    result_h = heuristic(weights.copy(), values.copy(), max_capacity)

    print(f'BruteForce Backpack: {result_bF[0]} has value {result_bF[1]}')
    print(f'Heuristic Backpack: {result_h[0]} has value {result_h[1]}')


if __name__ == "__main__":
    main()
