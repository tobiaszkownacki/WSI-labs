import json
from random import uniform
import sys
import argparse


class Node:
    def __init__(self, name: str, parents: list = None, prob: dict = None):
        self.name = name
        self.parents = parents if parents else []
        self.prob = prob if prob else {}


def load_data(path: str) -> list[Node]:
    nodes = []
    with open(path, "r") as file_handler:
        data = json.load(file_handler)
        for item in data:
            node = Node(item["name"], item["parents"])
            for key, value in item["probabilities"].items():
                key: tuple[bool] = tuple(
                    map(lambda x: x == "true", key.split(","))
                )
                node.prob[key] = value
            nodes.append(node)
    return nodes


def generate_sample(nodes: list[Node]) -> str:
    sample = {}
    for node in nodes:
        


def generate_samples_to_file(nodes: list[Node], samples: int) -> None:
    with open("samples.data", "w") as file_handler:
        for _ in range(samples):
            sample: str = generate_sample(nodes)
            file_handler.write(sample + "\n")


def main(arguments: list):
    parser = argparse.ArgumentParser(description="Bayesian Network")
    parser.add_argument(
        "number_of_samples", type=int, help="Number of samples to generate"
    )
    parser.add_argument(
        "network_path",
        type=str,
        help="Path to the data file that describes the network",
    )
    args = parser.parse_args(arguments[1:])

    nodes: list[Node] = load_data(args.network_path)
    generate_samples_to_file(nodes, args.number_of_samples)


if __name__ == "__main__":
    main(sys.argv)
