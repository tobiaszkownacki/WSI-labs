import pandas as pd
import numpy as np


class Node:
    def __init__(self, attribute_name: str = None, class_name=None):
        self.attribute_name: str = attribute_name
        self.edges: dict[str, Node] = {}
        self.class_name = class_name

    def add_edge(self, attribute_value, node: "Node"):
        self.edges[attribute_value] = node


class ID3:
    def __init__(
        self, data: pd.DataFrame, feature_names: list[str],
        class_col_name: str
    ):
        self.root = Node()
        self.data = data
        self.class_col_name = class_col_name
        self.feature_names = feature_names
        self.class_names = data[class_col_name].unique()

    def __entropy(self, data: pd.DataFrame):
        entropy = 0
        for class_name in self.class_names:
            p = len(data[data[self.class_col_name] == class_name]) / len(data)
            if p != 0:
                entropy -= p * np.log(p)
            else:
                entropy -= 0

        return entropy

    def __inf_gain(self, data: pd.DataFrame, feature_name: str):
        gain = self.__entropy(data)
        for value in data[feature_name].unique():
            value_data = data[data[feature_name] == value]
            gain -= len(value_data) / len(data) * self.__entropy(value_data)

        return gain

    def build_tree(self):
        self.__build_tree_recur(
            self.root, self.data, self.feature_names
        )

    def __build_tree_recur(
        self, node: Node, data: pd.DataFrame, feature_names: list[str]
    ):
        if len(data[self.class_col_name].unique()) == 1:
            node.class_name = data[self.class_col_name].unique()[0]
            return

        if len(feature_names) == 0:
            node.class_name = data[self.class_col_name].value_counts().idxmax()
            return

        inf_gains_list = [
            self.__inf_gain(data, feature_name)
            for feature_name in feature_names
        ]
        best_feature = feature_names[np.argmax(inf_gains_list)]
        node.attribute_name = best_feature
        for value in data[best_feature].unique():
            new_node = Node()
            node.add_edge(value, new_node)
            new_feature_names = feature_names.copy()
            new_feature_names.remove(best_feature)
            self.__build_tree_recur(
                new_node, data[data[best_feature] == value],
                new_feature_names
            )

    def test(self, test_data: pd.DataFrame) -> pd.Series:
        return test_data.apply(lambda row: self.classify(row), axis=1)

    def classify(self, row: pd.Series):
        node = self.root
        while node.class_name is None:
            try:
                node = node.edges[row[node.attribute_name]]
            except KeyError:
                return self.data[self.class_col_name].value_counts().idxmax()
            # inv-nodes 24-26 is only in one row (breast cancer dataset)

        return node.class_name
