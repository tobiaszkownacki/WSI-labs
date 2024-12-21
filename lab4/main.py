from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from model import ID3
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_heatmaps(cm_train, cm_test, classes):
    _, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(
        cm_train, annot=True, ax=axs[0], fmt='d', annot_kws={"size": 14}
    )
    sns.heatmap(
        cm_test, annot=True, ax=axs[1], fmt='d', annot_kws={"size": 14}
    )
    axs[0].set_title('Train Set', fontsize=16)
    axs[1].set_title('Test Set', fontsize=16)
    for idx in (0, 1):
        axs[idx].set_xticklabels(classes, rotation=45, ha='right', fontsize=12)
        axs[idx].set_yticklabels(classes, rotation=0, fontsize=12)
        axs[idx].set_xlabel('Predicted', fontsize=14)
        axs[idx].set_ylabel('True', fontsize=14)
    plt.tight_layout()


def main():
    data_set = fetch_ucirepo(id=14)
    # 14 - breast cancer dataset
    # 73 - mushroom dataset

    data = data_set.data.original
    data = data.fillna('?')  # fill missing values with '?'

    feature_names = data_set.data.features.columns.tolist()
    class_col_name = data_set.data.targets.columns[0]
    class_names = data_set.data.targets.iloc[:, 0].unique()

    train_data, test_data = train_test_split(
        data, test_size=0.4, random_state=25
    )

    id3 = ID3(train_data, feature_names, class_col_name)
    id3.build_tree()

    train_predictions = id3.test(train_data)
    test_predictions = id3.test(test_data)

    cm_train = confusion_matrix(
        train_data[class_col_name], train_predictions, labels=class_names
    )
    cm_test = confusion_matrix(
        test_data[class_col_name], test_predictions, labels=class_names
    )

    plot_heatmaps(cm_train, cm_test, class_names)
    plt.savefig('confusion_matrix.png')


if __name__ == '__main__':
    main()