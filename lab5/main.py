"""
Created on Mon Nov  8 16:51:50 2021

@author: RafaĹ Biedrzycki
Kodu tego mogą używać moi studenci na ćwiczeniach z przedmiotu WstÄp do Sztucznej Inteligencji.
Kod ten powstał aby przyspieszyć i ułatwić pracę studentów, aby mogli skupić się na algorytmach sztucznej inteligencji.
Kod nie jest wzorem dobrej jakoĹci programowania w Pythonie, nie jest również wzorem programowania obiektowego, moĹźe zawieraÄ bĹÄdy.

Nie ma obowiązku używania tego kodu.
"""
import matplotlib.pyplot as plt
import numpy as np

p = [1, 7]
np.random.seed(1)

L_BOUND = -5
U_BOUND = 5


def q(x):
    return np.sin(x * np.sqrt(p[0] + 1)) + np.cos(x * np.sqrt(p[1] + 1))


x = np.linspace(L_BOUND, U_BOUND, 300)
y = q(x)


# f logistyczna jako przykład sigmoidalnej
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# pochodna fun. 'sigmoid'
def d_sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)


# f. straty
def nloss(y_out, y):
    return (y_out - y) ** 2


# pochodna f. straty
def d_nloss(y_out: float, y: float):
    return 2 * (y_out - y)


class DlNet:
    def __init__(self, x_set: np.array, y_set: np.array):
        self.x_mean: float = np.mean(x_set)
        self.x_std: float = np.std(x_set)
        self.x_set: np.array = (x_set - self.x_mean) / self.x_std

        self.y_mean: float = np.mean(y_set)
        self.y_std: float = np.std(y_set)
        self.y_set: np.array = (y_set - self.y_mean) / self.y_std
        self.y_out: float = 0

        self.HIDDEN_L_SIZE = 13
        self.LR = 0.1

        self.hidden_weights: np.array = np.random.uniform(
            -1, 1, (self.HIDDEN_L_SIZE, 1)
        )
        self.hidden_bias: np.array = np.random.uniform(-1, 1, (self.HIDDEN_L_SIZE, 1))
        self.output_weights: np.array = np.zeros(shape=(1, self.HIDDEN_L_SIZE))
        self.output_bias: np.array = np.zeros(shape=(1, 1))

        # nazwy zmiennych pochodzą z notacji z wykładu
        self.y1: np.array = np.zeros(shape=(self.HIDDEN_L_SIZE, 1))
        self.s: np.array = np.zeros(shape=(self.HIDDEN_L_SIZE, 1))

    def forward(self, x: float):
        self.s = self.hidden_weights * x + self.hidden_bias
        self.y1 = sigmoid(self.s)
        self.y_out = np.dot(self.output_weights, self.y1) + self.output_bias

    def predict(self, x: float):
        x = (x - self.x_mean) / self.x_std
        self.forward(x)
        return (self.y_out * self.y_std + self.y_mean).item()

    def backward(self, x: float, y: float):
        # output layer
        d_output_weights: np.array = self.y1.T * d_nloss(self.y_out, y)
        d_output_biases: np.array = d_nloss(self.y_out, y)

        # hidden layer
        dq_dy: np.array = d_nloss(self.y_out, y) * self.output_weights.T
        dq_ds: np.array = dq_dy * d_sigmoid(self.s)
        d_hidden_weights: np.array = dq_ds * x
        d_hidden_biases: np.array = dq_ds

        return (d_output_weights, d_output_biases, d_hidden_weights, d_hidden_biases)

    def update_weights(
        self,
        d_output_weights: np.array,
        d_output_biases: np.array,
        d_hidden_weights: np.array,
        d_hidden_biases: np.array,
        batch_size: int,
    ):
        self.output_weights -= self.LR * d_output_weights / batch_size
        self.output_bias -= self.LR * d_output_biases / batch_size
        self.hidden_weights -= self.LR * d_hidden_weights / batch_size
        self.hidden_bias -= self.LR * d_hidden_biases / batch_size

    def train(self, iters, batch_size=10):
        for i in range(iters):
            indices = np.random.permutation(len(self.x_set))
            x_set_shuffled = self.x_set[indices]
            y_set_shuffled = self.y_set[indices]

            total_loss = 0

            for start_idx in range(0, len(self.x_set), batch_size):
                end_idx = start_idx + batch_size
                x_batch = x_set_shuffled[start_idx:end_idx]
                y_batch = y_set_shuffled[start_idx:end_idx]

                total_d_output_weights = np.zeros_like(self.output_weights)
                total_d_output_biases = np.zeros_like(self.output_bias)
                total_d_hidden_weights = np.zeros_like(self.hidden_weights)
                total_d_hidden_biases = np.zeros_like(self.hidden_bias)

                for x, y in zip(x_batch, y_batch):
                    self.forward(x)
                    total_loss += nloss(self.y_out, y)

                    d_output_weights, d_output_biases, d_hidden_weights, d_hidden_biases = self.backward(x, y)

                    total_d_output_weights += d_output_weights
                    total_d_output_biases += d_output_biases
                    total_d_hidden_weights += d_hidden_weights
                    total_d_hidden_biases += d_hidden_biases

                batch_size_actual = len(x_batch)
                self.update_weights(
                    total_d_output_weights,
                    total_d_output_biases,
                    total_d_hidden_weights,
                    total_d_hidden_biases,
                    batch_size_actual,
                )

            if (i + 1) % 1000 == 0:
                print(f"Iteracja {i + 1}/{iters}, Strata: {total_loss}")


def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def main():
    nn = DlNet(x, y)
    nn.train(15000, 10)
    yh = [nn.predict(xi) for xi in x]

    print(f"MSE: {calculate_mse(y, yh)}")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.plot(x, y, 'r', label='train')
    plt.plot(x, yh, 'b', label='test')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
