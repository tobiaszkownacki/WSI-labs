"""
Created on Mon Nov  8 16:51:50 2021

@author: RafaĹ Biedrzycki
Kodu tego mogą używać moi studenci na ćwiczeniach z przedmiotu WstÄp do Sztucznej Inteligencji.
Kod ten powstał aby przyspieszyć i ułatwić pracę studentów, aby mogli skupić się na algorytmach sztucznej inteligencji.
Kod nie jest wzorem dobrej jakoĹci programowania w Pythonie, nie jest również wzorem programowania obiektowego, moĹźe zawieraÄ bĹÄdy.

Nie ma obowiązku używania tego kodu.
"""

import numpy as np

# TODO tu prosze podac pierwsze cyfry numerow indeksow
p = [1, 7]  # swoją pierwszą już podałem

L_BOUND = -5
U_BOUND = 5


def q(x):
    return np.sin(x * np.sqrt(p[0] + 1)) + np.cos(x * np.sqrt(p[1] + 1))


x = np.linspace(L_BOUND, U_BOUND, 100)
y = q(x)

np.random.seed(1)


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
        x_mean: float = np.mean(x_set)
        x_std: float = np.std(x_set)
        self.x_set: np.array = (x_set - x_mean) / x_std

        self.y_mean: float = np.mean(y_set)
        self.y_std: float = np.std(y_set)
        self.y_set: np.array = (y_set - self.y_mean) / self.y_std
        self.y_out: float = 0

        self.HIDDEN_L_SIZE = 9
        self.LR = 0.003

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
        self.s = np.dot(self.hidden_weights, x) + self.hidden_bias
        self.y1 = sigmoid(self.s)
        self.y_out = np.dot(self.output_weights, self.y1) + self.output_bias

    def predict(self, x: float):
        self.forward(x)
        return self.y_out * self.y_std + self.y_mean

    def backward(self, x: float, y: float):
        # output layer
        d_output_weights: np.array = self.y1 * d_nloss(self.y_out, y)
        d_output_biases: np.array = d_nloss(self.y_out, y)

        # hidden layer
        dq_dy: np.array = d_nloss(self.y_out, y) * self.output_weights
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

    def train(self, x_set, y_set, iters):
        for i in range(0, iters):
            pass


# TODO


nn = DlNet(x, y)
nn.train(x, y, 15000)

yh = []  # TODO tu umieścić wyniki (y) z sieci

# import matplotlib.pyplot as plt


# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.spines['left'].set_position('center')
# ax.spines['bottom'].set_position('zero')
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')

# plt.plot(x,y, 'r')
# plt.plot(x,yh, 'b')

# plt.show()
