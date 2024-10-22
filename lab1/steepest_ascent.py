import numpy as np
import matplotlib.pyplot as plt
from autograd import grad

from cec2017.functions import f1, f2, f3

MAX_X = 100
PLOT_STEP = 0.1
EPSILON = 1e-6
MAX_ITER = 30000
UPPER_BOUND = 100
DIMENSIONALITY = 2


def init_plot(f, tytle: str = ''):
    x_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
    y_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
    X, Y = np.meshgrid(x_arr, y_arr)
    Z = np.empty(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    contour = plt.contour(X, Y, Z, 20)
    plt.colorbar(contour, label='Function Value')
    plt.title(tytle)


def draw_arrow(x: np.array, x0: np.array):
    plt.arrow(
        x0[0], x0[1], x[0] - x0[0], x[1] - x0[1],
        head_width=3, head_length=6, fc='k', ec='k'
    )


def gradient(f, x: np.array):
    return grad(f)(x)


def distance(x: np.array):
    return np.sqrt(np.sum(x**2))


def booth_function(x: np.array):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


def steepest_ascent(
    f, beta: float, dim: int = DIMENSIONALITY
):
    x = np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=dim)
    iterations = 0
    cords = [x]
    while iterations <= MAX_ITER:
        d = gradient(f, x)
        if distance(d) < EPSILON:
            break
        x = x + beta * d
        x = np.clip(x, -UPPER_BOUND, UPPER_BOUND)
        cords.append(x)
        iterations += 1
    return cords


def make_plot(f, beta: float, dim: int = DIMENSIONALITY):
    init_plot(f, f'Function: {f.__name__}, beta: {beta}')
    cords = steepest_ascent(f, beta, dim)
    for i in range(1, len(cords)):
        draw_arrow(cords[i], cords[i - 1])
    print(f'Function: {f.__name__}, Iterations: {len(cords)}, beta = {beta}')
    print(f'Initial point: {cords[0]}')
    print(f'Final point: {cords[-1]}')
    print('Function value: {:.6f}'.format(f(cords[-1])))
    print()


def main():
    beta_list = [-0.07, -0.00000001, -0.0000000000000003, -0.000000005]

    function_list = [booth_function, f1, f2, f3]
    for f, beta in zip(function_list, beta_list):
        for i in range(3):
            dim = 2 if f == booth_function else 10
            make_plot(f, beta, dim)
            plt.savefig(f'plots/{f.__name__}_{beta}_{i}.png')
            plt.clf()


if __name__ == '__main__':
    main()
