# [x] - reprezentacja funkcji w kodzie
#       [x] funkcja sumy i iloczynu w pythonie
# [x] - parametry algorytmu uczącego
# [x] - pseudokod obrazujący działanie algorytmu uczącego
# [x] - implementacja algorytmu
#       [x] funckja do liczenia gradientu
#       [ ] wizualizacja
#       [ ] refactoring
# [ ] - testy i zebranie wyników wedle polecenia
# [ ] - sprawozdanie

from matplotlib import pyplot as plt
import numpy as np
from typing import Tuple
import math
import random

# --------------------------------- Functions -------------------------------- #

d = 2


# x ∈ [-5,12; 5,12], x ∈ R²
def rastrigin(x: Tuple[float]) -> float:
    global d
    c = 10 * d
    s = sum(
        [x[i - 1] ** 2 - 10 * math.cos(2 * math.pi * x[i - 1]) for i in range(1, d + 1)]
    )
    return c + s


def grad_rastrigin(x: Tuple[float]) -> Tuple[float, float]:
    if len(x) != 2:
        raise ValueError("This func is for 2d only")

    return (
        2 * (x[0] + 10 * math.pi * math.sin(2 * math.pi * x[0])),
        2 * (x[1] + 10 * math.pi * math.sin(2 * math.pi * x[1])),
    )


# x ∈ [-5; 5], x ∈ R²
def griewank(x: Tuple[float]) -> float:
    global d
    s = sum([x[i - 1] ** 2 / 4000 for i in range(1, d + 1)])
    p = math.prod([math.cos(x[i - 1]) / math.sqrt(i) for i in range(1, d + 1)])
    return s - p + 1


def grad_griewank(x: Tuple[float]) -> Tuple[float, float]:
    if len(x) != 2:
        raise ValueError("This func is for 2d only")

    return (
        x[0] / 2000 + math.cos(x[1] / math.sqrt(2)) * math.sin(x[0]),
        x[1] / 2000 + math.cos(x[0]) * math.sin(x[1] / math.sqrt(2)) / math.sqrt(2),
    )


# ---------------------------------- Params ---------------------------------- #

step = 0.01

# --------------------------------- Algorithm -------------------------------- #


class SimpleGradient:

    def __init__(self, func, grad, x_start, step):
        self._func = func
        self._grad = grad
        self._x = x_start
        self._step = step
        self._points = []

    def init(self, x_start, step):
        self._x = x_start
        self._step = step
        self._points = []

    def run(self) -> float:
        self._points.append(self._x)
        prev = self._func(self._x)
        self._forward()
        while True:
            self._points.append(self._x)
            current = self._func(self._x)
            self._forward()
            if prev < current:
                return prev
            prev = current

    def _forward(self) -> None:
        x1, x2 = self._x
        gradient = self._grad(self._x)
        grad_x1, grad_x2 = gradient
        step = self._step

        # Update elements of the tuple using gradient descent
        new_x1 = x1 - step * x1 * grad_x1
        new_x2 = x2 - step * x2 * grad_x2

        # Update self._x with the new tuple
        self._x = (new_x1, new_x2)


start = (random.randrange(-512, 513) / 100, random.randrange(-512, 513) / 100)
optimizer = SimpleGradient(rastrigin, grad_rastrigin, start, step)
best = optimizer.run()

# Generate grid points
x = np.linspace(-5.12, 5.12, 100)
y = np.linspace(0, 90, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Evaluate function at each grid point
for i in range(len(x)):
    for j in range(len(y)):
        Z[i, j] = rastrigin((X[i, j], Y[i, j]))

# Plot the function
plt.contourf(X, Y, Z, levels=20)  # Contour plot

plt.scatter(
    [p[0] for p in optimizer._points],
    [p[1] for p in optimizer._points],
    color="red",
    marker="x",
    label="Additional Points",
)

plt.colorbar()  # Add color bar
plt.xlabel("X")
plt.ylabel("Y")
plt.title("2D Function Plot")
plt.show()
