import numpy as np
from matplotlib import pyplot as plt
from typing import Callable, List
import random
import copy


# x ∈ [-5; 5], x ∈ R²
# global minimum: (0, 0)
def griewank(x, y):
    return (x**2 + y**2) / 4000 - np.cos(x) * np.cos(y / np.sqrt(2)) + 1


GRIEWANK_MIN = griewank(0, 0)


# x ∈ [-5,12; 5,12], x ∈ R²
# global minimum: (0, 0)
def rastrigin(x, y):
    return 20 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)


RASTRIGIN_MIN = rastrigin(0, 0)


class Individual:
    def __init__(self, x1: float, x2: float) -> None:
        self.x1 = x1
        self.x2 = x2


class EvolutionalOptimizer:

    def __init__(
        self,
        cost_func: Callable[[float, float], float],
        population: List["Individual"],
        mutation_strength: float,
        p_mutation: float,
        p_crossing: float,
        num_generations: int,
    ):
        self.cost_func = cost_func
        self.population = population
        self.mutation_strength = mutation_strength
        self.p_mutation = p_mutation
        self.p_crossing = p_crossing
        self.num_generations = num_generations

        self.best: Individual = copy.deepcopy(min(self.population, key=self.cost_func))
        self.population_size: int = len(population)

    # Tournament reproduction
    def reproduce(self) -> List["Individual"]:
        reproduced = []
        for _ in range(self.population_size):
            # Pick two individuals for the tournament
            p1 = self.population[random.randrange(0, self.population_size)]
            p2 = self.population[random.randrange(0, self.population_size)]
            # Evaluate
            s1 = self.cost_func(p1)
            s2 = self.cost_func(p2)
            # pick better one
            reproduced.append(copy.deepcopy(p1) if s1 < s2 else copy.deepcopy(p2))

        return reproduced

    # Averaging crossover
    def crossover(self) -> List["Individual"]:
        crossed = []
        for i in range(self.population_size):
            individual = copy.deepcopy(self.population[i])

            if self.p_crossing > random.random():
                partner = self.population[random.randrange(0, self.population_size)]
                w1, w2 = random.random(), random.random()
                individual.x1 = w1 * individual.x1 + (1 - w1) * partner.x1
                individual.x2 = w2 * individual.x2 + (1 - w2) * partner.x2

            crossed.append(individual)

        return crossed

    # Gaussian mutation
    def mutate(self) -> List["Individual"]:
        mutated = []
        for i in range(self.population_size):
            p = copy.deepcopy(self.population[i])

            if self.p_mutation > random.random():
                p.x1 = p.x1 + random.gauss(0, self.mutation_strength**2)
                p.x2 = p.x2 + random.gauss(0, self.mutation_strength**2)

            mutated.append(p)

        return mutated

    # Elite succession with k = 1
    def success(self) -> List["Individual"]:
        worst = max(self.population, key=self.cost_func)
        worst_index = self.population.index(worst)
        self.population.pop(worst_index)
        self.population.append(copy.deepcopy(self.best))
        self.best = copy.deepcopy(min(self.population, key=self.cost_func))
        return self.population

    def run(self) -> "Individual":
        for _ in range(self.num_generations):
            self.population = self.reproduce()
            self.population = self.crossover()
            self.population = self.mutate()
            self.population = self.success()

        return min(self.population, key=self.cost_func)


def random_population(min: float, max: float, size: int) -> List["Individual"]:
    return [
        Individual(random.uniform(min, max), random.uniform(min, max))
        for _ in range(size)
    ]


def plot_relations(
    title: str,
    cost_func: Callable[["Individual"], float],
    population=random_population(-4, -3, 120),
    mutation_strength=0.2,
    p_mutation=1,
    p_crossing=0.2,
    num_generations=50,
):
    mutation_strength = np.linspace(0, 2, 50)
    result = []

    for sigma in mutation_strength:
        optimizer = EvolutionalOptimizer(
            cost_func=cost_func,
            population=population,
            mutation_strength=sigma,
            p_mutation=p_mutation,
            p_crossing=p_crossing,
            num_generations=num_generations,
        )

        res = optimizer.run()
        print(res.x1, res.x2, optimizer.cost_func(res))
        result.append(cost_func(res))

    fig, ax = plt.subplots()
    ax.plot(mutation_strength, result)
    ax.set(xlabel="mutation strength", ylabel="result", title=title)
    ax.grid()
    fig.savefig(f"{title}.png")
    plt.show()


def main():

    def cost_func(individual: "Individual"):
        penalty_factor = 1

        penalty = 0
        if abs(individual.x1) > 5.12:
            penalty += (abs(individual.x1) - 5.12) * penalty_factor
        if abs(individual.x2) > 5.12:
            penalty += (abs(individual.x2) - 5.12) * penalty_factor

        return rastrigin(individual.x1, individual.x2) + penalty

    plot_relations(title="Griewank, mutation strength relation", cost_func=cost_func)


if __name__ == "__main__":
    main()
