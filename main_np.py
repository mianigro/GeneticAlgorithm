import random
import numpy as np


class GeneticAlgorithm:
    
    def __init__(self):
        # Hyperparameters
        self.n_solutions = 1000
        self.sol_order = 10  # Initial solutions are between 0-1, this sets order of magnitude
        self.fitness_lim = 9999  # Goal fitness level

        # Setup
        self.goal_variable = 100  # The solution to the equation we are solving
        self.n_vars = 3  # Amount of variables
        self.found_best = False

    def gen_seed(self):
        out_seed = np.random.rand(self.n_solutions, self.n_vars) * self.sol_order
        return out_seed

    def goal_function(self, x, y, z):
        return 5 * x ** 2 + 7 * y - 20 * z - self.goal_variable

    def fitness(self, solution):
        result = self.goal_function(solution[0], solution[1], solution[2])

        if result == 0:
            return 9999999999999999
        else:
            return abs(1/result)  # Smaller x gets, the bigger the fitness is

    def test_solutions(self, sol_array):
        fitness_array = np.apply_along_axis(self.fitness, 1, sol_array)

        return fitness_array, sol_array

    def select_best(self, fitness_array, generation_sols):
        generation_sols = np.reshape(generation_sols, (len(generation_sols), self.n_vars))
        index_points = np.argsort(fitness_array)
        index_points = np.flip(index_points)
        
        # Returns the solutions ordered in descending order by fitness
        return fitness_array[index_points], generation_sols[index_points]

    def crossover_function(self, generation_x):
        # Make blank array
        crossed_solutions = np.zeros((len(generation_x), self.n_vars))

        # Save top 10 solutions
        crossed_solutions[:10] = generation_x[:10]
        generation_x = generation_x[:len(generation_x) - 10]

        # Use top n - 10 best solutions to generate children
        crossed_solutions[11::2, :1] = generation_x[::2, :1]
        crossed_solutions[10::2, :1] = generation_x[1::2, :1]

        crossed_solutions[10:, 1:2] = generation_x[:, 1:2]

        crossed_solutions[11::2, 2:] = generation_x[::2, 2:]
        crossed_solutions[10::2, 2:] = generation_x[1::2, 2:]

        return crossed_solutions

    def mutation(self, crossed_solutions):
        # Mutates a random choice for each line by between -2% and +2%
        for x in range(len(crossed_solutions)):
            n = random.randint(0, self.n_vars-1)
            crossed_solutions[x][n] = crossed_solutions[x][n] * np.random.uniform(0.98, 1.02, 1) 

        return crossed_solutions


alg_gen = GeneticAlgorithm()


# Algorithm
new_seed = alg_gen.gen_seed()
for i in range(100000):
    # 1. Generate seed
    seed = new_seed

    # 2. Put through fitness function to determine fitness
    fitness, solution_array = alg_gen.test_solutions(seed)

    if not alg_gen.found_best:
        # 3. Select the best solutions
        best_fitness, best_sols = alg_gen.select_best(fitness, solution_array)
        
        print(f"Iteration: {i} - Best Fitness: {int(best_fitness[0])}")
        
        if best_fitness[0] >= alg_gen.fitness_lim:
            alg_gen.found_best = True

        # 4. Crossover solutions
        crossed_sols = alg_gen.crossover_function(best_sols)

        # 5. Mutate solutions
        new_seed = alg_gen.mutation(crossed_sols)

    else:
        print(f"Found Best:\n- Solution: {best_sols[0]}\n- Fitness: {best_fitness[0]}")

        break
