import numpy as np
from deap import base, creator, tools, algorithms
from demo_controller import player_controller
from evoman.environment import Environment
import glob, os


n_hidden_neurons = 10

# Define the experiment name 
experiment_name = 'OPT_xinhai'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Genetic Algorithm parameters
mutation_rate = 0.2
population_size = 100
generations = 100

# Define the fitness function
def fitness(individual):
    # Run the simulation with the given individual as weights
    _, player_life, enemy_life, time = env.play(pcont=np.array(individual))
    return 0.9 * (100 - enemy_life) + 0.1 * player_life - np.log(time),

# Set up the DEAP Genetic Algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def run_optimization(enemy, save_best=False):
    # Initialize the EvoMan environment 
    global env
    env = Environment(experiment_name=experiment_name,
                      enemies=[enemy],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)
    
    # Number of weights for multilayer with n_hidden_neurons
    global n_vars
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    
    # Register individual and population creation operations in the toolbox
    toolbox.register("attr_float", np.random.uniform, -1, 1)  # Assuming weights are in the range [-1, 1]
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register other operations in the toolbox
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", fitness)

    # Statistics setup
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Fitness avg", np.mean)
    stats.register("Fitness std", np.std)
    stats.register("Fitness min", np.min)
    stats.register("Fitness max", np.max)
    
    # Here I choose the (μ + λ) evolutionary algorithm.
    population = toolbox.population(n=population_size)
    final_pop, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=200, 
                          cxpb=0.5, mutpb=mutation_rate, stats=stats, halloffame=None, 
                          ngen=generations)
    
    # Find the solution with the highest average fitness value
    avg_fitnesses = [np.mean(ind.fitness.values) for ind in final_pop]
    best_solution = final_pop[np.argmax(avg_fitnesses)]
    
    # Save the best solution only if save_best is True
    if save_best:
        if not os.path.exists('solutions_optimized'):
            os.makedirs('solutions_optimized')
        np.savetxt(f'solutions_optimized/best_solution_enemy_{enemy}.txt', best_solution)
    
    return logbook 


if __name__ == "__main__":
    for enemy in range(1, 9):
        print(f"Optimizing for Enemy {enemy}")
        run_optimization(enemy, save_best=True)
        print(f"Optimization for Enemy {enemy} completed\n")
