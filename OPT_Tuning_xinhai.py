import numpy as np
from deap import base, creator, tools, algorithms
from demo_controller import player_controller
from evoman.environment import Environment
import glob, os
from itertools import product
from multiprocessing import Pool
from functools import partial


n_hidden_neurons = 10

# Define the experiment name 
experiment_name = 'OPT_Tuning_xinhai'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# Define the fitness function
def fitness(individual):
    # Run the simulation with the given individual as weights
    _, player_life, enemy_life, time = env.play(pcont=np.array(individual))
    return 0.9 * (100 - enemy_life) + 0.1 * player_life - np.log(time),

# Set up the DEAP Genetic Algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def run_mu_plus_lambda(enemy, initial_diverse = 1, mutation_rate = 0.2, Crossover_prob = 0.5, 
population_size = 100, generations = 50, const = 3, save_best=False):
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
    toolbox.register("attr_float", np.random.uniform, -initial_diverse, initial_diverse)  
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
    final_pop, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_= const * population_size, 
                          cxpb=Crossover_prob, mutpb=mutation_rate, stats=stats, halloffame=None, 
                          ngen=generations)
    
    # Find the solution with the highest average fitness value
    avg_fitnesses = [np.mean(ind.fitness.values) for ind in final_pop]
    best_solution = final_pop[np.argmax(avg_fitnesses)]
    
    # Save the best solution only if save_best is True
    if save_best:
        if not os.path.exists('solutions_mu_plus_lambda'):
            os.makedirs('solutions_mu_plus_lambda')
        np.savetxt(f'solutions_mu_plus_lambda/best_solution_enemy_{enemy}.txt', best_solution)
    
    return logbook 

def run_mu_comma_lambda(enemy, initial_diverse = 1, mutation_rate = 0.2, Crossover_prob = 0.5, 
population_size = 100, generations = 50, const = 3, save_best=False):
    # Initialize the EvoMan environment 
    env = Environment(experiment_name=experiment_name,
                      enemies=[enemy],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)
    
    # Number of weights for multilayer with n_hidden_neurons
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    
    # Register individual and population creation operations in the toolbox
    toolbox.register("attr_float", np.random.uniform, -initial_diverse, initial_diverse)  
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
    
    # Here I choose the (μ, λ) evolutionary algorithm.
    population = toolbox.population(n=population_size)
    final_pop, logbook = algorithms.eaMuCommaLambda(population, toolbox, mu=population_size, lambda_= const * population_size, 
                          cxpb=Crossover_prob, mutpb=mutation_rate, stats=stats, halloffame=None, 
                          ngen=generations)
    
    # Find the solution with the highest average fitness value
    avg_fitnesses = [np.mean(ind.fitness.values) for ind in final_pop]
    best_solution = final_pop[np.argmax(avg_fitnesses)]
    
    # Save the best solution only if save_best is True
    if save_best:
        if not os.path.exists('solutions_mu_comma_lambda'):
            os.makedirs('solutions_mu_comma_lambda')
        np.savetxt(f'solutions_mu_comma_lambda/best_solution_enemy_{enemy}.txt', best_solution)

    return logbook






def grid_search(params, enemies, num_simulation):
    mutation_rate, crossover_prob, population_size, generations, const = params

    best_avg_for_each_enemy = []

    for enemy in enemies:
        best_this_run_all_simulations = []
        
        for sim in range(num_simulation):
            logbook1 = run_mu_plus_lambda(enemy, mutation_rate=mutation_rate,
                                          Crossover_prob=crossover_prob, population_size=population_size,
                                          generations=generations, const=const)
            logbook2 = run_mu_comma_lambda(enemy, mutation_rate=mutation_rate,
                                           Crossover_prob=crossover_prob, population_size=population_size,
                                           generations=generations, const=const)
            
            best_this_run = max(max(logbook1.select("Fitness avg")), max(logbook2.select("Fitness avg")))
            best_this_run_all_simulations.append(best_this_run)

        avg_best_this_run = sum(best_this_run_all_simulations) / num_simulation
        best_avg_for_each_enemy.append(avg_best_this_run)

    avg_best_fitness_across_all_enemies = sum(best_avg_for_each_enemy) / len(enemies)

    return (params, avg_best_fitness_across_all_enemies)


if __name__ == "__main__":
    # Parameter space
    mutation_rates = [0.1, 0.2, 0.3]
    crossover_probs = [0.4, 0.5, 0.6]
    population_sizes = [100]
    generations_list = [70]
    consts = [2, 3, 7]
    enemies = [1, 4, 5]
    num_simulation = 10  # Number of simulations for each hyperparameter set

    all_params = list(product(mutation_rates, crossover_probs, population_sizes, generations_list, consts))

    partial_grid_search = partial(grid_search, enemies=enemies, num_simulation=num_simulation)

    # Multiprocessing
    with Pool(8
              ) as pool:
        results = pool.map(partial_grid_search, all_params)

    # Find and print the best result
    best_params, best_avg_fitness_across_all_enemies = max(results, key=lambda x: x[1])
    print(f"Best hyperparameters across all enemies: {best_params}")
    print(f"Best average Fitness avg across all enemies: {best_avg_fitness_across_all_enemies}")


    
