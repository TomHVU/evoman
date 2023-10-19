import numpy as np
from deap import base, creator, tools, algorithms
from demo_controller import player_controller  
from evoman.environment import Environment  
import os
from functools import partial
from multiprocessing import Pool



# Define the number of hidden neurons
n_hidden_neurons = 10

# Define the experiment name
experiment_name = 'opt_generalist_Xinhai'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Define the fitness function
def fitness(individual):
    fitness, player_life, enemy_life, time = env.play(pcont=np.array(individual))
    return fitness,

# Initialize the EvoMan environment
env = Environment(experiment_name=experiment_name,
                    enemies=[1, 3, 5],  
                    multiplemode="yes",
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()



def run_mu_plus_lambda_generalist(training_enemies, initial_diverse=1, mutation_rate=0.2, 
                       Crossover_prob=0.5, population_size=100, generations=50, const=3, save_best=False):
    
    env.update_parameter('enemies', training_enemies)
    
    # Number of weights for multilayer with n_hidden_neurons
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    
    # Register individual and population creation operations in the toolbox
    toolbox.register("attr_float", np.random.uniform, -initial_diverse, initial_diverse)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register other operations
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selDoubleTournament, fitness_size=10, parsimony_size=1, fitness_first=True)
    toolbox.register("evaluate", fitness)

    
    # Statistics setup
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Fitness avg", np.mean)
    stats.register("Fitness std", np.std)
    stats.register("Fitness min", np.min)
    stats.register("Fitness max", np.max)
    
    # Run (μ + λ) evolutionary algorithm
    with Pool(24) as pool:
        toolbox.register("map", pool.map)

        population = toolbox.population(n=population_size)
        final_pop, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, 
                                                       lambda_=const * population_size,
                                                       cxpb=Crossover_prob, mutpb=mutation_rate, 
                                                       stats=stats, halloffame=None, ngen=generations)
        
    # Find the solution with the highest average fitness value
    avg_fitnesses = [np.mean(ind.fitness.values) for ind in final_pop]
    best_solution = final_pop[np.argmax(avg_fitnesses)]
    # print the avg fitness of the best solution
    print(f'Best solution fitness: {np.max(avg_fitnesses)}')

    
    # Save the best solution
    if save_best:
        if not os.path.exists('solutions_mu_plus_lambda_generalist'):
            os.makedirs('solutions_mu_plus_lambda_generalist')
        np.savetxt(f'solutions_mu_plus_lambda_generalist/solution_enemies_{training_enemies}.txt', best_solution)

    
    return logbook, best_solution

def validate_best_agent(best_agent, validation_enemies):
    total_gain = 0
    victory = 0
    fitness_values = {}  # Dictionary to store fitness values for each enemy
    
    for enemy in validation_enemies:
        fitness, player_life, enemy_life, time = env.run_single(enemy, np.array(best_agent), None)
        gain = player_life - enemy_life
        total_gain += gain
        if player_life > 0 and enemy_life <= 0:
            victory += 1
            
        fitness_values[enemy] = fitness 
    
    return total_gain, victory, fitness_values





if __name__ == '__main__':
    training_enemies = [1, 4, 7]
    validation_enemies = [2, 3, 5, 6, 8]
 

    logbook, best_solution = run_mu_plus_lambda_generalist(training_enemies, initial_diverse=1, mutation_rate=0.2, 
                                 Crossover_prob=0.5, population_size=200, generations=20, const=3, save_best=True)

    total_gain, victory, fitness_values = validate_best_agent(best_solution, validation_enemies)
    print(f'Total gain: {total_gain}')
    print(f'Victories: {victory}')
    print(f'Fitness values: {fitness_values}')


