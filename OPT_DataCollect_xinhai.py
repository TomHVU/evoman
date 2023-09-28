import numpy as np
from deap import base, creator, tools, algorithms
from demo_controller import player_controller  
from evoman.environment import Environment  
import glob, os
import pandas as pd
import time
from multiprocessing import Pool




# Number of hidden neurons
n_hidden_neurons = 10

# Experiment name
experiment_name = 'OPT_DataCollect_xinhai'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Create a new multi-objective fitness type
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Define the combined evaluation function
def combined_evaluation(individual):
    _, player_life, enemy_life, time = env.play(pcont=np.array(individual))
    fitness_value = 0.9 * (100 - enemy_life) + 0.1 * player_life - np.log(time)
    individual_gain_value = player_life - enemy_life
    return fitness_value, individual_gain_value,

toolbox = base.Toolbox()

# Rest of your code remains mostly the same with some modifications
def run_mu_plus_lambda(enemy, initial_diverse=1, mutation_rate=0.2, Crossover_prob=0.5, 
                       population_size=100, generations=50, save_best=False):
    global env
    env = Environment(experiment_name=experiment_name,
                      enemies=[enemy],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    global n_vars
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    
    toolbox.register("attr_float", np.random.uniform, -initial_diverse, initial_diverse)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)  # Using NSGA-II for selection
    toolbox.register("evaluate", combined_evaluation)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    population = toolbox.population(n=population_size)
    final_pop, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=3 * population_size,
                                                   cxpb=Crossover_prob, mutpb=mutation_rate, stats=stats, halloffame=None,
                                                   ngen=generations)

    if save_best:
        if not os.path.exists('solutions_mu_plus_lambda'):
            os.makedirs('solutions_mu_plus_lambda')
        
        best_solution = tools.selBest(final_pop, 1)[0]
        np.savetxt(f'solutions_mu_plus_lambda/best_solution_enemy_{enemy}.txt', best_solution)

    return logbook


def run_mu_comma_lambda(enemy, initial_diverse = 1, mutation_rate = 0.2, Crossover_prob = 0.5, 
                        population_size = 100, generations = 50, save_best=False):
    global env
    env = Environment(experiment_name=experiment_name,
                      enemies=[enemy],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    global n_vars
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    
    toolbox.register("attr_float", np.random.uniform, -initial_diverse, initial_diverse)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)  # Using NSGA-II for selection
    toolbox.register("evaluate", combined_evaluation)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    population = toolbox.population(n=population_size)
    final_pop, logbook = algorithms.eaMuCommaLambda(population, toolbox, mu=population_size, lambda_=3 * population_size,
                                                    cxpb=Crossover_prob, mutpb=mutation_rate, stats=stats, halloffame=None,
                                                    ngen=generations)

    if save_best:
        if not os.path.exists('solutions_mu_comma_lambda'):
            os.makedirs('solutions_mu_comma_lambda')
        
        best_solution = tools.selBest(final_pop, 1)[0]
        np.savetxt(f'solutions_mu_comma_lambda/best_solution_enemy_{enemy}.txt', best_solution)

    return logbook





import pandas as pd  # Importing pandas for DataFrame operations

def collect_data_from_logbook(logbook):
    gen_stats = logbook  # Assuming logbook is passed correctly
    data_list = []  # Initialize an empty list to hold dictionaries for each generation
    
    for generation, entry in enumerate(gen_stats):
        avg_values = entry['avg']
        std_values = entry['std']
        min_values = entry['min']
        max_values = entry['max']
        
        # Prepare the output as a dictionary
        output = {
            'Generation': generation,
            'Fitness_Avg': avg_values[0],
            'Fitness_Std': std_values[0],
            'Fitness_Max': max_values[0],
            'Gain_Avg': avg_values[1],
            'Gain_Std': std_values[1],
            'Gain_Max': max_values[1]
        }
        
        data_list.append(output)
        
    return data_list

def run_one_simulation(args):
    enemy, initial_diverse, simulation, pop_size, gen = args
    logbook_plus = run_mu_plus_lambda(enemy, initial_diverse=initial_diverse, population_size=pop_size, generations=gen)
    logbook_comma = run_mu_comma_lambda(enemy, initial_diverse=initial_diverse, population_size=pop_size, generations=gen)
    return enemy, initial_diverse, simulation, logbook_plus, logbook_comma

def run_sim_and_gather_data(enemy, ini_div, num_sim, pop_size, gen):
    start_time = time.time()  # Record start time
    
    columns = ['Enemy', 'Initial_Diversity', 'Simulation', 'Generation', 'Fitness_Avg', 'Fitness_Std', 'Fitness_Max', 'Gain_Avg', 'Gain_Std', 'Gain_Max']
    df_mu_plus = pd.DataFrame(columns=columns)
    df_mu_comma = pd.DataFrame(columns=columns)
    
    # Prepare arguments for multiprocessing
    args_list = [(e, id, s, pop_size, gen) for e in enemy for id in ini_div for s in range(num_sim)]

    # Create a pool of 32 workers
    with Pool(8) as p:
        results = p.map(run_one_simulation, args_list)
    
    # Process results and store them in DataFrame
    for enemy, initial_diverse, simulation, logbook_plus, logbook_comma in results:
        data_plus_list = collect_data_from_logbook(logbook_plus)
        data_comma_list = collect_data_from_logbook(logbook_comma)
        
        for data_plus, data_comma in zip(data_plus_list, data_comma_list):
            data_plus.update({'Enemy': enemy, 'Initial_Diversity': initial_diverse, 'Simulation': simulation})
            data_comma.update({'Enemy': enemy, 'Initial_Diversity': initial_diverse, 'Simulation': simulation})
            df_mu_plus = pd.concat([df_mu_plus, pd.DataFrame([data_plus])], ignore_index=True)
            df_mu_comma = pd.concat([df_mu_comma, pd.DataFrame([data_comma])], ignore_index=True)

    df_mu_plus.to_csv('mu_plus_lambda_metrics.csv', index=False)
    df_mu_comma.to_csv('mu_comma_lambda_metrics.csv', index=False)

    end_time = time.time()  # Record end time
    print(f"Total time taken: {end_time - start_time} seconds")  # Print total time taken




if __name__ == "__main__":
    enemy = [1, 4, 5]
    ini_div = [0, 0.1 , 0.4 ,0.7, 1]
    num_sim = 10
    pop_size = 100
    gen = 70
    run_sim_and_gather_data(enemy, ini_div, num_sim, pop_size, gen)