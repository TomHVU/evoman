import numpy as np
from deap import base, creator, tools, algorithms
from demo_controller import player_controller  
from evoman.environment import Environment  
import os
from functools import partial
from multiprocessing import Pool
from deap.tools import Logbook  
import pickle



# Define the number of hidden neurons
n_hidden_neurons = 10

# Define the experiment name
experiment_name = 'opt_generalist_Xinhai'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


# Initialize the EvoMan environment
env = Environment(experiment_name=experiment_name,
                    enemies=[1, 4, 7],  
                    multiplemode="yes",
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)

# Define the fitness function
def fitness(individual):
    fitness, player_life, enemy_life, time = env.play(pcont=np.array(individual))
    return fitness,

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def run_EA(training_enemies, initial_diverse=1, mutation_rate=0.2, 
                                Crossover_prob=0.5, tournament_size = 3, population_size=100, generations=50, const=3,
                                save_best=False, algorithm_type='eaMuCommaLambda', num_procs = 1):
    
    env.update_parameter('enemies', training_enemies)
    
    # Number of weights for multilayer with n_hidden_neurons
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    
    # Register individual and population creation operations in the toolbox
    toolbox.register("attr_float", np.random.uniform, -initial_diverse, initial_diverse)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register other operations
    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize= tournament_size)
    toolbox.register("evaluate", fitness)

    
    # Statistics setup
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Fitness avg", np.mean)
    stats.register("Fitness std", np.std)
    stats.register("Fitness min", np.min)
    stats.register("Fitness max", np.max)

     # Algorithm setup   
    with Pool(num_procs) as pool:
        toolbox.register("map", pool.map)
        population = toolbox.population(n=population_size)
        
        if algorithm_type == 'eaMuPlusLambda':
            final_pop, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, 
                                                           lambda_=const * population_size,
                                                           cxpb=Crossover_prob, mutpb=mutation_rate, 
                                                           stats=stats, halloffame=None, ngen=generations)
        elif algorithm_type == 'eaMuCommaLambda':
            final_pop, logbook = algorithms.eaMuCommaLambda(population, toolbox, mu=population_size, 
                                                            lambda_=const * population_size,
                                                            cxpb=Crossover_prob, mutpb=mutation_rate, 
                                                            stats=stats, halloffame=None, ngen=generations)
        else:
            raise ValueError("Invalid algorithm_type. Choose between 'eaMuPlusLambda' and 'eaMuCommaLambda'")
    
        
    # Find the best solution
    best_solution = tools.selBest(final_pop, k=1)[0]
    # print the best solution fitness
    print(f'Best solution fitness: {best_solution.fitness.values[0]}')

    # using the best solution to play the game and figure the victory rate for training enemies
    victory = 0
    for enemy in training_enemies:
        bfitness, bplayer_life, benemy_life, btime = env.run_single(enemy, np.array(best_solution), None)
        # print the fitness for each enemy
        print(f'Bets solution -- Enemy {enemy} fitness: {bfitness}')
        if bplayer_life > 0 and benemy_life <= 0:
            victory += 1

    print(f'Victory: {victory}, Number of enemies: {len(training_enemies)}')


    
    # Save the best solution
    if save_best:
        if not os.path.exists('solutions_mu_plus_lambda_generalist'):
            os.makedirs('solutions_mu_plus_lambda_generalist')
        np.savetxt(f'solutions_mu_plus_lambda_generalist/solution_enemies_{training_enemies}(5).txt', best_solution)

    
    return logbook, best_solution


def validate_best_agent(best_agent, validation_enemies):
    total_gain_v = 0
    victory = 0
    fitness_values = {}  # Dictionary to store fitness values for each enemy

    for enemy in validation_enemies:
        vfitness, vplayer_life, venemy_life, vtime = env.run_single(enemy, np.array(best_agent), None)
        gain = vplayer_life - venemy_life
        total_gain_v += gain
        fitness_values[enemy] = vfitness  
        if vplayer_life > 0 and venemy_life <= 0:
            victory += 1

    return total_gain_v, victory, fitness_values 



def data_collect():
    
    big_logbook_c_147 = Logbook()
    big_logbook_p_147 = Logbook()
    info_dict_c_147 = {}
    info_dict_p_147 = {}

    training_enemies = [1, 4, 7]

    # Run the algorithm 10 times for 10 independent runs
    for run in range(1, 11):
        print(f"Starting run {run}...")

        # Run the algorithm
        logbook_c, best_solution_c = run_EA(training_enemies, initial_diverse=1, mutation_rate=0.2, 
                                            Crossover_prob=0.5, tournament_size=5, population_size=200, 
                                            generations=100, const=3, save_best=False, algorithm_type='eaMuCommaLambda',
                                            num_procs=24)
        logbook_p, best_solution_p = run_EA(training_enemies, initial_diverse=1, mutation_rate=0.2,
                                            Crossover_prob=0.5, tournament_size=5, population_size=200,
                                            generations=100, const=3, save_best=False, algorithm_type='eaMuPlusLambda',
                                            num_procs=24)

        for record_c in logbook_c:
            record_c['Run'] = run
        for record_p in logbook_p:
            record_p['Run'] = run

        big_logbook_c_147 += logbook_c
        big_logbook_p_147 += logbook_p

        # Test the final best solution against all enemies 5 times
        all_enemies = list(range(1, 9))
        results_c = {}
        results_p = {}
        
        for enemy in all_enemies:
            gains_c = []
            gains_p = []
            victories_c = []
            victories_p = []
            fitness_c = []  # Store fitness values for 'eaMuCommaLambda'
            fitness_p = []  # Store fitness values for 'eaMuPlusLambda'
            
            for _ in range(5):  # Test 5 times
                gain1, v1, fitness_values_c = validate_best_agent(best_solution_c, [enemy])
                gains_c.append(gain1)
                victories_c.append(v1)
                fitness_c.append(fitness_values_c[enemy])  
                
                gain2, v2, fitness_values_p = validate_best_agent(best_solution_p, [enemy])
                gains_p.append(gain2)
                victories_p.append(v2)
                fitness_p.append(fitness_values_p[enemy]) 

            results_c[enemy] = {'gain': gains_c, 'victory': victories_c, 'fitness': fitness_c}  
            results_p[enemy] = {'gain': gains_p, 'victory': victories_p, 'fitness': fitness_p}  
        
        # Existing code to save the results into info_dict
        info_dict_c_147[run] = results_c
        info_dict_p_147[run] = results_p

        print(f"Finished run {run}.")
        print("----------------------------------------------------------------------")

    # Save 
    with open("big_logbook_c_147.pkl", "wb") as f:
        pickle.dump(big_logbook_c_147, f)

    with open("info_dict_c_147.pkl", "wb") as f:
        pickle.dump(info_dict_c_147, f)

    with open("big_logbook_p_147.pkl", "wb") as f:
        pickle.dump(big_logbook_p_147, f)

    with open("info_dict_p_147.pkl", "wb") as f:
        pickle.dump(info_dict_p_147, f)



if __name__ == '__main__':
    
    # data_collect()
    

    _, best_solu = run_EA([1, 4, 7], initial_diverse=1, mutation_rate=0.2,
                                        Crossover_prob=0.5, tournament_size=5, population_size=100,
                                        generations=100, const=3, save_best=True, algorithm_type='eaMuCommaLambda',
                                        num_procs=24)
    
    total_gain_v, victory,_ = validate_best_agent(best_solu, [1,2,3,4,5,6,7,8])
    print('total gain:', total_gain_v)
    print('victory:', victory)
    

    

