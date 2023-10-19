import numpy as np
from deap import base, creator, tools, algorithms
from deap.tools import MultiStatistics
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
experiment_name = 'NSGA_III_xinhai'
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

creator.create("FitnessMulti", base.Fitness, weights=(2.0, -18.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

# Register individual and population creation operations in the toolbox
toolbox.register("attr_float", np.random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


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


def multi_objective_fitness(individual):

    _, player_life, enemy_life, time = env.play(pcont=np.array(individual))    
    log_time = np.log(time) if time > 0 else 0


    return 0.1 * player_life, 0.9 * enemy_life, log_time

def NSGA_II(trianing_enemies, mutation_rate=0.2, Crossover_prob=0.5, population_size=100, 
            generations=50, const=3, save_best=False,  num_procs = 1):

    toolbox.register("evaluate", multi_objective_fitness)

    # setup for mating, mutation, and selection
    toolbox.register("mate", tools.cxSimulatedBinary, eta=20)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=20, low=-1, up=1, indpb=0.2)
    toolbox.register("select", tools.selNSGA3, ref_points=tools.uniform_reference_points(nobj=3, p=12))
    
    # Statistics
    stats = MultiStatistics(fitness1=tools.Statistics(key=lambda ind: ind.fitness.values[0]),
                            fitness2=tools.Statistics(key=lambda ind: ind.fitness.values[1]),
                            fitness3=tools.Statistics(key=lambda ind: ind.fitness.values[2]))
    
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("max", np.max)

    

    with Pool(num_procs) as pool:
        toolbox.register("map", pool.map)
        population = toolbox.population(n=population_size)
        
        final_pop, logbook = algorithms.eaMuCommaLambda(population, toolbox, mu=population_size, lambda_=const * population_size,
                                cxpb=Crossover_prob, mutpb=mutation_rate, ngen=generations,
                                stats=stats, halloffame=None, verbose=True)
        

    best_agent = tools.selBest(population, 1)[0]

    print('Best evaluation values: ', best_agent.fitness.values)

        # using the best solution to play the game and figure the victory rate for training enemies
    victory = 0
    for enemy in trianing_enemies:
        bfitness, bplayer_life, benemy_life, btime = env.run_single(enemy, np.array(best_agent), None)
        # print the fitness for each enemy
        print(f'Bets solution -- Enemy {enemy} fitness: {bfitness}')
        if bplayer_life > 0 and benemy_life <= 0:
            victory += 1

    print(f'Victory: {victory}, Number of enemies: {len(trianing_enemies)}')


        # Save the best solution
    if save_best:
        if not os.path.exists('solutions_NSGA_II'):
            os.makedirs('solutions_NSGA_II')
        np.savetxt(f'solutions_NSGA_II/solution_enemies_{trianing_enemies}.txt', best_agent)

    return logbook, best_agent

def data_collect():
    
    big_logbook = Logbook()
    info_dict = {}

    training_enemies = [1, 4, 7]

    # Run the algorithm 10 times for 10 independent runs
    for run in range(1, 11):
        print(f"Starting run {run}...")

        # Update parameters to match those in the triple-quoted function
        logbook, best_solution = NSGA_II(training_enemies, mutation_rate=0.05, 
                                         Crossover_prob=0.3, population_size=100, 
                                         generations=100, num_procs=24, save_best=False)

        for record in logbook:
            record['Run'] = run

        big_logbook += logbook

        # Test the final best solution against all enemies 5 times
        all_enemies = list(range(1, 9))
        results = {}
        
        for enemy in all_enemies:
            gains = []
            victories = []
            fitness = [] 
            
            for _ in range(5):  # Test 5 times
                gain, v, fit = validate_best_agent(best_solution, [enemy])
                gains.append(gain)
                victories.append(v)
                fitness.append(fit[enemy])  
            
            results[enemy] = {'gain': gains, 'victory': victories, 'fitness': fitness} 

        info_dict[run] = results

        print(f"Finished run {run}.")
        print("----------------------------------------------------------------------")

    # Save 
    with open("big_logbook_NSGA_147.pkl", "wb") as f:
        pickle.dump(big_logbook, f)

    with open("info_dict_NSGA_147.pkl", "wb") as f:
        pickle.dump(info_dict, f)



if __name__ == "__main__":
    
    data_collect()

    '''
    num_runs = 4
    NSGA_gain = []
    NSGA_victory = []

    # Repeat the code num_runs times
    for i in range(num_runs):
        _, best_solu = NSGA_II([1, 4 ,7], mutation_rate=0.03, Crossover_prob = 0.3, population_size = 100, generations=150, num_procs=24, save_best=False)
        total_gain_v, victory,_ = validate_best_agent(best_solu, [1,2,3,4,5,6,7,8])
        NSGA_gain.append(total_gain_v)
        NSGA_victory.append(victory)

    # Print the average result
    print('Average total gain:', np.mean(NSGA_gain))
    print('Average victory:', np.mean(NSGA_victory))
    '''