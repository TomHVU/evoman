################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
import numpy as np
import time

from evoman.environment import Environment
from demo_controller import player_controller


# Not using visuals (faster).
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"



experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


n_neurons = 10

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller= player_controller(n_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# For now copied from optimization_specialist_demo.py

env.state_to_log()

set_time = time.time()

# No idea yet what this does exactly.
run_mode = 'test' # train or test

# Sinds we are dealing with a neural network, we need to know the number of weights and set them.
n_vars = (env.get_num_sensors()+1)*n_neurons + (n_neurons+1)*5

# Note: this is what I think it means.
# Upper bound
dom_u = 1
# Lower bound
dom_l = -1
# Population size
npop = 1000
# Number of generations
generations = 5
# Mutation rate Decide if we want to mutate or not.
mutation = 0.1
# Best solution from last generation, needed?
last_best = 0
# Percentage of population to kill.
kill_percentage = .49



def simulation(env, x):
    # In order fitness, player life, enemy life, (run)time
    f, p, e, t = env.play(pcont=x)
    # print(f)
    return f 

def create_population():
    # This function creates a population of random solutions.
    np.random.seed(42)
    return np.random.uniform(dom_l, dom_u, (npop, n_vars))


def evolution_step(population):
    scores = evaluate_population(population)
    reduced_pop = remove_worst_performers(scores, kill_percentage)
    print(scores[0][0])
    new_pop = evo_alg(reduced_pop)
    
    
    return new_pop


def evaluate_population(population):
    members = []
    for i, pop in enumerate(population):
        members.append([simulation(env, pop), pop])
        # print(members[i])
    
    return sorted(members, key=lambda x: -x[0])
    
    # return members
    # np.sort(members, axis=1)

# No stupid protection against trying to remove more than there are.
def remove_worst_performers(population, kill):
    if kill > 0.9:
        print("Really? You want to kill more than 9/10 of the population?")
    # This function removes the worst performers from the population.
    
    # This is an option population will > tho.
    n_performers = len(population)
    print(n_performers)
    remove = int(n_performers * kill)
    reduced_population = population[:-remove or None]
    return reduced_population
    # # or keep best x performers?
    
    # return population[:npop]
    
    

# This function performs the evolutionary algorithm.
# So feel free to change it (in fact please do).

# I (Sander) just chose simple arithmetic crossover.
def evo_alg(scores):
    # For now it just comines all solutions with the best solution.
    best_performer = scores[0][1]
    
    n_scores = len(scores)
    
    new_population = []
    new_population.append(best_performer)
    
    for i in range(1, n_scores):
        new_population.append((best_performer + scores[i][1])/2)
        new_population.append(scores[i][1])
        
    return new_population


population = create_population()

for i in range(generations):
    gen = i+1
    print("Gen: " + str(gen))
    population = evolution_step(population)

    






# TODO 1: Create a population of random solutions.
# TODO 2: Evaluate the fitness of each solution.
# TODO 3: Remove the worst solutions? (how many?).
# TODO 4: Perform mutation and or crossover (select from fittest parents?).
# TODO 5: Repeat until a certain number of generations have passed.




# def evaluate(x):
#     # This function evaluates the fitness of a solution.
#     # This is done by running the simulation and returning the fitness.
#     return np.array(list(map(lambda y: simulation(env,y), x)))

# def limits(x):
#     # We need to keep in mind that the character can't go out of bounds.
#     if x > dom_u:
#         return dom_u
#     elif x < dom_l:
#         return dom_l
#     else:
#         return x
    
# Not really sure what this does.
# Does it select two actors from the population and compares them
# (return the best one)?
# def tournament(pop):
#     c1 = np.random.randint(0, pop.shape[0], 1)
#     c2 = np.random.randint(0, pop.shape[0], 1)
    
#     if fit_pop[c1] > fit_pop[c2]:
#         return pop[c1][0]
#     else:
#         return pop[c2][0]
    


# We need to either create a population or load one.
# if not os.path.exists('test/' + experiment_name+'/evoman_solstate'):
#     print( '\nNEW EVOLUTION\n')
#     # Now we need to create a random uniform population.
#     pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    
    
    
#     # Get a list with how the individuals perform.
#     # For now it only returns the fitness, but we could use more 
#     # information (like the time it took to run the simulation, player life and enemy life).
#     fit_pop = evaluate(pop)
#     # Get the best performing individual.
#     best = np.argmax(fit_pop)
#     # Get the average fitness of the population.
#     mean = np.mean(fit_pop)
#     # Get the standard deviation of the fitness of the population.
#     std = np.std(fit_pop)
#     # ?
#     ini_g = 0
#     solutions = [pop, fit_pop]
#     env.update_solutions(solutions)
    
    
# else:
#     print(' \nCONTINUING EVOLUTION\n' )
#     # Set the environment to a load state
#     env.load_state()
#     pop = env.solutions[0]
#     fit_pop = env.solutions[1]
    
#     best = np.argmax(fit_pop)
#     mean = np.mean(fit_pop)
#     std = np.std(fit_pop)
    
#     # Get the final generation from file.
#     file_aux = open('test/' + experiment_name+'/gen.txt','r')
#     ini_g = int(file_aux.readline())
#     file_aux.close()