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
                  enemies=[0],
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
npop = 10
# Number of generations
generations = 20
# Mutation rate Decide if we want to mutate or not.
mutation = 0.1
# Best solution from last generation, needed?
last_best = 0

def simulation(env, x):
    # In order fitness, player life, enemy life, (run)time
    f, p, e, t = env.play(pcont=x)
    return f 



# TODO 1: Create a population of random solutions.
# TODO 2: Evaluate the fitness of each solution.
# TODO 3: Remove the worst solutions? (how many?).
# TODO 4: Perform mutation and or crossover (select from fittest parents?).
# TODO 5: Repeat until a certain number of generations have passed.

# Parameters

# We can't edit the framework. However, we can change the output
# of this function. In this way, we can change the way the fitness
# function works. 
# # runs simulation
# def simulation(env,x):
#     f,p,e,t = env.play(pcont=x)
#     return f


# # normalizes
# def norm(x, pfit_pop):

#     if ( max(pfit_pop) - min(pfit_pop) ) > 0:
#         x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
#     else:
#         x_norm = 0

#     if x_norm <= 0:
#         x_norm = 0.0000000001
#     return x_norm

# NUMPY CAN ACTUALLY JUST DO THIS BETTER.

# Neural networks really like normalized values (most of them atleast).
# Although it is a bit strange, the paper says that values are normalized
# between -1 and 1 (which is not the case in the demo they use 0-1).
# So for now I take the demo as an example but it is something to keep in mind.
# def normal(x, fit_pop):
#     # Check if there is a difference between the best and worst fitness.
#     if (max(fit_pop)- min(fit_pop)) > 0:
#         # If there is a difference, normalize the fitnesslevel of current point.
#         x_norm = (x - min(fit_pop)) / (max(fit_pop)- min(fit_pop))
#         if x_norm <= 0:
#             x_norm = 0.0000000001
#     else:
#         # Set the normalized value to a very low number (taken from the demo)
#         x_norm = 0.0000000001
        
#     return x_norm

def evaluate(x):
    # This function evaluates the fitness of a solution.
    # This is done by running the simulation and returning the fitness.
    return np.array(list(map(lambda y: simulation(env,y), x)))

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
# (return the best one)? Yes, it compared two random actors. The best ones are already taken.
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