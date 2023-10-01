import numpy as np
from deap import base, creator, tools, algorithms
from demo_controller import player_controller
from evoman.environment import Environment
import glob, os
from itertools import product
import time

# Define the experiment name 
experiment_name = 'Random_search_xinhai'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

def fitness(individual):
    _, player_life, enemy_life, time = env.play(pcont=np.array(individual))
    return 0.9 * (100 - enemy_life) + 0.1 * player_life - np.log(time),

# Random search function
def random_search(enemy, iterations=1000):
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
    
    global n_vars
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    
    # Initialize variables to keep track of the best solution
    best_fitness = -np.inf  # Initialize with negative infinity
    best_individual = None
    
    for i in range(iterations):
        # Create a random individual
        individual = np.random.uniform(-1, 1, n_vars)
        
        # Evaluate its fitness
        current_fitness = fitness(individual)[0]
        
        # Update the best solution if needed
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_individual = individual
            
    return best_fitness, best_individual


def random_search_multiple_runs(enemy, num_runs=10):
    total_best_fitness = 0
    time_records = []  
    
    for run in range(num_runs):
        start_time = time.time()  
        
        best_fitness_tuple = random_search(enemy) 
        best_fitness = best_fitness_tuple[0]  # Extract the first element from the tuple
        total_best_fitness += best_fitness
        
        end_time = time.time() 
        elapsed_time = end_time - start_time  
        time_records.append(elapsed_time)  
        
        # Calculate and print the percentage of work done
        percentage_done = ((run + 1) / num_runs) * 100
        print(f"Run {run + 1} completed. Time taken: {elapsed_time:.2f} seconds. Percentage of work done: {percentage_done}%.")
    
    average_best_fitness = total_best_fitness / num_runs
    average_time = sum(time_records) / len(time_records)  # Calculate the average time taken
    
    return average_best_fitness, average_time


# Run the random search 10 times for enemies 1, 4, 5 and get the average highest fitness values
enemies_to_test = [1, 4, 5]
average_highest_fitness_values = {}
average_time_taken = {}

for enemy in enemies_to_test:
    average_best_fitness, average_time = random_search_multiple_runs(enemy)
    average_highest_fitness_values[enemy] = average_best_fitness
    average_time_taken[enemy] = average_time
    print(f"The average highest fitness value over 10 runs for enemy {enemy} is {average_best_fitness}.")
    print(f"The average time taken for each run for enemy {enemy} is {average_time:.2f} seconds.")
