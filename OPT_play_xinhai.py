
import sys, os
from evoman.environment import Environment
from demo_controller import player_controller 
import numpy as np

experiment_name = 'OPT_play_xinhai'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


n_hidden_neurons = 10  

# initializes environment with static enemy and ai player
env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  speed="normal",
                  enemymode="static",
                  level=2,
                  visuals=True)

# Demonstrate optimized solutions for all enemies from #1 to #8
for en in range(1, 9):
    # Update the enemy
    env.update_parameter('enemies', [en])

    # Load the saved solution for the enemy
    sol = np.loadtxt(f'solutions_optimized/best_solution_enemy_{en}.txt')
    print(f'\nLOADING OPTIMIZED SOLUTION FOR ENEMY {en}\n')
    env.play(sol)

