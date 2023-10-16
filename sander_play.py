from evoman.environment import Environment
from demo_controller import player_controller 
import numpy as np
import os

experiment_name = 'Sanders_tests'
n_hidden_neurons = 10  


headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"



env = Environment(experiment_name=experiment_name,
                    #   enemies=[enemy],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)


sol = np.loadtxt(experiment_name + ' _solutions/best_solution50_hof2.txt')
# sol = np.loadtxt(experiment_name + ' solutions/beat6.txt')
# individual = 2*(sol - np.min(sol))/(np.max(sol) - np.min(sol)) - 1

final = []

for enemy in range(1, 9):
    print("-----")
    print(f"Enemy: {enemy}")
    env.update_parameter('enemies', [enemy])
    print("Playing.....")
    _, player_life, enemy_life, time = env.play(pcont=np.array(sol))
    print("Score:")
    score = 0.9 * (100 - enemy_life) + 0.1 * player_life - np.log(time)
    print(score)
    
    if enemy_life == 0:
        final.append((enemy, score, "Win!"))
    else:
        final.append((enemy, score, "Lose :("))
    
    
    # env.play(individual)
    # print("Player life:")
    # print(env.get_playerlife())
    # print("Enemy life:")
    # print(env.get_enemylife())
    # print('-----')

print("All scores:")
print(final)