import numpy as np
from deap import base, creator, tools, algorithms
from demo_controller import player_controller
from evoman.environment import Environment
import glob, os
import multiprocessing
import random
import pickle

n_hidden_neurons = 10

# Define the experiment name 
experiment_name = 'Sanders_tests'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Genetic Algorithm parameters
mutation_rate = 0.5
population_size = 100
generations = 70

# Define the fitness function
def fitness(individual):
    # print(individual)
    # exit()
    env = Environment(experiment_name=experiment_name,
                      playermode="ai",
                      player_controller=player_controller(10),
                    #   multiplemode="yes",
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    games = [0,0,0,0,0]
    individual = 2*(individual - np.min(individual))/(np.max(individual) - np.min(individual)) - 1
    # list4 = [2, 3, 5, 7]
    
    list4 = [2, 4, 5, 8]
    count = 0
    
    for i in range(1, 9):
        env.update_parameter('enemies', [i])
        _, player_life, enemy_life, time = env.play(pcont=np.array(individual))
        score = 0.9 * (100 - enemy_life) + 0.1 * player_life - np.log(time)
        if enemy_life == 0:
            score += 20
                
        if i not in list4:
            games[4] += score
        else:
            games[count] = score
            count += 1

    games[4] = games[4]/4
    return games


def fitness2(individual):
    # print(individual)
    # exit()
    env = Environment(experiment_name=experiment_name,
                      playermode="ai",
                      player_controller=player_controller(10),
                    #   multiplemode="yes",
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    games = []
    # games.append(0)
    # individual = 2*(individual - np.min(individual))/(np.max(individual) - np.min(individual)) - 1
    # list4 = [1, 3, 5, 8]
    # list5 = [1, 4, 6, 8]
    # games = [0,0,0,0,0]
    # list5 = [1, 3, 8, 7]
    
    
    list1 = []
    # list2 = []
    # count = 0
    # choices = [list1, list2]
    
    score1 = 0
    score2 = 0
    
    # enemy_list = random.choice(choices)
    # enemy_list = [1, 2, 3, 4, 5, 6, 7, 8]
    for i in range(1, 9):    
        env.update_parameter('enemies', [i])
        _, player_life, enemy_life, time = env.play(pcont=np.array(individual))
        
        if time > 0:
            score = 0.9 * (100 - enemy_life) + 0.1 * player_life - np.log(time)
        else:
            score = 0.9 * (100 - enemy_life) + 0.1 * player_life - 20
        
        if enemy_life == 0:
            score1 += score
            list1.append(i)
        else:
            score2 += score
            # list2.append(i)
            
    if len(list1) == 8:
        return [2500]
    if len(list1) == 7:
        return [((score1)/8 + 1200)]

    final_score = ((6.5 * score1) + (3.5 * score2))/10
    
    return [final_score]



def fitness3(individual):
    # print(individual)
    # exit()
    env = Environment(experiment_name=experiment_name,
                      playermode="ai",
                      player_controller=player_controller(10),
                    #   multiplemode="yes",
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    games = []
    count = 0
    count2 = []
    
    games = []
    
    for i in range(1, 9):
        # if i == 1:
        #     continue
        env.update_parameter('enemies', [i])
        _, player_life, enemy_life, time = env.play(pcont=np.array(individual))
        score = 0.9 * (100 - enemy_life) + 0.1 * player_life - np.log(time)
        if enemy_life == 0:
            score += 20
        
        # if i == 6 or i == 7:
        #     score = score*2
            
        games.append(score)
        # if i not in list5:
        #     games[1] += score
        # else:
        #     count2.append(score)
        #     games[count] = score
        #     count += 1
            # if len(count2) == 3:
            #     count+=1
                
    # games[1] = games[1]/7
    # games[4] = games[4]/4
        
    # games.append(score)
    
    return [sum(games)/8]

# Set up the DEAP Genetic Algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def normalize(individual):
    return 2*(individual - np.min(individual))/(np.max(individual) - np.min(individual)) - 1


def create_env(n_hidden_neurons, enemy, experiment_name):
    return Environment(experiment_name=experiment_name,
                      enemies=[enemy],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

def run_optimization(save_best=False):
    # Initialize the EvoMan environment 
    env1 = create_env(n_hidden_neurons, 1, experiment_name)
    
    # Number of weights for multilayer with n_hidden_neurons
    global n_vars
    n_vars = (env1.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
    
    # print(n_vars)
    # exit()
    
    # Register individual and population creation operations in the toolbox
    toolbox.register("attr_float", np.random.uniform, -1, 1)  # Assuming weights are in the range [-1, 1]
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register other operations in the toolbox
    
    # min_vals = np.array([-1] * n_vars)
    # max_vals = np.array([1] * n_vars)
    
    toolbox.register("mate", tools.cxBlend, alpha=0.7)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.7)
    # toolbox.register("select", tools.selTournament, tournsize=20)
    
    # low = [-1] * n_vars
    # high = [1] * n_vars
    
    # test = np.random.uniform(-1, 1, size=(150, n_vars))
    
    # ref_points = np.array([low, high])
    # ref_point = tools.uniform_reference_points(nobj=1, p=265)
    
    
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", fitness)
    # Statistics setup
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Fitness avg", np.mean)
    stats.register("Fitness std", np.std)
    stats.register("Fitness min", np.min)
    stats.register("Fitness max", np.max)
    
    hof = tools.HallOfFame(100, similar=np.array_equal)
    # Here I choose the (μ + λ) evolutionary algorithm.
    population = toolbox.population(n=1000)
    # print(type(population))
    # exit()
    # print(population[0])
    wel_performed = creator.Individual(np.loadtxt(experiment_name + ' _solutions/best_solution50_hof2.txt'))
    wel_performed2 = creator.Individual(np.loadtxt(experiment_name + ' _solutions/beat6.txt'))
    wel_performed3 = creator.Individual(np.loadtxt(experiment_name + ' _solutions/best_solution60_hof2.txt'))
    wel_performed4 = creator.Individual(np.loadtxt(experiment_name + ' _solutions/best_solution61_hof2.txt'))
    wel_performed5 = creator.Individual(np.loadtxt(experiment_name + ' _solutions/best_solution66_hof2.txt'))
    wel_performed6 = creator.Individual(np.loadtxt(experiment_name + ' _solutions/best_solution68_hof2.txt'))
    wel_performed7 = creator.Individual(np.loadtxt(experiment_name + ' _solutions/best_solution69_hof2.txt'))
    wel_performed8 = creator.Individual(np.loadtxt(experiment_name + ' _solutions/best_solution70_hof2.txt'))
    wel_performed9 = creator.Individual(np.loadtxt(experiment_name + ' _solutions/best_solution71_hof2.txt'))
    wel_performed10 = creator.Individual(np.loadtxt(experiment_name + ' _solutions/best_solution72_hof2.txt'))
    wel_performed11 = creator.Individual(np.loadtxt(experiment_name + ' _solutions/best_solution73_hof2.txt'))
    wel_performed12 = creator.Individual(np.loadtxt(experiment_name + ' _solutions/best_solution75_hof2.txt'))
    wel_performed13 = creator.Individual(np.loadtxt(experiment_name + ' _solutions/best_solution76_hof2.txt'))
    # population[0] = wel_performed.tolist()
    # print(population[0
    # for _ in range(20):
    population.append(wel_performed2)
    population.append(wel_performed)
    population.append(wel_performed3) 
    population.append(wel_performed4) 
    population.append(wel_performed5) 
    population.append(wel_performed6)
    population.append(wel_performed7)
    population.append(wel_performed8)
    population.append(wel_performed9)
    population.append(wel_performed10)
    population.append(wel_performed11)
    population.append(wel_performed12)
    population.append(wel_performed13)
    # exit()
    
    # population = np.random.uniform(-1, 1, size=(5000, 265))
    # population = np.vstack([population, wel_performed])
    # population.append(wel_performed)
    
    # initial_pop = [creator.Individual([random.uniform(-1, 1) for _ in range(n_vars)]) for _ in range(population_size)]
    
    # final_pop, logbook1 = algorithms.eaMuCommaLambda(population, toolbox, mu=population_size, lambda_=200, 
    #                       cxpb=0.5, mutpb=mutation_rate, stats=stats, halloffame=hof, verbose=False, 
    #                       ngen=generations)
    
    # avg_fitnesses = [np.mean(ind.fitness.values) for ind in final_pop]
    # best_solution = final_pop[np.argmax(avg_fitnesses)]
    
    hof2 = tools.HallOfFame(1)
    
    toolbox.unregister("evaluate")
    toolbox.register("evaluate", fitness2)

    _, logbook2 = algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=150, 
                          cxpb=0.5, mutpb=mutation_rate, stats=stats, halloffame=hof2, verbose=False, 
                          ngen=generations)
    
    if save_best:
        if not os.path.exists(experiment_name + ' solutions'):
            os.makedirs(experiment_name + ' solutions')
        # np.savetxt(experiment_name + f' solutions/best_solution34.txt', normalize(best_solution))
        # np.savetxt(experiment_name + f' solutions/best_solution63_hof2.txt', normalize(hof2[0]))
        np.savetxt(experiment_name + f' _solutions/best_solution77_hof2.txt', hof2[0])
    # print(logbook)
    print(logbook2)
    return
    
    print(logbook1)
    print('--------------------------------')
    print(logbook2)
    return logbook1, logbook2


if __name__ == "__main__":
    
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    run_optimization(save_best=True)
    # pool.close()

