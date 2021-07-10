import json
import os
import copy
import random 
import numpy as np
from client import *
import matplotlib.pyplot as plt
import datetime
Team_Key = "VPN75t7UGO44XsrQlo9KuKkg8yRrqgSfyh13GITsQnyvLLyvV5" 
Team_Name = "Platypus_Perry"
max_limit = 10 
min_limit = -10
generations = 10
array_len = 11
population_size = 8
chromosome_size = 11
Mutation_Probability = 0.2         # probabilty that it will mutate
Mutation_Difference_Scale = 1000   # adding noice on mutation to actual value ratio
initial_array = [0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,  8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10]
valid_err_best_weights = 0
train_err_best_weights = 0
best_weights_set = []
fitness_best_weights = 0
population = []
best_valid_err_gen = []
best_train_err_gen = []
best_fit_gen = []
def add_noise(arr):
    for id, val in np.ndenumerate(arr):
        Mutation_Probability = 1    
#         Mutation_Probability = np.random.random()
        Mutation_Difference_Scale = np.random.randint(50, 70)
        chance = np.random.random()
        if chance < Mutation_Probability:
            range_var = val/Mutation_Difference_Scale
            if val == 0:
                range_var = np.mean(arr)/(Mutation_Difference_Scale*10)
            arr[id] = arr[id] + np.random.uniform(-range_var,0)
    return np.clip(arr, min_limit, max_limit)
def add_noise_populate(arr):
    for id, val in np.ndenumerate(arr):
        Mutation_Probability = 1      
#         Mutation_Probability = np.random.random()
        Mutation_Difference_Scale = np.random.randint(10, 15)
        chance = np.random.random()
        if chance < Mutation_Probability:
            range_var = val/Mutation_Difference_Scale
            if val == 0:
                range_var = np.mean(arr)/(Mutation_Difference_Scale*10)
            arr[id] = arr[id] + np.random.uniform(-range_var,0)
    return np.clip(arr, min_limit, max_limit)
def populate(arr):
    arra = []
    for i in range(0,population_size):
        arra.append(arr)
    arra = np.array(arra, dtype=np.double)
    arra = add_noise_populate(arra)
    arra[0] =  [0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,  8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10]
    return arra
def Roulette_Wheel_Selection(arr ,fitness_sum):
    random_nu = random.uniform(0,fitness_sum)
    for id, val in np.ndenumerate(arr):
        if (val > random_nu):
            return id
    return len(arr) - 1
def get_gen(name_file, i , population):
    j = i
    selected_population = []
    crossovered_population = []
    mutated_population = []
    # global population
    global fitness_best_weights
    global valid_err_best_weights
    global train_err_best_weights

    global best_fit_gen
    global best_train_err_gen
    global best_valid_err_gen
    #finding errors
    fitness = []
    train_error = []
    valid_error = []
    for chromosomes in population:
        train_err, valid_err = get_errors(Team_Key, list(chromosomes))
        train_error.append(train_err)
        valid_error.append(valid_err)
        fitness.append(-(train_err + valid_err))

    #updating err values
    best_fit_id = np.argmax(fitness)
    if ( (fitness_best_weights == 0) or (fitness[best_fit_id] > fitness_best_weights)):
        fitness_best_weights = fitness[best_fit_id]
        valid_err_best_weights = valid_error[best_fit_id]
        train_err_best_weights = train_error[best_fit_id]
        best_weights_set[:] = population[best_fit_id]
    
    best_fit_gen.append((-1)*fitness[best_fit_id])
    best_valid_err_gen.append(valid_error[best_fit_id])
    best_train_err_gen.append(train_error[best_fit_id])

    scaled_fitness = (fitness - np.min(fitness)) / np.ptp(fitness) #fitness is scaled from 0 to 1
    n = int(population_size - (population_size*2)/3)
    indic =  scaled_fitness.argsort()[:n]
    boolarray = np.zeros((population_size,),dtype=int)
    for value in indic:
        boolarray[value] = 1
    partial_sum = []
    sum_fitness = np.sum(scaled_fitness)
    part = 0.0
    z =0
    for val in scaled_fitness:
        if boolarray[z]==1:
            val = 0.0
        part = part + val
        z = z+1
        partial_sum.append(part)
    crossover =[]
    for i in range(0, population_size//2):
        # selection
        index1 = Roulette_Wheel_Selection(partial_sum ,part)
        index2 = Roulette_Wheel_Selection(partial_sum ,part)
        selected_f = population[index1]
        selected_s = population[index2]
        selected_population.append(selected_f)
        selected_population.append(selected_s)

        # crossover 
        crossovered_f = np.empty(11)        
        crossovered_s = np.empty(11)
        u = random.random()
        nc = 3
        if (u < 0.5):
            beta = (2 * u)**((nc + 1)**-1)
        else:
            beta = ((2*(1-u))**-1)**((nc + 1)**-1)
        parent1 = np.array(selected_f)
        parent2 = np.array(selected_s)
        crossovered_f = 0.5*((1 + beta) * parent1 + (1 - beta) * parent2)
        crossovered_s = 0.5*((1 - beta) * parent1 + (1 + beta) * parent2)
        crossovere_f = np.copy(crossovered_f)
        crossovere_s = np.copy(crossovered_s)
        crossover_point = np.random.randint(4,9)
        crossovered_f[0:crossover_point] = crossovere_s[0:crossover_point]
        crossovered_s[0:crossover_point] = crossovere_f[0:crossover_point]
        crossovered_population.append(crossovered_f)
        crossovered_population.append(crossovered_s)
        crossover.append(crossovered_f)
        crossover.append(crossovered_s)
        
    
    #mutation
    crossover = np.array(crossover, dtype=np.double)
    population = add_noise(crossover)
    mutated_population = population

    #creating generation files
    # file_name = str(name_file) + "/" + "generations_" + str(j+1) + ".txt"
    
    # selected_population = np.array(selected_population)
    # crossovered_population = np.array(crossovered_population)
    # mutated_population = np.array(mutated_population)
    # with open(file_name, 'w') as write_file:
    #     json.dump(selected_population.tolist(), write_file)
    #     write_file.write('\n' + '\n')
    #     json.dump(crossovered_population.tolist(), write_file)
    #     write_file.write('\n' + '\n')
    #     json.dump(mutated_population.tolist(), write_file)
    return population

name_file = "generations1"

# if (os.path.isdir(name_file) != 1):
#     os.mkdir(name_file)
population = populate(initial_array)
population = np.array(population , dtype=np.double)

for i in range(0,generations):
    population = get_gen(name_file, i , population)
    
print(best_fit_gen)
print(fitness_best_weights)