
# selection.py
from cutting_stock.fitness_functions import *

import random
import math
import copy

# For all the functions here, it's strongly recommended to
# review the documentation for Python's random module:
# https://docs.python.org/3/library/random.html

# Parent selection functions---------------------------------------------------
def uniform_random_selection(population, n, **kwargs):
    # TODO: select n individuals uniform randomly
    chosen = []
    for _ in range(n):
        chosen.append(random.choice(population))
    
    return chosen


def k_tournament_with_replacement(population, n, k, **kwargs):
    # TODO: perform n k-tournaments with replacement to select n individuals
    winners = []
    for _ in range(n):
        contestants = random.sample(population, k)
        
        # find the best individual (and its fitness) for k contestants
        best_individual = None
        max_fitness = -math.inf 
        
        for contestant in contestants:
            # print(contestant.fitness, max_fitness)
            if contestant.fitness > max_fitness:
                max_fitness = contestant.fitness
                best_individual = contestant
                
        winners.append(best_individual)
        
    return winners


def fitness_proportionate_selection(population, n, **kwargs):
    # TODO: select n individuals using fitness proportionate selection
    
    minimum_fitness = math.inf
    total_fitness = 0
    fitnesses = []
    evaluated = []
    
    # find the minimum fitness and total fitness of the population
    for individual in population:
        # individual.fitness = base_fitness_function(individual.genes, **kwargs)['fitness']
        fitnesses.append(individual.fitness)
        evaluated.append(individual)
        
        minimum_fitness = min(minimum_fitness, individual.fitness)
        total_fitness += individual.fitness 
    
    # handle cases where fitness is negative
    if minimum_fitness < 0:
        weights = [f - minimum_fitness for f in fitnesses]
        total_fitness -= minimum_fitness
    else:
        weights = fitnesses

    # calculate probabilities based off of weights (fitnesses)
    total_weight = sum(weights)
    if total_weight == 0:
        # if all fitnesses are 0, then select uniformly
        probabilities = [1 / len(weights) for _ in weights]
    else:
        probabilities = [w / total_weight for w in weights]
    
    cumulative_probs = []
    cumulative_sum = 0
    for p in probabilities:
        cumulative_sum += p
        cumulative_probs.append(cumulative_sum)

    selected = []
    for _ in range(n):
        combined = list(zip(evaluated, cumulative_probs))
        shuffled_evaluated, shuffled_cumulative_probs = zip(*combined)

        # chose random value to compare probability to
        r = random.random()
        
        for index, cumulative_prob in enumerate(shuffled_cumulative_probs):
            if r <= cumulative_prob:
                selected.append(shuffled_evaluated[index])
                break
        else:
            print(f"No individual selected for {r}")

    return selected


# Survival selection functions-------------------------------------------------
def truncation(population, n, **kwargs):
    # TODO: perform truncation selection to select n individuals

    # sort population based off fitness, then only return the top n individuals
    sorted_pop = sorted(population, key=lambda individual: individual.fitness, reverse=True)
    truncated = sorted_pop[0:n]

    return truncated



def k_tournament_without_replacement(population, n, k, **kwargs):
    # TODO: perform n k-tournaments without replacement to select n individuals
    # Note: an individual should never be cloned from surviving twice!
    # Also note: be careful if using list.remove(), list.pop(), etc.
    # since this can be EXTREMELY slow on large populations if not handled properly
    # A better alternative to my_list.pop(i) is the following:
    # my_list[i] = my_list[-1]
    winners = []
    selected = [False] * len(population)

    for _ in range(n):
        # Randomly sample k contestants, ensuring they are not already selected
        valid_contestants = [i for i, is_selected in enumerate(selected) if not is_selected]
        
        # Ensure there are enough individuals left for the tournament
        if len(valid_contestants) < k:
            contestants = valid_contestants
        else:
            contestants = random.sample(valid_contestants, k)
        
        # Find the best individual among the contestants
        best_individual = None
        max_fitness = -math.inf
        best_index = None
        
        for idx in contestants:
            contestant = population[idx]
            if contestant.fitness > max_fitness:  
                max_fitness = contestant.fitness
                best_individual = contestant
                best_index = idx
        
        if best_individual is not None:
            winners.append(best_individual)
            selected[best_index] = True  # Mark the individual as selected
    
    return winners




# Yellow deliverable parent selection function---------------------------------
def stochastic_universal_sampling(population, n, **kwargs):
    # Recall that yellow deliverables are required for students in the grad
    # section but bonus for those in the undergrad section.
    # TODO: select n individuals using stochastic universal sampling
    pass