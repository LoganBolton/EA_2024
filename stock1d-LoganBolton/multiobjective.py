from math import inf
import math
import numpy as np


# TODO: Return True if A dominates B based on the objective member variables of both objects.
#       If attempting the YELLOW deliverable, your code must be able to gracefully handle
#       any number of objectives, i.e., don't hardcode an assumption that there are 2 objectives.
def dominates(A, B):
    # HINT: We strongly recommend use of the built-in functions any() and all()
    all_geq = all(a >= b for a, b in zip(A.objectives, B.objectives))
    not_equal = any(a > b for a, b in zip(A.objectives, B.objectives))
    
    return all_geq and not_equal


# TODO: Use the dominates function (above) to sort the input population into levels
#       of non-domination, and assign to the level members based on an individual's level.
def nondomination_sort(population):
    for individual in population:
        individual.dominated_set = []
        individual.domination_count = 0  

    cache = {}

    # Populate domination relationships
    for i, A in enumerate(population):
        for j, B in enumerate(population):
            if i == j:
                continue

            pair = (i, j)
            if pair in cache:
                A_dominates_B = cache[pair]
            else:
                A_dominates_B = dominates(A, B)
                cache[pair] = A_dominates_B

            if A_dominates_B:
                A.dominated_set.append(B)
            else:
                reverse_pair = (j, i)
                if reverse_pair in cache:
                    B_dominates_A = cache[reverse_pair]
                else:
                    B_dominates_A = dominates(B, A)
                    cache[reverse_pair] = B_dominates_A

                if B_dominates_A:
                    A.domination_count += 1  

    # Init the first Pareto front
    fronts = [[]]
    for individual in population:
        if individual.domination_count == 0:
            individual.level = 1
            fronts[0].append(individual)

    current_front = 0
    while fronts[current_front]:
        next_front = []
        for individual in fronts[current_front]:
            for dominated_individual in individual.dominated_set:
                dominated_individual.domination_count -= 1
                if dominated_individual.domination_count == 0:
                    dominated_individual.level = current_front + 2 
                    next_front.append(dominated_individual)
        current_front += 1
        fronts.append(next_front)


# TODO: Calculate the crowding distance from https://ieeexplore.ieee.org/document/996017
#       For each individual in the population, and assign this value to the crowding member variable.
#       Use the inf constant (imported at the top of this file) to represent infinity where appropriate.
# IMPORTANT: Note that crowding should be calculated for each level of nondomination independently.
#            That is, only individuals within the same level should be compared against each other for crowding.
from math import inf

def assign_crowding_distances(population):
    # Assign individuals to their level
    levels = {}
    for individual in population:
        level = individual.level
        if level not in levels:
            levels[level] = []
        levels[level].append(individual)
    
    num_objectives = len(population[0].objectives)

    # Determine crowding distance by level
    for individuals in levels.values():
        # Init crowding for all individuals
        for individual in individuals:
            individual.crowding = 0.0
        
        for m in range(num_objectives):
            # Sort individuals based on the current objective
            sorted_individuals = sorted(individuals, key=lambda individual: individual.objectives[m])
            
            # Assign boundary individuals
            sorted_individuals[0].crowding = inf
            sorted_individuals[-1].crowding = inf
            
            # Find the range
            min_obj = sorted_individuals[0].objectives[m]
            max_obj = sorted_individuals[-1].objectives[m]
            range_obj = max_obj - min_obj
            
            if range_obj == 0:
                range_obj = inf
            
            # Calculate normalized distance
            for i in range(1, len(sorted_individuals) - 1):
                first_objective = sorted_individuals[i - 1].objectives[m]
                second_objective = sorted_individuals[i + 1].objectives[m]
                
                if range_obj == inf:
                    distance = inf
                else:
                    distance = (second_objective - first_objective) / range_obj
                
                # If either neighbor has infinity, assign infinity
                if sorted_individuals[i].crowding == inf or distance == inf:
                    sorted_individuals[i].crowding = inf
                else:
                    sorted_individuals[i].crowding += distance


# This function is implemented for you. You should not modify it.
# It uses the above functions to assign fitnesses to the population.
def assign_fitnesses(population, crowding, failure_fitness, **kwargs):
    # Assign levels of nondomination.
    nondomination_sort(population)

    # Assign fitnesses.
    max_level = max(map(lambda x:x.level, population))
    for individual in population:
        individual.fitness = max_level + 1 - individual.level

    # Check if we should apply crowding penalties.
    if not crowding:
        for individual in population:
            individual.crowding = 0

    # Apply crowding penalties.
    else:
        assign_crowding_distances(population)
        for individual in population:
            if individual.crowding != inf:
                assert 0 <= individual.crowding <= len(individual.objectives),\
                    f'A crowding distance ({individual.crowding}) was not in the correct range. ' +\
                    'Make sure you are calculating them correctly in assign_crowding_distances.'
                individual.fitness -= 1 - 0.999 * (individual.crowding / len(individual.objectives))




# The remainder of this file is code used to calculate hypervolumes.
# You do not need to read, modify or understand anything below this point.
# Implementation based on https://ieeexplore.ieee.org/document/5766730


def calculate_hypervolume(front, reference_point=None):
    point_set = [individual.objectives for individual in front]
    if reference_point is None:
        # Defaults to (-1)^n, which assumes the minimal possible scores are 0.
        reference_point = [-1] * len(point_set[0])
    return wfg_hypervolume(list(point_set), reference_point, True)


def wfg_hypervolume(pl, reference_point, preprocess=False):
    if preprocess:
        pl_set = {tuple(p) for p in pl}
        pl = list(pl_set)
        if len(pl[0]) >= 4:
            pl.sort(key=lambda x: x[0])

    if len(pl) == 0:
        return 0
    return sum([wfg_exclusive_hypervolume(pl, k, reference_point) for k in range(len(pl))])


def wfg_exclusive_hypervolume(pl, k, reference_point):
    return wfg_inclusive_hypervolume(pl[k], reference_point) - wfg_hypervolume(limit_set(pl, k), reference_point)


def wfg_inclusive_hypervolume(p, reference_point):
    return math.prod([abs(p[j] - reference_point[j]) for j in range(len(p))])


def limit_set(pl, k):
    ql = []
    for i in range(1, len(pl) - k):
        ql.append([min(pl[k][j], pl[k+i][j]) for j in range(len(pl[0]))])
    result = set()
    for i in range(len(ql)):
        interior = False
        for j in range(len(ql)):
            if i != j:
                if all(ql[j][d] >= ql[i][d] for d in range(len(ql[i]))) and any(ql[j][d] > ql[i][d] for d in range(len(ql[i]))):
                    interior = True
                    break
        if not interior:
            result.add(tuple(ql[i]))
    return list(result)
