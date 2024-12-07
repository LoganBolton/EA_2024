
# genetic_programming.py

import random
from copy import deepcopy
from base_evolution import BaseEvolutionPopulation

class GeneticProgrammingPopulation(BaseEvolutionPopulation):
    def generate_children(self):
        children = []
        recombined_child_count = 0
        mutated_child_count = 0

        while len(children) < self.num_children:
            # Mutate
            if random.random() < self.mutation_rate:
                parents = self.parent_selection(
                    self.population,
                    n=1,
                    **self.parent_selection_kwargs
                )
                parent = parents[0]
                mutant = deepcopy(parent)
                mutant.mutate()
                children.append(mutant)
                mutated_child_count += 1
            else: # Normal Recombination
                parents = self.parent_selection(
                    self.population,
                    n=2,
                    **self.parent_selection_kwargs
                )
                parent1, parent2 = parents
                child = deepcopy(parent1)
                child.recombine(parent2, **self.recombination_kwargs)
                children.append(child)
                recombined_child_count += 1
        # Log the generation details
        self.log.append(f'Number of children generated: {len(children)}')
        self.log.append(f'Number of recombined children: {recombined_child_count}')
        self.log.append(f'Number of mutated children: {mutated_child_count}')

        return children
