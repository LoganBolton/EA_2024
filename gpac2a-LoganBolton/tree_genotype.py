
# tree_genotype.py

import random
from copy import deepcopy
from fitness import manhattan
import math

class TreeGenotype():
    def __init__(self):
        self.fitness = None
        self.genes = ParseTree()


    @classmethod
    def initialization(cls, mu, **kwargs):
        population = [cls() for _ in range(mu)]

        # 2a TODO: Initialize genes member variables of individuals
        #          in population using ramped half-and-half.
        #          Pass **kwargs to your functions to give them
        #          the sets of terminal and nonterminal primitives.
        terminals = kwargs['terminals']
        nonterminals = kwargs['nonterminals']
        depth_limit = kwargs['depth_limit']
        for i in range(mu):
            if i % 2 != 0:
                # Create grow tree
                population[i].genes.root = ParseTree.create_grow(
                    depth_limit, 
                    terminals, 
                    nonterminals
                )
            else:
                # Create full tree
                population[i].genes.root = ParseTree.create_full(
                    depth_limit, 
                    terminals, 
                    nonterminals
                )
        
        return population


    def serialize(self):
        # 2a TODO: Return a string representing self.genes in the required format.
        def dfs(node, depth):
            if node is None:
                return ''
            
            result = '|' * depth
            if node.type == 'C':
                result += str(node.value)
            else:
                result += str(node.type)
            result += '\n'
            
            # Add children's representations
            result += dfs(node.left, depth + 1)
            result += dfs(node.right, depth + 1)
            
            return result
        
        return dfs(self.genes.root, 0)

    def deserialize(self, serialization):
        lines = serialization.strip().split('\n')
        self.genes = ParseTree()
        
        root_primitive = lines[0].strip('|')
        root = TreeNode(root_primitive)
        if root.type == 'C':
            root.value = int(root_primitive)
            root.type = 'C'
        self.genes.root = root
        
        parent_stack = [(root, 0)]
        for line in lines[1:]:
            current_depth = line.count('|')
            primitive = line.strip('|')
            
            new_node = TreeNode(primitive)
            if new_node.type == 'C':
                new_node.value = int(primitive)
                new_node.type = 'C'
                
            while parent_stack and parent_stack[-1][1] >= current_depth:
                parent_stack.pop()
                
            if parent_stack:
                parent, parent_depth = parent_stack[-1]
                if not parent.left:
                    parent.left = new_node
                else:
                    parent.right = new_node
                    
            parent_stack.append((new_node, current_depth))

    def recombine(self, mate, depth_limit, **kwargs):
        child = self.__class__()

        # 2b TODO: Recombine genes of mate and genes of self to
        #          populate child's genes member variable.
        #          We recommend using deepcopy, but also recommend
        #          that you deepcopy the minimal amount possible.

        return child


    def mutate(self, depth_limit, **kwargs):
        mutant = self.__class__()
        mutant.genes = deepcopy(self.genes)

        # 2b TODO: Mutate mutant.genes to produce a modified tree.

        return mutant
    def evaluate_state(self, state):
        return self.genes.evaluate(state)
    
class ParseTree:
    def __init__(self):
        self.root = None
    
    @staticmethod
    def create_full(depth_limit, terminals, nonterminals, current_depth=0):
        if current_depth == depth_limit:
            # At max depth, must use terminal
            primitive = random.choice(terminals)
            node = TreeNode(primitive)
            return node
        else:
            # Not at max depth, use nonterminal
            primitive = random.choice(nonterminals)
            node = TreeNode(primitive)
            node.left = ParseTree.create_full(depth_limit, terminals, nonterminals, current_depth + 1)
            node.right = ParseTree.create_full(depth_limit, terminals, nonterminals, current_depth + 1)
            return node
        
    @staticmethod
    def create_grow(depth_limit, terminals, nonterminals, current_depth=0):
        if current_depth == depth_limit:
            # At max depth, use terminal
            primitive = random.choice(terminals)
            node = TreeNode(primitive)
            return node
        else:
            # Not at max depth, can use either terminal or nonterminal
            # Combine both sets and choose randomly
            all_primitives = terminals + nonterminals
            primitive = random.choice(all_primitives)
            
            node = TreeNode(primitive)
            
            # If we choose a nonterminal, need to create children
            if primitive in nonterminals:
                node.left = ParseTree.create_grow(depth_limit, terminals, nonterminals, current_depth + 1)
                node.right = ParseTree.create_grow(depth_limit, terminals, nonterminals, current_depth + 1)
            
            return node
    
    def evaluate(self, state):
        if self.root is None:
            raise ValueError("Cannot evaluate empty tree")
        return self.root.evaluate(state)
    
class TreeNode:
    def __init__(self, primitive_type):
        self.left = None
        self.right = None
        
        self.type = primitive_type
        self.value = None
        if self.type == 'C':
            self.value = random.uniform(-8, 8)

    def evaluate(self, state, cache=None):
        try:
            if cache is None:
                cache = {}
                
            if self.type == 'C':
                return float(self.value)
                
            # For terminals, check cache first
            if self.type in ['G', 'P', 'F', 'W']:
                if self.type in cache:
                    return cache[self.type]
                    
            if self.type == 'G':  # Ghost distance
                min_dist = float('inf')
                pac_pos = state['players']['m']

                for player in state['players']:
                    if 'm' not in player:  # is a ghost
                        pos = state['players'][player]
                        dist = manhattan(pac_pos, pos)
                        min_dist = min(min_dist, dist)
                result = float(min_dist-15)

                if self.type not in cache:
                    cache[self.type] = result
                return result
                
            elif self.type == 'P':  # Pill distance
                min_dist = float('inf')
                pac_pos = state['players']['m']

                for pill_pos in state['pills']:
                    dist = manhattan(pac_pos, pill_pos)
                    min_dist = min(min_dist, dist)
                
                result = 5.0 / (float(min_dist) + 1)  # Adding 1 to avoid division by zero
                if self.type not in cache:
                    cache[self.type] = result

                return result
                
            elif self.type == 'F':  # Fruit distance
                # large distance from fruit means bad
                # want the lowest score possible 
                if state['fruit'] is None:
                    return 0
                pac_pos = state['players']['m']
                dist = manhattan(pac_pos, state['fruit'])
                
                result = 250.0 / (float(dist) + 1)  # Adding 1 to avoid division by zero
                if self.type not in cache:
                    cache[self.type] = result
                    
                return result
                
            elif self.type == 'W':  # Number of Adjacent Walls
                count = 0
                pac_x, pac_y = state['players']['m']
                # left, right, up, down
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

                for dx, dy in directions:
                    adj_x, adj_y = pac_x + dx, pac_y + dy
                    if 0 <= adj_y < len(state['walls']) and 0 <= adj_x < len(state['walls'][adj_y]):
                        # next to a wall
                        if state['walls'][adj_y][adj_x]:
                            count += 1

                result = -float(count) 
                if self.type not in cache:
                    cache[self.type] = result
                # result = -(float(count) ** 5)
                # print(f"walls: {count}")
                return result
                
            # Evaluate non-terminals
            left_val = self.left.evaluate(state, cache)
            right_val = self.right.evaluate(state, cache)
            
            if self.type == '+':
                return left_val + right_val
            elif self.type == '-':
                return left_val - right_val
            elif self.type == '*':
                return left_val * right_val
            elif self.type == '/':
                # Protected division to avoid divide by zero
                if right_val == 0:
                    return left_val
                return left_val / right_val
            elif self.type == 'RAND':
                return random.uniform(left_val, right_val)
            
            raise ValueError(f"Unknown node type: {self.type}")
        except Exception as e:
            print(f"Evaluation Error")
            return 0.0