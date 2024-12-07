
# ghost_genotype.py

import random
from copy import deepcopy
from fitness import manhattan

class GhostTreeGenotype():
    def __init__(self):
        self.fitness = None
        self.genes = GhostParseTree()

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
        # print(mu)
        pattern = ["Full", "Grow", "Grow", "Full"]
        pattern_length = len(pattern)
        for i in range(mu):
            tree_type = pattern[i % pattern_length]
            if tree_type == "Grow":
                # Create grow tree
                population[i].genes.root = GhostParseTree.create_grow(
                    depth_limit, 
                    terminals, 
                    nonterminals
                )
            elif tree_type == "Full":
                # Create full tree
                # choose a random depth between 1 and depth_limit
                tree_depth = random.randint(1, depth_limit)
                population[i].genes.root = GhostParseTree.create_full(
                    tree_depth, 
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
        self.genes = GhostParseTree()
        
        root_primitive = lines[0].strip('|')
        root = GhostTreeNode(root_primitive)
        if root.type == 'C':
            root.value = int(root_primitive)
            root.type = 'C'
        self.genes.root = root
        
        parent_stack = [(root, 0)]
        for line in lines[1:]:
            current_depth = line.count('|')
            primitive = line.strip('|')
            
            new_node = GhostTreeNode(primitive)
            if new_node.type == 'C':
                # Handle constant values
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

    def is_tree_valid(self, node, depth_limit, current_depth=0):
        if node is None:
            return True
        
        if current_depth > depth_limit:
            return False
        
        nonterminals = {'+', '-', '*', '/', 'RAND'}
        
        if node.type in nonterminals:
            if not node.left or not node.right:
                return False
            return (self.is_tree_valid(node.left, depth_limit, current_depth + 1) and
                    self.is_tree_valid(node.right, depth_limit, current_depth + 1))
        else:
            # Terminal nodes should not have children
            return node.left is None and node.right is None
            
    def recombine(self, mate, depth_limit, **kwargs):
        max_attempts = 100 
        attempts = 0
        
        terminals = {'G', 'P', 'F', 'W', 'C', 'M'}
        nonterminals = {'+', '-', '*', '/', 'RAND'}

        child = GhostTreeGenotype()
        child.genes = deepcopy(self.genes)

        self_terminal_nodes = child.genes.root.get_terminal_nodes()
        self_nonterminal_nodes = child.genes.root.get_nonterminal_nodes()

        mate_terminal_nodes = mate.genes.root.get_terminal_nodes()
        mate_nonterminal_nodes = mate.genes.root.get_nonterminal_nodes()
        
        while attempts < max_attempts:

            isTerminal = random.choice([True, False])

            if len(self_nonterminal_nodes) == 0 or len(mate_nonterminal_nodes) == 0 or isTerminal:
                if not self_terminal_nodes or not mate_terminal_nodes:
                    attempts += 1
                    continue  # Cannot perform crossover with terminals, try again
                crossover_point_self = random.choice(self_terminal_nodes)
                crossover_point_mate = random.choice(mate_terminal_nodes)
            else:
                crossover_point_self = random.choice(self_nonterminal_nodes)
                crossover_point_mate = random.choice(mate_nonterminal_nodes)
            
            # Deepcopy the subtree from mate
            replacement_subtree = deepcopy(crossover_point_mate)
            replacement_depth = replacement_subtree.get_depth()

            # Find the depth to the crossover point in self tree
            depth_to_crossover = child.genes.root.get_depth_to_node(crossover_point_self)

            if depth_to_crossover is None:
                attempts += 1
                continue 

            # Allowed depth for the replacement subtree
            allowed_subtree_depth = depth_limit - depth_to_crossover

            # If the replacement subtree is too deep, trim it
            if replacement_depth > allowed_subtree_depth:
                replacement_subtree = replacement_subtree.trim(allowed_subtree_depth)
                if replacement_subtree is None:
                    attempts += 1
                    if crossover_point_self.type in nonterminals and crossover_point_mate.type in nonterminals:
                        self_nonterminal_nodes.remove(crossover_point_self)
                        self_nonterminal_nodes.append(crossover_point_mate)
                    if crossover_point_self.type in terminals and crossover_point_mate.type in terminals:
                        self_terminal_nodes.remove(crossover_point_self)
                        self_terminal_nodes.append(crossover_point_mate)
                    continue  # Replacement resulted in no subtree, try again

            # Perform the subtree replacement
            child.genes.root = child.genes.root.replace_subtree(crossover_point_self, replacement_subtree)
            
            # Check if the tree is valid
            if self.is_tree_valid(child.genes.root, depth_limit):
                return child
            
            attempts += 1

        # If no valid crossover was found after max_attempts, throw error
        raise ValueError(f"No valid crossover found - Self:\n\
            {self.serialize()}\n    \
            {mate.serialize()}\n\
            {isTerminal}\n\
            {crossover_point_mate.type}\n\
            {crossover_point_self.type}\n\
            {replacement_depth}\n\
            {allowed_subtree_depth}\n\
            {replacement_subtree}\n")

    def mutate(self, **kwargs):
        mutant = self.__class__()
        mutant.genes = deepcopy(self.genes)
        terminals = {'G', 'P', 'F', 'W', 'C', 'M'}
        nonterminals = {'+', '-', '*', '/', 'RAND'}
        
        # Mutate ~20% of the nodes
        total_nodes = mutant.genes.root.get_all_nodes()
        num_mutations = max(1, len(total_nodes) // 5)
        
        terminal_nodes = mutant.genes.root.get_terminal_nodes()

        for _ in range(num_mutations):
            node = random.choice(total_nodes)
            # change to a different type of the same node
            if node in terminal_nodes:
                possible_terminals = [t for t in terminals if t != node.type]
                if possible_terminals:
                    node.type = random.choice(possible_terminals)
                    if node.type == 'C':
                        new_value = random.uniform(-8, 8)
                        node.value = new_value
            else:
                possible_nonterminals = [n for n in nonterminals if n != node.type]
                if possible_nonterminals:
                    node.type = random.choice(possible_nonterminals)
                    
        return mutant

    def evaluate_state(self, state, current_player):
        return self.genes.evaluate(state, current_player)
    
class GhostParseTree:
    def __init__(self):
        self.root = None
    
    @staticmethod
    def create_full(depth_limit, terminals, nonterminals, current_depth=0):
        if current_depth == depth_limit:
            # At max depth, must use terminal
            primitive = random.choice(terminals)

            node = GhostTreeNode(primitive)
            return node
        else:
            # Not at max depth, use nonterminal
            primitive = random.choice(nonterminals)
            node = GhostTreeNode(primitive)
            node.left = GhostParseTree.create_full(depth_limit, terminals, nonterminals, current_depth + 1)
            node.right = GhostParseTree.create_full(depth_limit, terminals, nonterminals, current_depth + 1)
            return node
        
    @staticmethod
    def create_grow(depth_limit, terminals, nonterminals, current_depth=0):
        if current_depth == depth_limit:
            # At max depth, must use terminal
            primitive = random.choice(terminals)
            node = GhostTreeNode(primitive)
            return node
        else:
            # Not at max depth, can use either terminal or nonterminal
            # Combine both sets and choose randomly
            all_primitives = terminals + nonterminals
            primitive = random.choice(all_primitives)
            
            node = GhostTreeNode(primitive)
            
            # If we chose a nonterminal, need to create children
            if primitive in nonterminals:
                node.left = GhostParseTree.create_grow(depth_limit, terminals, nonterminals, current_depth + 1)
                node.right = GhostParseTree.create_grow(depth_limit, terminals, nonterminals, current_depth + 1)
            
            return node
    
    def evaluate(self, state, current_player):
        if self.root is None:
            raise ValueError("Cannot evaluate empty tree")
        return self.root.evaluate(state, current_player)
    
class GhostTreeNode:
    def __init__(self, primitive_type):
        self.left = None
        self.right = None
        
        self.type = primitive_type
        self.value = None
        if self.type == 'C':
            self.value = random.uniform(-2, 2)
    
    def get_all_nodes(self):
        nodes = [self]
        if self.left:
            nodes.extend(self.left.get_all_nodes())
        if self.right:
            nodes.extend(self.right.get_all_nodes())
        return nodes
            
    def evaluate(self, state, current_player, cache=None):
        try:
            if cache is None:
                cache = {}
                
            if self.type == 'C':
                return float(self.value)
                
            # For terminals, check cache first
            if self.type in ['G', 'P', 'F', 'W', 'M']:
                if self.type in cache:
                    return cache[self.type]
                    
            # Evaluate terminals
            if self.type == 'G':  # Ghost distance
                min_dist = float('inf')
                current_pos = state['players'][current_player]
                for player in state['players']:
                    if 'm' in player or current_player is player:
                        continue
                
                    pos = state['players'][player]
                    dist = manhattan(current_pos, pos)
                    min_dist = min(min_dist, dist)
                result = float(min_dist/5)
                if self.type not in cache:
                    cache[self.type] = result
                return result
                
            elif self.type == 'P':  # Pill distance
                min_dist = float('inf')
                pac_pos = state['players']['m']
                for pill_pos in state['pills']:
                    dist = manhattan(pac_pos, pill_pos)
                    min_dist = min(min_dist, dist)
                
                # seems like inverse square works
                result = 5.0 / (float(min_dist) + 1)  # Adding 1 to avoid division by zero

                if self.type not in cache:
                    cache[self.type] = result
                # print(f"p dist: {min_dist}")
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

                pac_x, pac_y = state['players'][current_player]
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
                return result

            elif self.type == 'M': # distance to pacman
                pac_pos = state['players']['m']
                ghost_pos = state['players'][current_player]
                dist = manhattan(pac_pos, ghost_pos)
                result = dist
                if self.type not in cache:
                    cache[self.type] = result
                return result
                
                
            # Evaluate non-terminals
            left_val = self.left.evaluate(state, current_player, cache)
            right_val = self.right.evaluate(state, current_player, cache)
            
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
            print(f"Error evaluating node: {e}")
    def get_all_nodes(self):
        nodes = [self]
        if self.left:
            nodes.extend(self.left.get_all_nodes())
        if self.right:
            nodes.extend(self.right.get_all_nodes())
        return nodes
    
    def replace_subtree(self, target, replacement):
        if self is target:
            return deepcopy(replacement) if replacement else None

        new_node = GhostTreeNode(self.type)
        new_node.value = self.value

        if self.left:
            new_node.left = self.left.replace_subtree(target, replacement)
        else:
            new_node.left = None

        if self.right:
            new_node.right = self.right.replace_subtree(target, replacement)
        else:
            new_node.right = None

        return new_node

    def get_depth(self):
        if self is None:
            return 0
        left_depth = self.left.get_depth() if self.left else 0
        right_depth = self.right.get_depth() if self.right else 0
        return 1 + max(left_depth, right_depth)
    
    # Used to find out how deep a node is in the tree
    def get_depth_to_node(self, target, current_depth=0):
        if self is target:
            return current_depth
        depth = None
        if self.left:
            depth = self.left.get_depth_to_node(target, current_depth + 1)
            if depth is not None:
                return depth
        if self.right:
            depth = self.right.get_depth_to_node(target, current_depth + 1)
            if depth is not None:
                return depth
        return None
    
    def trim(self, max_depth, current_depth=0):
        if current_depth > max_depth:
            return None  # Cut branch

        if self.type in {"+", "-", "*", "/", "RAND"}:
            if current_depth == max_depth:
                # At max_depth, operator cannot have children, so remove this node
                return None
            else:
                trimmed_left = self.left.trim(max_depth, current_depth + 1) if self.left else None
                trimmed_right = self.right.trim(max_depth, current_depth + 1) if self.right else None

                if trimmed_left and trimmed_right:
                    new_node = GhostTreeNode(self.type)
                    new_node.value = self.value
                    new_node.left = trimmed_left
                    new_node.right = trimmed_right
                    return new_node
                else:
                    return None
        else:
            if current_depth <= max_depth:
                # Keep terminal node as leaf
                new_node = GhostTreeNode(self.type)
                new_node.value = self.value
                return new_node
            else:
                # Prune this branch if depth exceeded
                return None

    def get_terminal_nodes(self):
        terminals = {'G', 'P', 'F', 'W', 'C', 'M'}
        nodes = []
        if self.type in terminals:
            nodes.append(self)
        if self.left:
            nodes.extend(self.left.get_terminal_nodes())
        if self.right:
            nodes.extend(self.right.get_terminal_nodes())
        return nodes
    
    def get_nonterminal_nodes(self):
        nonterminals = {'+', '-', '*', '/', 'RAND'}
        nodes = []

        if self.type in nonterminals:
            # print(self.type, end = ' ')
            nodes.append(self)
        if self.left:
            nodes.extend(self.left.get_nonterminal_nodes())
        if self.right:
            nodes.extend(self.right.get_nonterminal_nodes())
        return nodes