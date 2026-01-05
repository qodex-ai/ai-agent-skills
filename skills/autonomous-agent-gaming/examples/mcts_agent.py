"""
Monte Carlo Tree Search (MCTS) Implementation

Probabilistic game tree exploration algorithm.
Effective for games with high branching factors where full exploration is infeasible.
"""

import math
import random
from dataclasses import dataclass
from typing import Optional, List, Any


@dataclass
class MCTSNode:
    """Node in the Monte Carlo Tree Search tree."""

    game_state: Any  # 'GameState' type
    parent: Optional['MCTSNode'] = None
    children: Optional[List['MCTSNode']] = None
    visits: int = 0
    value: float = 0.0

    def __post_init__(self):
        """Initialize children list if not provided."""
        if self.children is None:
            self.children = []


class MCTSAgent:
    """Game agent using Monte Carlo Tree Search algorithm."""

    def __init__(self, iterations=1000, exploration_constant=1.414):
        """
        Initialize MCTS agent.

        Args:
            iterations: Number of MCTS iterations to run
            exploration_constant: Balance exploration vs exploitation (UCT constant)
        """
        self.iterations = iterations
        self.c = exploration_constant  # Exploration constant for UCT formula

    def get_best_move(self, game_state):
        """
        Get the best move using Monte Carlo Tree Search.

        Runs multiple simulations of the game tree, balancing exploration
        of new moves with exploitation of known good moves.

        Args:
            game_state: Current game state

        Returns:
            Best move based on MCTS simulations
        """
        root = MCTSNode(game_state)

        # Run MCTS iterations
        for _ in range(self.iterations):
            # Selection and expansion
            node = self.tree_policy(root)
            # Simulation and evaluation
            reward = self.default_policy(node.game_state)
            # Backpropagation
            self.backup(node, reward)

        # Return move with highest visit count
        best_child = max(
            root.children,
            key=lambda child: child.visits
        )
        return self.get_move(root.game_state, best_child.game_state)

    def tree_policy(self, node):
        """
        Select and expand nodes in the tree.

        Follows UCT (Upper Confidence bounds applied to Trees) formula
        to balance exploration and exploitation.

        Args:
            node: Starting node for tree traversal

        Returns:
            New node to simulate from
        """
        while not node.game_state.is_terminal():
            if not self.fully_expanded(node):
                # Expand tree with new node
                return self.expand(node)
            else:
                # Select best child using UCT
                node = self.best_uct(node)
        return node

    def expand(self, node):
        """
        Add a new child node to the tree.

        Selects an unexplored move and creates a new node for it.

        Args:
            node: Parent node to expand from

        Returns:
            New child node
        """
        legal_moves = node.game_state.get_legal_moves()
        explored_moves = {
            self.get_move(node.game_state, child.game_state)
            for child in node.children
        }

        unexplored_move = random.choice(
            [m for m in legal_moves if m not in explored_moves]
        )

        child_state = node.game_state.apply_move(unexplored_move)
        child = MCTSNode(child_state, parent=node)
        node.children.append(child)
        return child

    def default_policy(self, game_state):
        """
        Simulate game with random moves (playout).

        Runs a fast simulation from the given state to a terminal state
        using random moves.

        Args:
            game_state: Starting state for simulation

        Returns:
            Game result/reward from simulation
        """
        state = game_state.copy()

        while not state.is_terminal():
            move = random.choice(state.get_legal_moves())
            state = state.apply_move(move)

        return state.get_result()

    def backup(self, node, reward):
        """
        Backpropagate simulation results up the tree.

        Updates visit counts and value sums for all nodes from the
        simulated node back to the root.

        Args:
            node: Node to backpropagate from
            reward: Reward value from simulation
        """
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def best_uct(self, node):
        """
        Select best child using Upper Confidence Bounds for Trees (UCT).

        Formula: UCT = exploitation + exploration
                     = (child_value / child_visits) + c * sqrt(ln(parent_visits) / child_visits)

        Args:
            node: Parent node

        Returns:
            Child node with highest UCT value
        """
        best_child = None
        best_uct = float('-inf')

        for child in node.children:
            uct = self.calculate_uct(child, node.visits)
            if uct > best_uct:
                best_uct = uct
                best_child = child

        return best_child

    def calculate_uct(self, node, parent_visits):
        """
        Calculate UCT value for a node.

        Balances exploitation (average reward) with exploration
        (uncertainty in estimates).

        Args:
            node: Child node to evaluate
            parent_visits: Number of visits to parent node

        Returns:
            UCT value for the node
        """
        exploitation = node.value / (node.visits + 1)
        exploration = self.c * math.sqrt(math.log(parent_visits) / (node.visits + 1))
        return exploitation + exploration

    def fully_expanded(self, node):
        """
        Check if all children of a node have been explored.

        Args:
            node: Node to check

        Returns:
            True if all legal moves have child nodes
        """
        return len(node.children) == len(node.game_state.get_legal_moves())

    def get_move(self, state1, state2):
        """
        Extract the move that transforms state1 into state2.

        Args:
            state1: Before state
            state2: After state

        Returns:
            The move made
        """
        pass
