"""
Q-Learning Agent Implementation

Reinforcement learning agent that learns optimal policies through
interaction with the game environment using the Q-learning algorithm.
"""

import numpy as np
import random
from collections import defaultdict
from typing import Optional, List, Any


class QLearningAgent:
    """Reinforcement learning agent using Q-learning algorithm."""

    def __init__(self, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        """
        Initialize the Q-learning agent.

        Args:
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Exploration probability for epsilon-greedy action selection
        """
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))

    def get_action(self, state):
        """
        Select an action using epsilon-greedy strategy.

        With probability epsilon, explores a random action.
        With probability 1-epsilon, exploits the best known action.

        Args:
            state: Current game state

        Returns:
            Selected action
        """
        if np.random.random() < self.epsilon:
            # Explore: select random action
            return self.random_action(state)
        else:
            # Exploit: select best known action
            return self.best_action(state)

    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using Q-learning update rule.

        Q(s,a) := Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
        """
        # Get best action for next state
        best_next_action = self.best_action(next_state)

        # Calculate TD target
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]

        # Calculate TD error
        td_error = td_target - self.q_table[state][action]

        # Update Q-value
        self.q_table[state][action] += self.alpha * td_error

    def best_action(self, state):
        """
        Get the action with highest Q-value for a state.

        Args:
            state: Current game state

        Returns:
            Best action, or None if no actions available
        """
        actions = self.get_legal_actions(state)
        if not actions:
            return None

        q_values = {a: self.q_table[state][a] for a in actions}
        return max(q_values, key=q_values.get)

    def random_action(self, state):
        """
        Select a random legal action.

        Args:
            state: Current game state

        Returns:
            Random legal action
        """
        return random.choice(self.get_legal_actions(state))

    def get_legal_actions(self, state):
        """
        Get available actions in the current state.

        Args:
            state: Current game state

        Returns:
            List of legal actions
        """
        pass

    def get_q_value(self, state, action):
        """
        Get Q-value for a state-action pair.

        Args:
            state: Game state
            action: Action to evaluate

        Returns:
            Stored Q-value
        """
        return self.q_table[state][action]

    def get_q_values(self, state):
        """
        Get all Q-values for a state.

        Args:
            state: Game state

        Returns:
            Dictionary of actions to Q-values
        """
        actions = self.get_legal_actions(state)
        return {a: self.q_table[state][a] for a in actions}

    def decay_epsilon(self, decay_rate=0.995):
        """
        Decay exploration probability over time.

        Gradually shift from exploration to exploitation as learning progresses.

        Args:
            decay_rate: Multiplicative decay rate per update
        """
        self.epsilon *= decay_rate
        self.epsilon = max(self.epsilon, 0.01)  # Maintain some exploration

    def save_q_table(self, filename):
        """
        Save Q-table to file for later use.

        Args:
            filename: Path to save file
        """
        import json
        # Convert defaultdict to regular dict for JSON serialization
        q_dict = {str(state): dict(actions) for state, actions in self.q_table.items()}
        with open(filename, 'w') as f:
            json.dump(q_dict, f)

    def load_q_table(self, filename):
        """
        Load Q-table from file.

        Args:
            filename: Path to load file
        """
        import json
        with open(filename, 'r') as f:
            q_dict = json.load(f)
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state, actions in q_dict.items():
            for action, value in actions.items():
                self.q_table[state][action] = value

    def reset(self):
        """Clear all learned Q-values."""
        self.q_table.clear()
