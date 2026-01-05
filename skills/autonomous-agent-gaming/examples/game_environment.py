"""
Custom Game Environment Implementation

Base classes and utilities for building custom game environments
compatible with game agents.
"""

from enum import Enum
from typing import Any, Tuple, Optional
from abc import ABC, abstractmethod


class GameState(Enum):
    """Enumeration of possible game states."""

    PLAYING = 1
    AGENT_THINKING = 2
    GAME_OVER = 3


class GameEnvironment(ABC):
    """
    Abstract base class for game environments.

    Provides interface for agents to interact with games.
    """

    def __init__(self, width=800, height=600):
        """
        Initialize the game environment.

        Args:
            width: Display width (if rendering)
            height: Display height (if rendering)
        """
        self.width = width
        self.height = height
        self.state = self.reset()

    def reset(self) -> Any:
        """
        Initialize or reset the game.

        Returns:
            Initial game state
        """
        return self.get_initial_state()

    def step(self, action) -> Tuple[Any, float, bool]:
        """
        Execute one action in the environment.

        Args:
            action: Action to execute

        Returns:
            Tuple of (next_state, reward, done)
            - next_state: Resulting game state
            - reward: Reward for the action
            - done: True if episode is finished
        """
        next_state = self.apply_action(self.state, action)
        reward = self.calculate_reward(self.state, action, next_state)
        done = self.is_terminal(next_state)
        self.state = next_state

        return next_state, reward, done

    def render(self):
        """
        Render the current game state (if graphics are available).

        Can be overridden to use pygame, matplotlib, or other rendering.
        """
        pass

    @abstractmethod
    def get_initial_state(self) -> Any:
        """
        Get the initial game state.

        Returns:
            Initial state
        """
        pass

    @abstractmethod
    def apply_action(self, state: Any, action: Any) -> Any:
        """
        Apply an action to a state.

        Args:
            state: Current game state
            action: Action to apply

        Returns:
            Resulting game state
        """
        pass

    @abstractmethod
    def calculate_reward(self, state: Any, action: Any, next_state: Any) -> float:
        """
        Calculate reward for an action.

        Args:
            state: State before action
            action: Action taken
            next_state: State after action

        Returns:
            Reward value
        """
        pass

    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        """
        Check if a state is terminal (game over).

        Args:
            state: Game state to check

        Returns:
            True if game is over
        """
        pass

    @abstractmethod
    def get_legal_actions(self, state: Any):
        """
        Get available actions from a state.

        Args:
            state: Current game state

        Returns:
            List of legal actions
        """
        pass


class PygameGameEnvironment(GameEnvironment):
    """
    Game environment with pygame rendering support.

    Extends GameEnvironment with pygame-based graphics.
    """

    def __init__(self, width=800, height=600, fps=60):
        """
        Initialize pygame environment.

        Args:
            width: Screen width
            height: Screen height
            fps: Frames per second for rendering
        """
        try:
            import pygame
            self.pygame = pygame
            self.has_pygame = True
        except ImportError:
            self.has_pygame = False
            print("Warning: pygame not installed. Graphics disabled.")

        self.fps = fps
        self.clock = None
        self.screen = None

        if self.has_pygame:
            self.pygame.init()
            self.screen = self.pygame.display.set_mode((width, height))
            self.clock = self.pygame.time.Clock()

        super().__init__(width, height)

    def render(self):
        """
        Render current game state with pygame.

        Fills screen white and calls draw_state for custom graphics.
        """
        if not self.has_pygame or not self.screen:
            return

        self.screen.fill((255, 255, 255))
        self.draw_state(self.state)
        self.pygame.display.flip()
        self.clock.tick(self.fps)

    def draw_state(self, state: Any):
        """
        Draw the game state on screen.

        Override this method to implement custom graphics.

        Args:
            state: Game state to render
        """
        pass

    def get_initial_state(self) -> Any:
        """Get initial state - must be implemented by subclass."""
        pass

    def apply_action(self, state: Any, action: Any) -> Any:
        """Apply action - must be implemented by subclass."""
        pass

    def calculate_reward(self, state: Any, action: Any, next_state: Any) -> float:
        """Calculate reward - must be implemented by subclass."""
        pass

    def is_terminal(self, state: Any) -> bool:
        """Check if terminal - must be implemented by subclass."""
        pass

    def get_legal_actions(self, state: Any):
        """Get legal actions - must be implemented by subclass."""
        pass

    def close(self):
        """Clean up pygame resources."""
        if self.has_pygame:
            self.pygame.quit()
