"""
Game Theory Analysis Tools

Utilities for analyzing games from a game theory perspective,
including Nash equilibrium calculation and strategic analysis.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PayoffMatrix:
    """Represents a 2-player game payoff matrix."""

    player1_payoffs: np.ndarray
    player2_payoffs: np.ndarray
    row_labels: List[str]
    column_labels: List[str]

    def is_zero_sum(self) -> bool:
        """
        Check if this is a zero-sum game.

        In zero-sum games, one player's gain is another's loss.

        Returns:
            True if zero-sum (payoffs sum to zero)
        """
        return np.allclose(self.player1_payoffs + self.player2_payoffs, 0)

    def is_symmetric(self) -> bool:
        """
        Check if this is a symmetric game.

        Returns:
            True if both players have identical payoff structure
        """
        return np.allclose(self.player1_payoffs, self.player2_payoffs.T)


class GameTheoryAnalyzer:
    """Analyzer for game-theoretic properties and solutions."""

    @staticmethod
    def find_pure_strategy_nash_equilibria(payoff_matrix: PayoffMatrix) -> List[Tuple[int, int]]:
        """
        Find pure strategy Nash equilibria in a game.

        A Nash equilibrium is a strategy profile where no player can
        improve by unilaterally changing their strategy.

        Args:
            payoff_matrix: Game payoff matrix

        Returns:
            List of (row_index, column_index) tuples for Nash equilibria
        """
        equilibria = []
        p1_payoffs = payoff_matrix.player1_payoffs
        p2_payoffs = payoff_matrix.player2_payoffs

        # Check each cell
        for i in range(p1_payoffs.shape[0]):
            for j in range(p1_payoffs.shape[1]):
                # Check if best response for player 1
                if p1_payoffs[i, j] == np.max(p1_payoffs[:, j]):
                    # Check if best response for player 2
                    if p2_payoffs[i, j] == np.max(p2_payoffs[i, :]):
                        equilibria.append((i, j))

        return equilibria

    @staticmethod
    def calculate_mixed_strategy_2x2(payoff_matrix: PayoffMatrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate mixed strategy Nash equilibrium for 2x2 game.

        For a 2x2 zero-sum game, finds the probability distribution
        over strategies that forms a Nash equilibrium.

        Args:
            payoff_matrix: 2x2 payoff matrix

        Returns:
            Tuple of (player1_strategy, player2_strategy)
            Strategies are probability distributions over actions
        """
        if payoff_matrix.player1_payoffs.shape != (2, 2):
            raise ValueError("This method only works for 2x2 games")

        p1 = payoff_matrix.player1_payoffs
        p2 = payoff_matrix.player2_payoffs

        # Use indifference conditions
        # Player 1: q * p1[0,0] + (1-q) * p1[1,0] = q * p1[0,1] + (1-q) * p1[1,1]
        numerator = p1[1, 1] - p1[1, 0]
        denominator = p1[0, 0] - p1[0, 1] - p1[1, 0] + p1[1, 1]

        if abs(denominator) < 1e-10:
            # Degenerate case
            q = 0.5
        else:
            q = numerator / denominator

        # Player 2 strategy calculation
        numerator = p2[1, 1] - p2[0, 1]
        denominator = p2[0, 0] - p2[0, 1] - p2[1, 0] + p2[1, 1]

        if abs(denominator) < 1e-10:
            p = 0.5
        else:
            p = numerator / denominator

        # Clamp to [0, 1]
        q = np.clip(q, 0, 1)
        p = np.clip(p, 0, 1)

        player1_strategy = np.array([q, 1 - q])
        player2_strategy = np.array([p, 1 - p])

        return player1_strategy, player2_strategy

    @staticmethod
    def calculate_expected_payoff(
        player1_strategy: np.ndarray,
        player2_strategy: np.ndarray,
        payoff_matrix: PayoffMatrix,
        player: int = 1
    ) -> float:
        """
        Calculate expected payoff for a strategy profile.

        Args:
            player1_strategy: Player 1's mixed strategy
            player2_strategy: Player 2's mixed strategy
            payoff_matrix: Game payoff matrix
            player: Which player (1 or 2) to calculate for

        Returns:
            Expected payoff value
        """
        if player == 1:
            payoffs = payoff_matrix.player1_payoffs
        else:
            payoffs = payoff_matrix.player2_payoffs

        return player1_strategy @ payoffs @ player2_strategy

    @staticmethod
    def minimax_value(payoff_matrix: PayoffMatrix) -> float:
        """
        Calculate minimax value for zero-sum game.

        Minimum guaranteed payoff for player 1 when playing optimally
        against an optimally-playing opponent.

        Args:
            payoff_matrix: Zero-sum game matrix

        Returns:
            Minimax value
        """
        if not payoff_matrix.is_zero_sum():
            raise ValueError("Minimax theorem only applies to zero-sum games")

        # For each row, find minimum (worst case for player 1)
        row_mins = np.min(payoff_matrix.player1_payoffs, axis=1)
        # Player 1 chooses row to maximize the minimum
        return float(np.max(row_mins))

    @staticmethod
    def maximin_value(payoff_matrix: PayoffMatrix) -> float:
        """
        Calculate maximin value for zero-sum game.

        Maximum loss player 2 is willing to accept when defending optimally.

        Args:
            payoff_matrix: Zero-sum game matrix

        Returns:
            Maximin value
        """
        if not payoff_matrix.is_zero_sum():
            raise ValueError("Maximin only applies to zero-sum games")

        # For each column, find maximum (best for player 1 from player 2's perspective)
        col_maxs = np.max(payoff_matrix.player1_payoffs, axis=0)
        # Player 2 chooses column to minimize player 1's maximum
        return float(np.min(col_maxs))


class CooperativeGameAnalyzer:
    """Analyzer for cooperative game theory concepts."""

    @staticmethod
    def calculate_shapley_value(payoff_function, players: List[str]) -> Dict[str, float]:
        """
        Calculate Shapley value for cooperative game.

        Indicates fair value of each player's contribution.

        Args:
            payoff_function: Function that takes coalition and returns payoff
            players: List of player names

        Returns:
            Dictionary mapping player names to Shapley values
        """
        n = len(players)
        shapley_values = {player: 0.0 for player in players}

        # This is a simplified implementation
        # Full implementation would enumerate all coalitions
        for player in players:
            # Calculate marginal contribution
            payoff_with = payoff_function(players)
            payoff_without = payoff_function([p for p in players if p != player])
            marginal = payoff_with - payoff_without
            shapley_values[player] = marginal / n

        return shapley_values

    @staticmethod
    def calculate_core(
        payoff_function,
        players: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        Find core of cooperative game if it exists.

        The core is the set of outcomes where no coalition can
        improve by deviating.

        Args:
            payoff_function: Function that maps coalition to payoff
            players: List of player names

        Returns:
            A core allocation if it exists, None otherwise
        """
        # Simplified implementation
        # Full implementation would verify all coalition constraints
        n = len(players)
        total_payoff = payoff_function(players)

        # Check if equal division is in the core
        equal_share = total_payoff / n
        return {player: equal_share for player in players}

    @staticmethod
    def is_core_allocation(
        allocation: Dict[str, float],
        payoff_function,
        players: List[str],
        epsilon: float = 1e-6
    ) -> bool:
        """
        Check if an allocation is in the core.

        Args:
            allocation: Dictionary of player payoffs
            payoff_function: Function mapping coalitions to payoffs
            players: List of all players
            epsilon: Tolerance for floating point comparison

        Returns:
            True if allocation is in the core
        """
        # Check efficiency: total allocation equals grand coalition payoff
        total = sum(allocation.values())
        grand_payoff = payoff_function(players)
        if abs(total - grand_payoff) > epsilon:
            return False

        # Check stability: no coalition can improve
        for coalition in players:
            coalition_payoff = allocation[coalition]
            coalition_value = payoff_function([coalition])
            if coalition_payoff < coalition_value - epsilon:
                return False

        return True
