"""
Game Agent Benchmarking and Tournament Tools

Utilities for testing and evaluating game agents including
tournament play, rating systems, and performance metrics.
"""

import random
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class AgentStats:
    """Statistics for an agent's performance."""

    name: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    elo_rating: float = 1600.0
    games_played: int = 0

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.wins + self.losses + self.draws
        return self.wins / total if total > 0 else 0.0

    @property
    def score(self) -> float:
        """Calculate total score (wins + 0.5 * draws)."""
        return self.wins + 0.5 * self.draws


class GameAgentBenchmark:
    """Benchmark suite for game-playing agents."""

    def __init__(self):
        """Initialize benchmark."""
        self.agent_stats: Dict[str, AgentStats] = {}

    def run_tournament(self, agents: List, num_games: int = 100) -> Dict[str, int]:
        """
        Run round-robin tournament between agents.

        Each pair plays both as white and black (if applicable).

        Args:
            agents: List of agent objects to compete
            num_games: Total games to play

        Returns:
            Dictionary mapping agent names to win counts
        """
        results = {agent.name: 0 for agent in agents}

        games_per_matchup = num_games // (len(agents) * (len(agents) - 1))

        for _ in range(games_per_matchup):
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    # Play agent i vs agent j
                    winner = self.play_game(agents[i], agents[j])
                    if winner:
                        results[winner.name] += 1

                    # Play agent j vs agent i
                    winner = self.play_game(agents[j], agents[i])
                    if winner:
                        results[winner.name] += 1

        return results

    def evaluate_elo_rating(self, agents: List, num_games: int = 100) -> Dict[str, float]:
        """
        Calculate Elo ratings for agents.

        Uses Elo rating system where rating changes depend on
        performance vs expected strength.

        Args:
            agents: List of agents to rate
            num_games: Number of games to play

        Returns:
            Dictionary mapping agent names to Elo ratings
        """
        # Initialize ratings
        elo_ratings = {agent.name: 1600.0 for agent in agents}

        for _ in range(num_games):
            agent1 = random.choice(agents)
            agent2 = random.choice([a for a in agents if a != agent1])

            winner = self.play_game(agent1, agent2)

            # Calculate expected outcome for each agent
            diff = elo_ratings[agent2.name] - elo_ratings[agent1.name]
            expected_1 = 1 / (1 + 10 ** (diff / 400))
            expected_2 = 1 - expected_1

            # Determine actual scores
            if winner == agent1:
                score_1 = 1.0
                score_2 = 0.0
            elif winner == agent2:
                score_1 = 0.0
                score_2 = 1.0
            else:  # Draw
                score_1 = 0.5
                score_2 = 0.5

            # Update ratings (K-factor = 32)
            elo_ratings[agent1.name] += 32 * (score_1 - expected_1)
            elo_ratings[agent2.name] += 32 * (score_2 - expected_2)

        return elo_ratings

    def glicko2_rating(self, agents: List, num_games: int = 100) -> Dict[str, float]:
        """
        Calculate Glicko-2 ratings (improved Elo system).

        Accounts for rating deviation (uncertainty).

        Args:
            agents: List of agents to rate
            num_games: Number of games to play

        Returns:
            Dictionary mapping agent names to Glicko-2 ratings
        """
        # Simplified Glicko-2 implementation
        ratings = {agent.name: 1600.0 for agent in agents}
        deviations = {agent.name: 350.0 for agent in agents}

        for _ in range(num_games):
            agent1 = random.choice(agents)
            agent2 = random.choice([a for a in agents if a != agent1])

            winner = self.play_game(agent1, agent2)

            # Calculate expected outcome
            diff = (ratings[agent2.name] - ratings[agent1.name]) / 173.7178
            expected = 1 / (1 + math.exp(-diff))

            # Determine actual score
            if winner == agent1:
                score = 1.0
            elif winner == agent2:
                score = 0.0
            else:
                score = 0.5

            # Update ratings with deviation factor
            d_squared = 1 / (0.0055 + 0.0001 * ((score - expected) ** 2))
            d = math.sqrt(d_squared)

            ratings[agent1.name] += (0.0055 / (1 / (deviations[agent1.name] ** 2) + 1 / d_squared)) * (score - expected)
            deviations[agent1.name] = math.sqrt(1 / (1 / (deviations[agent1.name] ** 2) + 1 / d_squared))

        return ratings

    def play_game(self, agent1, agent2) -> Optional:
        """
        Play a single game between two agents.

        Args:
            agent1: First agent
            agent2: Second agent

        Returns:
            Winning agent, or None for draw
        """
        pass

    def rate_agent_strength(self, agent, baseline_agents: List, num_games: int = 20) -> float:
        """
        Rate agent strength relative to baseline agents.

        Args:
            agent: Agent to rate
            baseline_agents: Baseline agents to compare against
            num_games: Games per baseline agent

        Returns:
            Strength score (higher = stronger)
        """
        wins = 0
        total = 0

        for baseline in baseline_agents:
            for _ in range(num_games):
                winner = self.play_game(agent, baseline)
                if winner == agent:
                    wins += 1
                total += 1

        return (wins / total * 100) if total > 0 else 0.0

    def performance_profile(self, agent, test_positions: List, time_limit: float = 1.0) -> Dict:
        """
        Get performance profile for an agent.

        Args:
            agent: Agent to test
            test_positions: Test positions to evaluate
            time_limit: Time limit per position in seconds

        Returns:
            Performance metrics dictionary
        """
        correct_moves = 0
        avg_time = 0
        move_quality_scores = []

        for position in test_positions:
            # Get agent's move
            move = agent.get_best_move()  # Would need timing implementation

            # Evaluate move quality
            quality = self.evaluate_move_quality(position, move)
            move_quality_scores.append(quality)

            if quality > 0.8:
                correct_moves += 1

        return {
            'accuracy': correct_moves / len(test_positions) if test_positions else 0.0,
            'avg_move_quality': sum(move_quality_scores) / len(move_quality_scores) if move_quality_scores else 0.0,
            'total_positions_tested': len(test_positions)
        }

    def evaluate_move_quality(self, position, move) -> float:
        """
        Evaluate quality of a move (0.0 to 1.0).

        Args:
            position: Game position
            move: Move to evaluate

        Returns:
            Quality score
        """
        pass

    def head_to_head_comparison(self, agent1, agent2, num_games: int = 50) -> Dict:
        """
        Detailed comparison between two agents.

        Args:
            agent1: First agent
            agent2: Second agent
            num_games: Number of games to play

        Returns:
            Comparison statistics
        """
        agent1_wins = 0
        agent2_wins = 0
        draws = 0

        for _ in range(num_games):
            winner = self.play_game(agent1, agent2)
            if winner == agent1:
                agent1_wins += 1
            elif winner == agent2:
                agent2_wins += 1
            else:
                draws += 1

        total = agent1_wins + agent2_wins + draws

        return {
            'agent1': agent1.name,
            'agent2': agent2.name,
            'agent1_wins': agent1_wins,
            'agent2_wins': agent2_wins,
            'draws': draws,
            'agent1_win_rate': agent1_wins / total if total > 0 else 0.0,
            'agent2_win_rate': agent2_wins / total if total > 0 else 0.0,
            'draw_rate': draws / total if total > 0 else 0.0
        }
