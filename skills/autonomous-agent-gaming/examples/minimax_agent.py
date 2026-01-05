"""
Minimax with Alpha-Beta Pruning Implementation

Optimal decision-making algorithm for turn-based games.
Uses minimax tree search with alpha-beta pruning for efficiency.
"""


class MinimaxGameAgent:
    """Game agent using minimax algorithm with alpha-beta pruning."""

    def __init__(self, max_depth=4):
        """
        Initialize the minimax agent.

        Args:
            max_depth: Maximum search depth for the minimax tree
        """
        self.max_depth = max_depth
        self.nodes_evaluated = 0

    def get_best_move(self, game_state):
        """
        Find the best move using minimax with alpha-beta pruning.

        Args:
            game_state: Current game state

        Returns:
            Best move to play
        """
        _, best_move = self.minimax(
            game_state,
            self.max_depth,
            True,  # Maximizing player
            float('-inf'),
            float('inf')
        )
        return best_move

    def minimax(self, game_state, depth, maximizing, alpha, beta):
        """
        Minimax algorithm with alpha-beta pruning.

        Recursively evaluates the game tree, alternating between maximizing
        and minimizing player moves. Prunes branches that cannot affect the
        final decision.

        Args:
            game_state: Current game state
            depth: Remaining search depth
            maximizing: True if maximizing player's turn
            alpha: Best value for maximizer so far
            beta: Best value for minimizer so far

        Returns:
            Tuple of (evaluation_score, best_move)
        """
        self.nodes_evaluated += 1

        # Terminal node or max depth reached
        if depth == 0 or game_state.is_terminal():
            return self.evaluate(game_state), None

        if maximizing:
            max_eval = float('-inf')
            best_move = None

            for move in game_state.get_legal_moves():
                next_state = game_state.apply_move(move)
                eval_score, _ = self.minimax(
                    next_state, depth - 1, False, alpha, beta
                )

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff - prune remaining branches

            return max_eval, best_move

        else:  # Minimizing player
            min_eval = float('inf')
            best_move = None

            for move in game_state.get_legal_moves():
                next_state = game_state.apply_move(move)
                eval_score, _ = self.minimax(
                    next_state, depth - 1, True, alpha, beta
                )

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff - prune remaining branches

            return min_eval, best_move

    def evaluate(self, game_state):
        """
        Static evaluation function for terminal or leaf nodes.

        Args:
            game_state: Game state to evaluate

        Returns:
            Numeric evaluation score
        """
        if game_state.is_checkmate():
            return float('-inf') if game_state.current_player else float('inf')
        if game_state.is_stalemate():
            return 0

        # Material count (can be extended with other evaluation factors)
        return game_state.evaluate_material()

    def reset_stats(self):
        """Reset evaluation statistics."""
        self.nodes_evaluated = 0
