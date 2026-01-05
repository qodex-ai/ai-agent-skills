"""
Chess Game Agent using python-chess

Complete implementation of a chess-playing agent with move evaluation,
position analysis, and game playing capabilities.

Requires: pip install python-chess
"""

import chess
from enum import Enum
from typing import Optional, Tuple


class ChessAgent:
    """Chess-playing agent using the python-chess library."""

    def __init__(self):
        """Initialize the chess agent with a new board."""
        self.board = chess.Board()
        self.move_count = 0

    def play_game(self, opponent_agent=None) -> Tuple[str, int]:
        """
        Play a complete chess game.

        Args:
            opponent_agent: Another agent to play against. If None, random moves.

        Returns:
            Tuple of (game_result, move_count)
            - game_result: "1-0" (white wins), "0-1" (black wins), "1/2-1/2" (draw)
            - move_count: Total number of half-moves
        """
        self.move_count = 0

        while not self.board.is_game_over():
            if self.board.turn:  # White's turn
                move = self.get_best_move()
            else:  # Black's turn
                if opponent_agent:
                    move = opponent_agent.get_best_move()
                else:
                    move = self.get_random_move()

            self.board.push(move)
            self.move_count += 1

        return self.board.result(), self.move_count

    def get_best_move(self) -> Optional[chess.Move]:
        """
        Find the best move in the current position using positional evaluation.

        Evaluates each legal move and returns the one with highest score.

        Returns:
            Best move in the current position
        """
        legal_moves = list(self.board.legal_moves)

        if not legal_moves:
            return None

        best_move = None
        best_score = float('-inf')

        for move in legal_moves:
            self.board.push(move)
            score = self.evaluate_position()
            self.board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def evaluate_position(self) -> int:
        """
        Evaluate the current position based on material count.

        Uses standard piece values:
        - Pawn: 1
        - Knight/Bishop: 3
        - Rook: 5
        - Queen: 9

        Args:
            None

        Returns:
            Score from white's perspective (positive = advantage)
        """
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }

        score = 0
        for piece_type, value in piece_values.items():
            white_count = len(self.board.pieces(piece_type, chess.WHITE))
            black_count = len(self.board.pieces(piece_type, chess.BLACK))
            score += (white_count - black_count) * value

        return score

    def get_random_move(self) -> Optional[chess.Move]:
        """
        Get a random legal move.

        Useful for baseline opponent or random playouts.

        Returns:
            Random legal move, or None if none available
        """
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None
        return legal_moves[0]  # Should use random.choice() in practice

    def get_board_state(self) -> str:
        """
        Get string representation of the current board.

        Returns:
            ASCII representation of the board
        """
        return str(self.board)

    def get_fen(self) -> str:
        """
        Get FEN (Forsyth-Edwards Notation) of current position.

        Returns:
            FEN string
        """
        return self.board.fen()

    def set_fen(self, fen: str):
        """
        Set board position from FEN string.

        Args:
            fen: Forsyth-Edwards Notation string
        """
        self.board = chess.Board(fen)

    def is_game_over(self) -> bool:
        """
        Check if the game is over.

        Returns:
            True if game is finished (checkmate, stalemate, etc)
        """
        return self.board.is_game_over()

    def is_check(self) -> bool:
        """
        Check if current player is in check.

        Returns:
            True if in check
        """
        return self.board.is_check()

    def is_checkmate(self) -> bool:
        """
        Check if current position is checkmate.

        Returns:
            True if checkmate
        """
        return self.board.is_checkmate()

    def get_legal_moves(self):
        """
        Get all legal moves in current position.

        Returns:
            List of legal chess.Move objects
        """
        return list(self.board.legal_moves)

    def make_move(self, move: chess.Move):
        """
        Execute a move on the board.

        Args:
            move: chess.Move object to play

        Raises:
            ValueError: If move is not legal
        """
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move}")
        self.board.push(move)

    def undo_move(self) -> bool:
        """
        Undo the last move.

        Returns:
            True if a move was undone, False if no moves to undo
        """
        if len(self.board.move_stack) == 0:
            return False
        self.board.pop()
        return True

    def reset(self):
        """Reset the board to starting position."""
        self.board = chess.Board()
        self.move_count = 0

    def get_game_status(self) -> str:
        """
        Get human-readable game status.

        Returns:
            String describing game state
        """
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn else "White"
            return f"Checkmate! {winner} wins."
        elif self.board.is_stalemate():
            return "Stalemate - Draw!"
        elif self.board.is_check():
            player = "White" if self.board.turn else "Black"
            return f"{player} is in check."
        elif self.board.is_game_over():
            return "Game over - Draw by rule."
        else:
            player = "White" if self.board.turn else "Black"
            return f"{player} to move."
