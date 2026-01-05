"""
Performance Optimization Tools

Utilities for optimizing game agent performance including
transposition tables, killer heuristics, and parallel search.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, List, Any, Callable
from dataclasses import dataclass, field


@dataclass
class TranspositionEntry:
    """Entry in a transposition table."""

    depth: int
    score: int
    flag: str  # 'exact', 'lower', 'upper'
    move: Optional[Any] = None


class TranspositionTable:
    """
    Cache for evaluated positions to avoid re-computation.

    Stores evaluation results indexed by position hash.
    """

    def __init__(self, max_size: int = 1000000):
        """
        Initialize transposition table.

        Args:
            max_size: Maximum number of entries to store
        """
        self.table: Dict[int, TranspositionEntry] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

    def store(self, position_hash: int, depth: int, score: int, flag: str, move: Optional[Any] = None):
        """
        Store a position evaluation.

        Only stores if new evaluation is at greater depth.

        Args:
            position_hash: Hash of position
            depth: Search depth
            score: Evaluation score
            flag: Type of bound ('exact', 'lower', 'upper')
            move: Best move at this position
        """
        with self.lock:
            if position_hash not in self.table or self.table[position_hash].depth <= depth:
                self.table[position_hash] = TranspositionEntry(
                    depth=depth,
                    score=score,
                    flag=flag,
                    move=move
                )

    def lookup(self, position_hash: int, depth: int) -> Optional[int]:
        """
        Retrieve a stored evaluation.

        Args:
            position_hash: Hash of position
            depth: Required search depth

        Returns:
            Score if found at sufficient depth, None otherwise
        """
        with self.lock:
            if position_hash in self.table:
                entry = self.table[position_hash]
                if entry.depth >= depth:
                    self.hits += 1
                    return entry.score

            self.misses += 1
            return None

    def lookup_move(self, position_hash: int, depth: int) -> Optional[Any]:
        """
        Retrieve best move from transposition table.

        Args:
            position_hash: Hash of position
            depth: Required search depth

        Returns:
            Best move if available, None otherwise
        """
        with self.lock:
            if position_hash in self.table:
                entry = self.table[position_hash]
                if entry.depth >= depth:
                    return entry.move
            return None

    def hit_rate(self) -> float:
        """
        Calculate transposition table hit rate.

        Returns:
            Fraction of lookups that hit (0.0 to 1.0)
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def clear(self):
        """Clear all entries from table."""
        with self.lock:
            self.table.clear()
            self.hits = 0
            self.misses = 0

    def size(self) -> int:
        """
        Get current table size.

        Returns:
            Number of entries stored
        """
        return len(self.table)

    def efficiency(self) -> float:
        """
        Get table efficiency metric.

        Returns:
            Hit rate percentage (0-100)
        """
        return self.hit_rate() * 100


class KillerHeuristic:
    """
    Track moves that cause cutoffs at similar depths.

    Killer moves are moves at a given depth that have caused
    cutoffs at other nodes - likely to be good moves to try.
    """

    def __init__(self, max_depth: int = 20, num_killers: int = 2):
        """
        Initialize killer heuristic.

        Args:
            max_depth: Maximum search depth to track
            num_killers: Number of killer moves to track per depth
        """
        self.max_depth = max_depth
        self.num_killers = num_killers
        self.killers = [[None] * num_killers for _ in range(max_depth)]

    def record_killer(self, move: Any, depth: int):
        """
        Record a killer move at a depth.

        Args:
            move: Move that caused cutoff
            depth: Depth where move was played
        """
        if depth >= self.max_depth:
            return

        # Shift existing killers down
        if move != self.killers[depth][0]:
            for i in range(self.num_killers - 1, 0, -1):
                self.killers[depth][i] = self.killers[depth][i - 1]
            self.killers[depth][0] = move

    def get_killers(self, depth: int) -> List[Any]:
        """
        Get killer moves for a depth.

        Args:
            depth: Search depth

        Returns:
            List of killer moves for this depth
        """
        if depth >= self.max_depth:
            return []
        return [k for k in self.killers[depth] if k is not None]

    def is_killer(self, move: Any, depth: int) -> bool:
        """
        Check if a move is a killer at a depth.

        Args:
            move: Move to check
            depth: Search depth

        Returns:
            True if move is a killer at this depth
        """
        if depth >= self.max_depth:
            return False
        return move in self.killers[depth]

    def clear(self):
        """Clear all killer moves."""
        self.killers = [[None] * self.num_killers for _ in range(self.max_depth)]


class ParallelSearchCoordinator:
    """
    Coordinate parallel search over game tree.

    Distributes search work across multiple threads.
    """

    def __init__(self, num_threads: int = 4):
        """
        Initialize parallel search coordinator.

        Args:
            num_threads: Number of worker threads
        """
        self.num_threads = num_threads
        self.executor = ThreadPoolExecutor(max_workers=num_threads)

    def parallel_evaluate_moves(
        self,
        moves: List[Any],
        evaluate_func: Callable[[Any], int]
    ) -> Dict[Any, int]:
        """
        Evaluate multiple moves in parallel.

        Args:
            moves: List of moves to evaluate
            evaluate_func: Function that evaluates a move

        Returns:
            Dictionary mapping moves to scores
        """
        futures = {}
        for move in moves:
            future = self.executor.submit(evaluate_func, move)
            futures[move] = future

        results = {}
        for move, future in futures.items():
            results[move] = future.result()

        return results

    def parallel_minimax(
        self,
        moves: List[Any],
        minimax_func: Callable[[Any], int]
    ) -> tuple[Any, int]:
        """
        Run minimax search for multiple root moves in parallel.

        Args:
            moves: Root moves to explore
            minimax_func: Function that returns evaluation for a move

        Returns:
            Tuple of (best_move, best_score)
        """
        best_move = None
        best_score = float('-inf')

        futures = {}
        for move in moves:
            future = self.executor.submit(minimax_func, move)
            futures[move] = future

        for move, future in futures.items():
            score = future.result()
            if score > best_score:
                best_score = score
                best_move = move

        return best_move, best_score

    def shutdown(self):
        """Shutdown thread pool."""
        self.executor.shutdown(wait=True)


class SearchStatistics:
    """Track and report search statistics."""

    def __init__(self):
        """Initialize statistics tracker."""
        self.nodes_evaluated = 0
        self.nodes_cached = 0
        self.cutoffs = 0
        self.time_start = None
        self.time_end = None

    def record_node(self):
        """Record evaluation of one node."""
        self.nodes_evaluated += 1

    def record_cache_hit(self):
        """Record cache hit."""
        self.nodes_cached += 1

    def record_cutoff(self):
        """Record alpha-beta cutoff."""
        self.cutoffs += 1

    def branching_factor(self) -> float:
        """
        Calculate effective branching factor.

        Returns:
            Estimated branching factor
        """
        if self.nodes_evaluated == 0:
            return 0.0
        return self.nodes_evaluated / max(1, self.nodes_evaluated - self.cutoffs)

    def pruning_efficiency(self) -> float:
        """
        Calculate efficiency of pruning.

        Returns:
            Percentage of nodes pruned (0-100)
        """
        if self.nodes_evaluated == 0:
            return 0.0
        return (self.cutoffs / self.nodes_evaluated) * 100

    def cache_hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        Returns:
            Percentage of cached vs evaluated (0-100)
        """
        total = self.nodes_cached + self.nodes_evaluated
        if total == 0:
            return 0.0
        return (self.nodes_cached / total) * 100

    def summary(self) -> str:
        """
        Get summary of search statistics.

        Returns:
            Formatted string with statistics
        """
        return (
            f"Nodes evaluated: {self.nodes_evaluated}\n"
            f"Nodes cached: {self.nodes_cached}\n"
            f"Alpha-beta cutoffs: {self.cutoffs}\n"
            f"Branching factor: {self.branching_factor():.2f}\n"
            f"Pruning efficiency: {self.pruning_efficiency():.2f}%\n"
            f"Cache hit rate: {self.cache_hit_rate():.2f}%"
        )
