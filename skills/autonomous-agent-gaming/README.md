# Autonomous Agent Gaming - Code Structure

This directory contains extracted, well-organized Python code for building autonomous game-playing agents. The refactoring separates implementation code from documentation for better maintainability and reusability.

## Directory Structure

```
autonomous-agent-gaming/
├── SKILL.md                      # Main skill documentation with concepts and references
├── README.md                     # This file
├── examples/                     # Agent implementations and game environments
│   ├── rule_based_agent.py      # Agents using predefined heuristics
│   ├── minimax_agent.py         # Minimax with alpha-beta pruning
│   ├── mcts_agent.py            # Monte Carlo Tree Search
│   ├── qlearning_agent.py       # Q-learning reinforcement learning
│   ├── chess_engine.py          # Chess-specific implementation
│   ├── game_environment.py      # Base classes for custom game environments
│   └── strategy_modules.py      # Opening books, endgame tablebases, adaptive strategies
└── scripts/                      # Utility and analysis tools
    ├── performance_optimizer.py  # Transposition tables, killer heuristic, parallel search
    ├── game_theory_analyzer.py  # Nash equilibrium, Shapley values, cooperative games
    └── agent_benchmark.py        # Tournament evaluation, Elo ratings, profiling
```

## File Descriptions

### Examples (agents and environments)

#### rule_based_agent.py
Simple agents using predefined rules and heuristics. Fast decision-making suitable for real-time games.

**Main Classes:**
- `RuleBasedGameAgent`: Evaluates positions based on material, positional factors, and control

**Key Methods:**
- `decide_action(game_state)`: Choose action based on rules
- `evaluate_position(game_state)`: Heuristic evaluation

#### minimax_agent.py
Optimal decision-making for turn-based games using exhaustive tree search with alpha-beta pruning.

**Main Classes:**
- `MinimaxGameAgent`: Minimax with alpha-beta pruning

**Key Methods:**
- `get_best_move(game_state)`: Find optimal move
- `minimax(game_state, depth, maximizing, alpha, beta)`: Core algorithm
- `evaluate(game_state)`: Static evaluation function

**Performance:**
- Pruning reduces complexity from O(b^d) to O(b^(d/2))
- Adjustable depth for speed/quality tradeoff

#### mcts_agent.py
Probabilistic game tree exploration using Monte Carlo Tree Search (AlphaGo algorithm).

**Main Classes:**
- `MCTSNode`: Node in MCTS tree
- `MCTSAgent`: MCTS implementation

**Four Phases:**
1. **Tree Policy**: Selection using UCT, then expansion
2. **Default Policy**: Random playout from leaf
3. **Backup**: Backpropagate results up tree
4. **Iteration**: Repeat until time/iteration limit

**Key Methods:**
- `get_best_move(game_state)`: Run MCTS iterations and return best move
- `calculate_uct(node, parent_visits)`: UCT = exploitation + exploration

#### qlearning_agent.py
Reinforcement learning agent that learns optimal policies through interaction.

**Main Classes:**
- `QLearningAgent`: Q-learning algorithm

**Key Methods:**
- `get_action(state)`: Epsilon-greedy action selection
- `update_q_value(state, action, reward, next_state)`: Update Q-table
- `decay_epsilon()`: Gradually reduce exploration
- `save_q_table()` / `load_q_table()`: Persistence

**Hyperparameters:**
- `learning_rate (α)`: How fast to adapt (0.0-1.0)
- `discount_factor (γ)`: Future reward importance (0.0-1.0)
- `epsilon (ε)`: Exploration probability (0.0-1.0)

#### chess_engine.py
Full chess implementation using python-chess library.

**Main Classes:**
- `ChessAgent`: Chess-playing agent

**Key Methods:**
- `play_game(opponent_agent)`: Play complete game
- `get_best_move()`: Find best move by evaluation
- `evaluate_position()`: Material-based evaluation
- `get_legal_moves()`: List legal moves
- `make_move(move)` / `undo_move()`: Move management

**Features:**
- Full game rules validation
- FEN notation support
- Move history tracking
- Game status reporting

#### game_environment.py
Abstract base classes for custom game environments.

**Main Classes:**
- `GameEnvironment`: Abstract base class
- `PygameGameEnvironment`: Pygame-based rendering

**Key Methods:**
- `reset()`: Initialize game
- `step(action)`: Execute action, return (state, reward, done)
- `render()`: Display game
- `get_legal_actions(state)`: List valid moves
- `is_terminal(state)`: Check game over

#### strategy_modules.py
Advanced strategies for different game phases.

**Main Classes:**
- `OpeningBook`: Pre-computed opening moves
- `EndgameTablebase`: Pre-computed endgame solutions
- `AdaptiveGameAgent`: Combines strategies by phase
- `CompositeStrategy`: Priority-based strategy selection
- `StrategyModule`: Base class for pluggable strategies

**Game Phases:**
- **Opening** (Material > 30): Use opening book
- **Middlegame** (10-30): Use search engine
- **Endgame** (Material < 10): Use tablebase

### Scripts (utilities and analysis)

#### performance_optimizer.py
Tools for optimizing search performance.

**Main Classes:**
- `TranspositionTable`: Cache for evaluated positions
- `KillerHeuristic`: Track cutoff-causing moves
- `ParallelSearchCoordinator`: Distribute search across threads
- `SearchStatistics`: Track search metrics

**Optimization Techniques:**
- **Transposition Tables**: Avoid re-evaluating positions
  - Storage: position_hash -> (depth, score, flag)
  - Bound types: 'exact', 'lower', 'upper'
  - Hit rate tracking for efficiency analysis

- **Killer Heuristic**: Improve move ordering
  - Killer moves cause cutoffs at given depths
  - Try killers early in move ordering

- **Parallel Search**: Distribute work across threads
  - Evaluate multiple root moves in parallel
  - Thread-safe for concurrent access

- **Search Statistics**: Measure optimization effectiveness
  - Nodes evaluated/pruned
  - Branching factor
  - Pruning efficiency percentage

#### game_theory_analyzer.py
Game-theoretic analysis and solution concepts.

**Main Classes:**
- `PayoffMatrix`: 2-player game representation
- `GameTheoryAnalyzer`: Non-cooperative game analysis
- `CooperativeGameAnalyzer`: Coalition and fairness analysis

**Key Concepts:**
- **Nash Equilibrium**: Strategy profile where no player can improve unilaterally
  - Pure strategy: No randomization
  - Mixed strategy: Probability distribution over actions

- **Minimax Theorem**: In zero-sum games, minimax = maximin

- **Shapley Value**: Fair allocation based on marginal contributions

- **Core**: Allocations where no coalition wants to deviate

**Key Methods:**
- `find_pure_strategy_nash_equilibria(payoff_matrix)`: Identify equilibria
- `calculate_mixed_strategy_2x2(payoff_matrix)`: Mixed Nash for 2x2 games
- `minimax_value()` / `maximin_value()`: Zero-sum game values
- `calculate_shapley_value()`: Fair allocation
- `calculate_core()`: Stable coalitional outcomes

#### agent_benchmark.py
Comprehensive benchmarking and evaluation toolkit.

**Main Classes:**
- `AgentStats`: Track agent performance metrics
- `GameAgentBenchmark`: Tournament and rating systems

**Key Methods:**
- `run_tournament(agents, num_games)`: Round-robin tournament
- `evaluate_elo_rating(agents, num_games)`: Elo rating system
- `glicko2_rating(agents, num_games)`: Glicko-2 ratings with uncertainty
- `head_to_head_comparison(agent1, agent2, num_games)`: Detailed comparison
- `rate_agent_strength(agent, baselines, num_games)`: Strength evaluation
- `performance_profile(agent, test_positions, time_limit)`: Position accuracy

**Rating Systems:**
- **Elo**: Traditional rating system
  - K-factor = 32
  - Based on strength differential

- **Glicko-2**: Improved system with uncertainty
  - Accounts for rating deviation
  - Better for irregular schedules

## Quick Start

### 1. Import and Use an Agent

```python
from examples.minimax_agent import MinimaxGameAgent

# Create agent
agent = MinimaxGameAgent(max_depth=6)

# Get best move
best_move = agent.get_best_move(game_state)
```

### 2. Train a Q-Learning Agent

```python
from examples.qlearning_agent import QLearningAgent

agent = QLearningAgent(learning_rate=0.1, discount_factor=0.99, epsilon=0.1)

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_value(state, action, reward, next_state)
        state = next_state

    agent.decay_epsilon()  # Reduce exploration over time

# Save learned policy
agent.save_q_table('q_table.json')
```

### 3. Play Chess

```python
from examples.chess_engine import ChessAgent

agent1 = ChessAgent()
agent2 = ChessAgent()

result, moves = agent1.play_game(agent2)
print(f"Result: {result} ({moves} moves)")
```

### 4. Benchmark Agents

```python
from scripts.agent_benchmark import GameAgentBenchmark

benchmark = GameAgentBenchmark()

# Run tournament
results = benchmark.run_tournament(agents, num_games=100)

# Get Elo ratings
ratings = benchmark.evaluate_elo_rating(agents, num_games=100)
for agent in agents:
    print(f"{agent.name}: {ratings[agent.name]:.0f}")
```

### 5. Optimize with Transposition Tables

```python
from scripts.performance_optimizer import TranspositionTable

tt = TranspositionTable(max_size=1000000)

# During search
position_hash = hash(game_state)
cached_score = tt.lookup(position_hash, depth=6)

if cached_score is None:
    # Evaluate position
    score = evaluate(game_state)
    tt.store(position_hash, depth=6, score, flag='exact')
else:
    score = cached_score

print(f"Cache hit rate: {tt.hit_rate():.1%}")
```

### 6. Game Theory Analysis

```python
from scripts.game_theory_analyzer import GameTheoryAnalyzer, PayoffMatrix
import numpy as np

# Prisoner's Dilemma
payoffs_p1 = np.array([[-1, -3], [0, -2]])
payoffs_p2 = np.array([[-1, 0], [-3, -2]])

matrix = PayoffMatrix(payoffs_p1, payoffs_p2)

analyzer = GameTheoryAnalyzer()
equilibria = analyzer.find_pure_strategy_nash_equilibria(matrix)
print(f"Nash equilibria: {equilibria}")
```

## Integration with SKILL.md

The SKILL.md file contains conceptual explanations and usage examples that reference these code files. When reading SKILL.md:

- Quick Start section shows how to run each module
- Each algorithm section references the corresponding .py file
- Code examples show imports from examples/ and scripts/
- For detailed implementation, refer to the corresponding file

## Dependencies

### Core (built-in)
- dataclasses
- enum
- typing
- abc
- threading
- concurrent.futures
- collections
- math
- random

### Optional (install as needed)
- python-chess: `pip install python-chess` (for chess_engine.py)
- pygame: `pip install pygame` (for PygameGameEnvironment)
- numpy: `pip install numpy` (for game_theory_analyzer.py)
- gym: `pip install gym` (for OpenAI Gym integration)

## Design Principles

1. **Modularity**: Each agent/algorithm is self-contained and reusable
2. **Extensibility**: Abstract base classes allow easy customization
3. **Simplicity**: Code is readable with clear method names and docstrings
4. **Type Hints**: Full type annotations for IDE support and documentation
5. **No Redundancy**: Shared functionality factored into utility modules
6. **Documentation**: Inline docstrings explain complex algorithms

## Best Practices When Using

1. **Start Simple**: Begin with RuleBasedGameAgent, then progress to Minimax/MCTS
2. **Profile Before Optimizing**: Use SearchStatistics to identify bottlenecks
3. **Benchmark Regularly**: Compare agents using GameAgentBenchmark
4. **Version Control**: Save trained models (Q-tables, opening books)
5. **Document Changes**: Track why you modified evaluation functions or hyperparameters
6. **Test Edge Cases**: Verify behavior at game boundaries and end states

## Common Patterns

### Creating a Custom Agent
```python
class MyAgent:
    def __init__(self):
        # Initialize parameters
        pass

    def get_action(self, game_state):
        # Return best action
        pass
```

### Combining Strategies
```python
from examples.strategy_modules import CompositeStrategy

composite = CompositeStrategy([
    opening_strategy,
    middlegame_strategy,
    endgame_strategy
])

move = composite.get_move(game_state)
```

### Parallel Search
```python
from scripts.performance_optimizer import ParallelSearchCoordinator

coordinator = ParallelSearchCoordinator(num_threads=4)
best_move, score = coordinator.parallel_minimax(root_moves, search_func)
```

## Further Reading

- See SKILL.md for detailed algorithm explanations
- Read docstrings in each module for implementation details
- Check method signatures for parameter and return types
- Explore examples/ and scripts/ for working code patterns
