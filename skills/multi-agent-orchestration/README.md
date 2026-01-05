# Multi-Agent Orchestration - Code Structure

This skill uses supporting Python files to keep documentation lean and maintainable.

## Directory Structure

```
multi-agent-orchestration/
├── SKILL.md                           # Main documentation (patterns, concepts)
├── README.md                          # This file
├── examples/                          # Implementation examples
│   ├── orchestration_patterns.py      # Sequential, parallel, hierarchical, consensus
│   └── framework_implementations.py   # CrewAI, AutoGen, LangGraph, Swarm templates
└── scripts/                           # Utility modules
    ├── agent_communication.py         # Message broker, shared memory, protocols
    ├── workflow_management.py         # Workflow execution and optimization
    └── benchmarking.py                # Performance and collaboration metrics
```

## Running Examples

### 1. Orchestration Patterns
```bash
python examples/orchestration_patterns.py
```
Demonstrates sequential, parallel, hierarchical, and consensus orchestration.

### 2. Framework Templates
```bash
python examples/framework_implementations.py
```
Templates and configurations for CrewAI, AutoGen, LangGraph, and Swarm frameworks.

## Using the Utilities

### Agent Communication
```python
from scripts.agent_communication import MessageBroker, SharedMemory, CommunicationProtocol

# Set up communication
broker = MessageBroker()
shared_memory = SharedMemory()
protocol = CommunicationProtocol(broker, shared_memory)

# Send messages between agents
protocol.request_analysis("agent_a", "agent_b", "Analyze this topic")

# Share findings
protocol.share_findings("agent_a", "analysis_results", {"findings": "..."})

# Get communication stats
stats = broker.get_statistics()
```

### Workflow Management
```python
from scripts.workflow_management import WorkflowExecutor, WorkflowOptimizer

# Create and execute workflow
executor = WorkflowExecutor()
workflow = executor.create_workflow("workflow_1", "Analysis Workflow")

# Add tasks
executor.add_task("workflow_1", "task_1", "researcher", "Research the topic")
executor.add_task("workflow_1", "task_2", "analyst", "Analyze findings", dependencies=["task_1"])

# Execute
results = executor.execute_workflow("workflow_1", executor_func)

# Analyze workflow
analysis = WorkflowOptimizer.analyze_dependencies(workflow)
print(f"Critical path: {analysis['critical_path']}")
```

### Benchmarking
```python
from scripts.benchmarking import TeamBenchmark, AgentEffectiveness, CollaborationMetrics

# Benchmark team performance
benchmark = TeamBenchmark()
result = benchmark.run_benchmark("sequential_test", orchestrator, test_data)

# Track agent effectiveness
effectiveness = AgentEffectiveness()
effectiveness.record_agent_task("agent_a", "task_1", success=True, quality_score=0.95, duration=2.5)

# Get agent rankings
rankings = effectiveness.rank_agents()
for rank, agent, score, metrics in rankings:
    print(f"{rank}. {agent}: {score:.2f}")

# Analyze collaboration
collaboration = CollaborationMetrics()
collaboration.record_interaction("agent_a", "agent_b", "request", response_time=0.5, successful=True)
interaction_metrics = collaboration.get_interaction_metrics()
```

## Integration with SKILL.md

- SKILL.md contains conceptual information, orchestration patterns, and best practices
- Code examples are in `examples/` for clarity and runnable implementations
- Utilities are in `scripts/` for modular, reusable components
- This keeps token costs low while maintaining full functionality

## Orchestration Patterns Covered

1. **Sequential Orchestration** - Tasks execute one after another
2. **Parallel Orchestration** - Multiple agents work simultaneously
3. **Hierarchical Orchestration** - Manager coordinates specialist teams
4. **Consensus-Based** - Agents debate and reach consensus
5. **Adaptive Workflows** - Orchestration changes based on progress
6. **DAG-Based** - Workflow as directed acyclic graph

## Framework Implementations

- **CrewAI** - Clear roles, hierarchical structure
- **AutoGen** - Multi-turn conversations, group discussions
- **LangGraph** - State management, complex workflows
- **Swarm** - Simple handoffs, conversational workflows

## Key Features

- **Token Efficient**: Modular code structure reduces LLM context usage
- **Production Ready**: Includes monitoring, optimization, and benchmarking
- **Framework Agnostic**: Works with any agent framework
- **Communication Patterns**: Direct, tool-mediated, and manager-based
- **Performance Metrics**: Team and individual agent effectiveness tracking

## Communication Patterns

- **Direct Communication**: Agent-to-agent message passing
- **Tool-Mediated**: Agents use shared memory/database
- **Manager-Based**: Central coordinator manages communication
- **Broadcast**: One-to-many messaging

## Next Steps

1. Define agent roles and expertise
2. Choose orchestration pattern (sequential, parallel, hierarchical)
3. Select communication approach (direct, shared memory, manager)
4. Implement workflow with task definitions
5. Set up monitoring and metrics
6. Benchmark and optimize
7. Deploy and iterate
