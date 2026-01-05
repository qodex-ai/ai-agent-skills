"""
Multi-Agent System Benchmarking

Evaluate team performance and agent effectiveness.
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
import time
import statistics


@dataclass
class BenchmarkResult:
    """Result from benchmark run."""

    test_name: str
    total_time: float
    task_count: int
    success_count: int
    failure_count: int
    quality_score: float
    cost_tokens: int


class TeamBenchmark:
    """Benchmark multi-agent team performance."""

    def __init__(self):
        """Initialize benchmarking suite."""
        self.results: List[BenchmarkResult] = []
        self.agent_metrics: Dict[str, Dict] = {}

    def run_benchmark(
        self,
        test_name: str,
        orchestrator,
        test_data: List[Dict],
        quality_metric: Optional[Callable] = None,
    ) -> BenchmarkResult:
        """
        Run benchmark test.

        Args:
            test_name: Name of benchmark
            orchestrator: Orchestrator instance
            test_data: Test cases
            quality_metric: Function to evaluate quality

        Returns:
            Benchmark result
        """
        start_time = time.time()
        results = []
        costs = []

        for test_case in test_data:
            try:
                result = orchestrator.execute(test_case)
                results.append(result)
                # Assume cost in test_case
                costs.append(test_case.get("cost", 0))
            except Exception:
                pass

        end_time = time.time()

        success_count = len(results)
        failure_count = len(test_data) - success_count
        total_time = end_time - start_time

        # Calculate quality score
        quality_score = (
            quality_metric(results) if quality_metric else success_count / len(test_data)
        )

        benchmark_result = BenchmarkResult(
            test_name=test_name,
            total_time=total_time,
            task_count=len(test_data),
            success_count=success_count,
            failure_count=failure_count,
            quality_score=quality_score,
            cost_tokens=sum(costs),
        )

        self.results.append(benchmark_result)
        return benchmark_result

    def compare_orchestration_methods(
        self,
        methods: Dict[str, Callable],
        test_data: List[Dict],
    ) -> Dict[str, Any]:
        """
        Compare different orchestration methods.

        Args:
            methods: Dict of method name to orchestrator
            test_data: Test cases

        Returns:
            Comparison results
        """
        comparison = {}

        for method_name, orchestrator in methods.items():
            result = self.run_benchmark(method_name, orchestrator, test_data)
            comparison[method_name] = {
                "total_time": result.total_time,
                "success_rate": result.success_count / result.task_count,
                "quality_score": result.quality_score,
                "efficiency": result.success_count / (result.total_time + 0.001),
                "cost_per_success": (
                    result.cost_tokens / result.success_count
                    if result.success_count > 0
                    else float("inf")
                ),
            }

        return comparison

    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary."""
        if not self.results:
            return {"status": "no_results"}

        times = [r.total_time for r in self.results]
        success_rates = [
            r.success_count / r.task_count for r in self.results
        ]
        quality_scores = [r.quality_score for r in self.results]

        return {
            "total_benchmarks": len(self.results),
            "avg_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "avg_success_rate": statistics.mean(success_rates),
            "avg_quality": statistics.mean(quality_scores),
            "benchmarks": [r.__dict__ for r in self.results],
        }


class AgentEffectiveness:
    """Measure individual agent effectiveness."""

    def __init__(self):
        """Initialize effectiveness tracker."""
        self.agent_stats: Dict[str, Dict] = {}

    def record_agent_task(
        self,
        agent: str,
        task_id: str,
        success: bool,
        quality_score: float,
        duration: float,
    ) -> None:
        """Record agent task execution."""
        if agent not in self.agent_stats:
            self.agent_stats[agent] = {
                "tasks": [],
                "successes": 0,
                "failures": 0,
            }

        stats = self.agent_stats[agent]
        stats["tasks"].append({
            "task_id": task_id,
            "success": success,
            "quality": quality_score,
            "duration": duration,
        })

        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1

    def get_agent_metrics(self, agent: str) -> Dict[str, Any]:
        """Get metrics for specific agent."""
        if agent not in self.agent_stats:
            return {"status": "no_data"}

        stats = self.agent_stats[agent]
        tasks = stats["tasks"]

        if not tasks:
            return {"status": "no_tasks"}

        success_rate = stats["successes"] / (stats["successes"] + stats["failures"])
        quality_scores = [t["quality"] for t in tasks]
        durations = [t["duration"] for t in tasks]

        return {
            "agent": agent,
            "total_tasks": len(tasks),
            "success_rate": success_rate,
            "avg_quality": statistics.mean(quality_scores),
            "avg_duration": statistics.mean(durations),
            "reliability": success_rate,  # Alias for clarity
        }

    def rank_agents(self) -> List[tuple]:
        """Rank agents by effectiveness."""
        rankings = []

        for agent in self.agent_stats.keys():
            metrics = self.get_agent_metrics(agent)
            if "success_rate" in metrics:
                score = (
                    metrics["success_rate"] * 0.4 +
                    metrics["avg_quality"] * 0.4 +
                    (1 - min(metrics["avg_duration"] / 10.0, 1.0)) * 0.2
                )
                rankings.append((agent, score, metrics))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_team_report(self) -> Dict[str, Any]:
        """Get team effectiveness report."""
        rankings = self.rank_agents()

        return {
            "total_agents": len(self.agent_stats),
            "rankings": [
                {
                    "rank": i + 1,
                    "agent": agent,
                    "score": score,
                    "metrics": metrics,
                }
                for i, (agent, score, metrics) in enumerate(rankings)
            ],
        }


class CollaborationMetrics:
    """Measure collaboration effectiveness between agents."""

    def __init__(self):
        """Initialize collaboration metrics."""
        self.interactions: List[Dict] = []

    def record_interaction(
        self,
        agent_a: str,
        agent_b: str,
        message: str,
        response_time: float,
        successful: bool,
    ) -> None:
        """Record agent-to-agent interaction."""
        self.interactions.append({
            "agent_a": agent_a,
            "agent_b": agent_b,
            "message": message,
            "response_time": response_time,
            "successful": successful,
        })

    def get_collaboration_graph(self) -> Dict[str, List[str]]:
        """Get collaboration graph between agents."""
        graph = {}

        for interaction in self.interactions:
            agent_a = interaction["agent_a"]
            agent_b = interaction["agent_b"]

            if agent_a not in graph:
                graph[agent_a] = []
            if agent_b not in graph[agent_a]:
                graph[agent_a].append(agent_b)

        return graph

    def get_interaction_metrics(self) -> Dict[str, Any]:
        """Get interaction metrics."""
        if not self.interactions:
            return {"total_interactions": 0}

        response_times = [i["response_time"] for i in self.interactions]
        success_rate = sum(1 for i in self.interactions if i["successful"]) / len(
            self.interactions
        )

        return {
            "total_interactions": len(self.interactions),
            "success_rate": success_rate,
            "avg_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "max_response_time": max(response_times),
            "collaboration_score": success_rate * (1 - min(statistics.mean(response_times) / 10.0, 1.0)),
        }

    def get_agent_collaboration_analysis(self, agent: str) -> Dict[str, Any]:
        """Get collaboration analysis for specific agent."""
        agent_interactions = [
            i for i in self.interactions
            if i["agent_a"] == agent or i["agent_b"] == agent
        ]

        if not agent_interactions:
            return {"agent": agent, "status": "no_interactions"}

        partners = set()
        for interaction in agent_interactions:
            if interaction["agent_a"] == agent:
                partners.add(interaction["agent_b"])
            else:
                partners.add(interaction["agent_a"])

        success_rate = sum(
            1 for i in agent_interactions if i["successful"]
        ) / len(agent_interactions)

        return {
            "agent": agent,
            "collaboration_partners": list(partners),
            "total_interactions": len(agent_interactions),
            "collaboration_success_rate": success_rate,
        }


def create_benchmark_suite() -> Dict[str, Callable]:
    """Create standard benchmark suite."""
    return {
        "sequential_tasks": lambda orchestrator: orchestrator.execute(
            {"type": "sequential", "task_count": 5}
        ),
        "parallel_tasks": lambda orchestrator: orchestrator.execute(
            {"type": "parallel", "task_count": 10}
        ),
        "complex_workflow": lambda orchestrator: orchestrator.execute(
            {"type": "complex", "task_count": 20}
        ),
        "error_handling": lambda orchestrator: orchestrator.execute(
            {"type": "with_errors", "task_count": 10}
        ),
    }
