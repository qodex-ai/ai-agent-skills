"""
Workflow Management for Multi-Agent Systems

Handle workflow execution, monitoring, and optimization.
"""

from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
import time


class WorkflowStatus(Enum):
    """Workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """Task to be executed by an agent."""

    task_id: str
    agent: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class Workflow:
    """Workflow orchestration."""

    workflow_id: str
    name: str
    tasks: Dict[str, Task] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class WorkflowExecutor:
    """Execute workflows with multiple agents."""

    def __init__(self):
        """Initialize workflow executor."""
        self.workflows: Dict[str, Workflow] = {}
        self.execution_history: List[Dict] = []

    def create_workflow(self, workflow_id: str, name: str) -> Workflow:
        """Create new workflow."""
        workflow = Workflow(workflow_id=workflow_id, name=name)
        self.workflows[workflow_id] = workflow
        return workflow

    def add_task(
        self,
        workflow_id: str,
        task_id: str,
        agent: str,
        description: str,
        dependencies: Optional[List[str]] = None,
    ) -> Task:
        """Add task to workflow."""
        task = Task(
            task_id=task_id,
            agent=agent,
            description=description,
            dependencies=dependencies or [],
        )
        self.workflows[workflow_id].tasks[task_id] = task
        return task

    def execute_workflow(
        self, workflow_id: str, executor_func: Callable
    ) -> Dict[str, Any]:
        """
        Execute workflow.

        Args:
            workflow_id: Workflow ID
            executor_func: Function to execute tasks

        Returns:
            Execution results
        """
        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.RUNNING
        workflow.start_time = time.time()

        executed_tasks = set()
        results = {}

        while len(executed_tasks) < len(workflow.tasks):
            # Find ready tasks
            ready_tasks = self._get_ready_tasks(workflow, executed_tasks)

            if not ready_tasks:
                # Check if workflow is stuck
                if len(executed_tasks) > 0:
                    break

            for task_id in ready_tasks:
                task = workflow.tasks[task_id]
                task.status = TaskStatus.RUNNING

                try:
                    # Execute task
                    result = executor_func(task)
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    results[task_id] = result
                except Exception as e:
                    task.error = str(e)
                    task.status = TaskStatus.FAILED
                    results[task_id] = None

                task.end_time = time.time()
                executed_tasks.add(task_id)

        # Finalize workflow
        workflow.end_time = time.time()
        if all(task.status == TaskStatus.COMPLETED for task in workflow.tasks.values()):
            workflow.status = WorkflowStatus.COMPLETED
        else:
            workflow.status = WorkflowStatus.FAILED

        execution_record = {
            "workflow_id": workflow_id,
            "status": workflow.status.value,
            "duration": workflow.end_time - workflow.start_time,
            "results": results,
        }
        self.execution_history.append(execution_record)

        return results

    def _get_ready_tasks(
        self, workflow: Workflow, executed: set
    ) -> List[str]:
        """Get tasks ready to execute."""
        ready = []
        for task_id, task in workflow.tasks.items():
            if task_id not in executed and task.status == TaskStatus.PENDING:
                # Check if all dependencies are complete
                deps_met = all(dep in executed for dep in task.dependencies)
                if deps_met:
                    ready.append(task_id)

        return ready

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status."""
        workflow = self.workflows[workflow_id]
        return {
            "id": workflow.workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "tasks": {
                task_id: task.status.value
                for task_id, task in workflow.tasks.items()
            },
        }


class WorkflowOptimizer:
    """Optimize workflow execution."""

    @staticmethod
    def analyze_dependencies(workflow: Workflow) -> Dict[str, Any]:
        """Analyze task dependencies."""
        dep_graph = {}
        for task_id, task in workflow.tasks.items():
            dep_graph[task_id] = task.dependencies

        # Find critical path
        critical_path = WorkflowOptimizer._find_critical_path(
            dep_graph, workflow.tasks
        )

        # Find parallelizable tasks
        parallel_groups = WorkflowOptimizer._find_parallel_groups(dep_graph)

        return {
            "dependency_graph": dep_graph,
            "critical_path": critical_path,
            "parallelizable_groups": parallel_groups,
        }

    @staticmethod
    def _find_critical_path(dep_graph: Dict, tasks: Dict) -> List[str]:
        """Find tasks on critical path."""
        # Simple implementation - actual critical path is more complex
        return list(dep_graph.keys())

    @staticmethod
    def _find_parallel_groups(dep_graph: Dict) -> List[List[str]]:
        """Find groups of tasks that can execute in parallel."""
        groups = []
        remaining = set(dep_graph.keys())

        while remaining:
            # Find tasks with no dependencies in remaining set
            independent = [
                task
                for task in remaining
                if not any(dep in remaining for dep in dep_graph.get(task, []))
            ]
            if independent:
                groups.append(independent)
                remaining -= set(independent)
            else:
                break

        return groups

    @staticmethod
    def estimate_execution_time(
        workflow: Workflow, task_times: Dict[str, float]
    ) -> float:
        """Estimate total execution time."""
        parallel_groups = WorkflowOptimizer._find_parallel_groups({
            task_id: task.dependencies
            for task_id, task in workflow.tasks.items()
        })

        total_time = 0
        for group in parallel_groups:
            group_time = max(
                task_times.get(task_id, 1.0) for task_id in group
            )
            total_time += group_time

        return total_time

    @staticmethod
    def suggest_parallelization(workflow: Workflow) -> List[Dict[str, Any]]:
        """Suggest ways to parallelize workflow."""
        suggestions = []

        dep_graph = {
            task_id: task.dependencies
            for task_id, task in workflow.tasks.items()
        }

        # Find sequential chains that could be parallelized
        for task_id, dependencies in dep_graph.items():
            if len(dependencies) == 0:
                suggestions.append({
                    "task": task_id,
                    "suggestion": "Can be parallelized with other independent tasks",
                })

        return suggestions


class WorkflowMonitor:
    """Monitor workflow execution."""

    def __init__(self):
        """Initialize monitor."""
        self.events: List[Dict] = []

    def record_event(
        self, workflow_id: str, task_id: str, event_type: str, data: Dict
    ) -> None:
        """Record workflow event."""
        event = {
            "workflow_id": workflow_id,
            "task_id": task_id,
            "event_type": event_type,
            "timestamp": time.time(),
            "data": data,
        }
        self.events.append(event)

    def get_workflow_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow metrics."""
        workflow_events = [e for e in self.events if e["workflow_id"] == workflow_id]

        task_times = {}
        for event in workflow_events:
            task_id = event["task_id"]
            if event["event_type"] == "start":
                task_times[task_id] = {"start": event["timestamp"]}
            elif event["event_type"] == "complete":
                if task_id in task_times:
                    task_times[task_id]["end"] = event["timestamp"]

        # Calculate durations
        durations = {
            task_id: times.get("end", time.time()) - times["start"]
            for task_id, times in task_times.items()
        }

        return {
            "total_tasks": len(task_times),
            "task_durations": durations,
            "avg_duration": sum(durations.values()) / len(durations)
            if durations
            else 0,
            "max_duration": max(durations.values()) if durations else 0,
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Get overall performance report."""
        if not self.events:
            return {"status": "no_events"}

        workflow_ids = set(e["workflow_id"] for e in self.events)

        report = {}
        for workflow_id in workflow_ids:
            report[workflow_id] = self.get_workflow_metrics(workflow_id)

        return report
