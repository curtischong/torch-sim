"""Profiling utilities for torch-sim simulations.

This module provides profiling tools to identify performance bottlenecks in
simulations. It uses PyTorch's built-in profiler which can export to Chrome
Trace format for viewing flame graphs in Chrome DevTools (chrome://tracing).

Example usage::

    import torch_sim as ts
    from torch_sim.profiling import Profiler

    # Profile an optimization
    with Profiler("my_optimization") as prof:
        state = ts.optimize(system, model, optimizer="fire", max_steps=100)

    # View results in Chrome DevTools by opening: chrome://tracing
    # Then load the generated JSON file: my_optimization_trace.json

    # Or use the convenience function for quick profiling
    from torch_sim.profiling import profile_run

    result, prof = profile_run(
        ts.optimize,
        system, model, optimizer="fire", max_steps=100,
        profile_name="optimize_profile"
    )
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from torch.profiler import ProfilerActivity, profile, record_function


if TYPE_CHECKING:
    import types
    from collections.abc import Callable, Generator

    from torch.profiler import profile as _profile_context_manager


__all__ = [
    "Profiler",
    "ProfilerConfig",
    "profile_run",
    "profiling_section",
    "record_function",
]


@dataclass
class ProfilerConfig:
    """Configuration for the Profiler.

    Attributes:
        activities: Which activities to profile. Defaults to CPU and CUDA.
        record_shapes: Whether to record tensor shapes.
        profile_memory: Whether to profile memory allocations.
        with_stack: Whether to record stack traces. Enables better flame graphs
            but adds overhead.
        with_flops: Whether to estimate FLOPs for operators.
        with_modules: Whether to record module hierarchy.
        row_limit: Number of rows to show in printed table output.
    """

    activities: list[ProfilerActivity] = field(
        default_factory=lambda: [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    )
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = True
    with_flops: bool = True
    with_modules: bool = True
    row_limit: int = 20


class Profiler:
    """A profiler for torch-sim simulations that exports Chrome Trace format.

    This profiler wraps PyTorch's profiler and provides convenient export to
    Chrome Trace format for viewing flame graphs in Chrome DevTools.

    Example::

        # Basic usage
        with Profiler("simulation") as prof:
            state = ts.optimize(system, model, optimizer="fire", max_steps=100)

        # Custom output directory
        with Profiler("simulation", output_dir="./profiles") as prof:
            state = ts.integrate(system, model, integrator="nvt_langevin", n_steps=1000)

        # Access profiler stats
        print(prof.key_averages().table(sort_by="cuda_time_total"))

    Attributes:
        name: Name for the profiling session, used in output filenames.
        output_dir: Directory for output files. Defaults to current directory.
        config: ProfilerConfig with detailed settings.
        trace_path: Path to the exported Chrome Trace file after profiling.
    """

    def __init__(
        self,
        name: str = "torch_sim_profile",
        output_dir: str | Path = ".",
        config: ProfilerConfig | None = None,
        *,
        auto_export: bool = True,
    ) -> None:
        """Initialize the profiler.

        Args:
            name: Name for the profiling session, used in output filenames.
            output_dir: Directory for output files.
            config: ProfilerConfig with detailed settings. If None, uses defaults.
            auto_export: Whether to automatically export Chrome Trace on exit.
        """
        self.name = name
        self.output_dir = Path(output_dir)
        self.config = config or ProfilerConfig()
        self.auto_export = auto_export
        self._profiler: _profile_context_manager | None = None
        self.trace_path: Path | None = None

    def __enter__(self) -> Self:
        """Start profiling."""
        self._profiler = profile(
            activities=self.config.activities,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_flops=self.config.with_flops,
            with_modules=self.config.with_modules,
        )
        self._profiler.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Stop profiling and optionally export results."""
        if self._profiler is not None:
            self._profiler.__exit__(exc_type, exc_val, exc_tb)
            if self.auto_export and exc_type is None:
                self.export_chrome_trace()

    def export_chrome_trace(self, path: str | Path | None = None) -> Path:
        """Export profiling data to Chrome Trace format.

        The exported JSON file can be viewed in Chrome DevTools:
        1. Open Chrome and navigate to chrome://tracing
        2. Click "Load" and select the generated JSON file
        3. Use WASD keys to navigate the flame graph

        Args:
            path: Output path for the trace file. If None, generates a
                timestamped filename in output_dir.

        Returns:
            Path to the exported trace file.

        Raises:
            RuntimeError: If profiler hasn't been run yet.
        """
        if self._profiler is None:
            raise RuntimeError("Profiler has not been run. Use within a 'with' block.")

        if path is None:
            timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}_trace.json"
            path = self.output_dir / filename

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._profiler.export_chrome_trace(str(path))
        self.trace_path = path
        return path

    def export_stacks(self, path: str | Path | None = None) -> Path:
        """Export profiling data as stacks for flamegraph.pl or speedscope.

        This format can be used with:
        - flamegraph.pl: https://github.com/brendangregg/FlameGraph
        - speedscope: https://www.speedscope.app/

        Args:
            path: Output path for the stacks file. If None, generates a
                timestamped filename in output_dir.

        Returns:
            Path to the exported stacks file.

        Raises:
            RuntimeError: If profiler hasn't been run yet or with_stack was False.
        """
        if self._profiler is None:
            raise RuntimeError("Profiler has not been run. Use within a 'with' block.")

        if not self.config.with_stack:
            raise RuntimeError("Stack traces not recorded. Set with_stack=True.")

        if path is None:
            timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}_stacks.txt"
            path = self.output_dir / filename

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._profiler.export_stacks(str(path), metric="self_cpu_time_total")
        return path

    def key_averages(
        self,
        *,
        group_by_input_shape: bool = False,
        group_by_stack_n: int = 0,
    ) -> Any:
        """Get key averages from profiler results.

        Args:
            group_by_input_shape: Group operators by input shapes.
            group_by_stack_n: Group by top N stack frames.

        Returns:
            EventList with aggregated profiler events.

        Raises:
            RuntimeError: If profiler hasn't been run yet.
        """
        if self._profiler is None:
            raise RuntimeError("Profiler has not been run. Use within a 'with' block.")
        return self._profiler.key_averages(
            group_by_input_shape=group_by_input_shape,
            group_by_stack_n=group_by_stack_n,
        )

    def print_summary(
        self,
        sort_by: str = "cuda_time_total",
        row_limit: int | None = None,
    ) -> None:
        """Print a summary table of profiling results.

        Args:
            sort_by: Column to sort by. Options include:
                - "cpu_time_total": Total CPU time
                - "cuda_time_total": Total CUDA time
                - "self_cpu_time_total": Self CPU time (excluding children)
                - "self_cuda_time_total": Self CUDA time
                - "cpu_memory_usage": CPU memory allocated
                - "cuda_memory_usage": CUDA memory allocated
            row_limit: Number of rows to display. Uses config default if None.

        Raises:
            RuntimeError: If profiler hasn't been run yet.
        """
        if self._profiler is None:
            raise RuntimeError("Profiler has not been run. Use within a 'with' block.")

        row_limit = row_limit or self.config.row_limit
        print(self.key_averages().table(sort_by=sort_by, row_limit=row_limit))  # noqa: T201


def profile_run[T](
    fn: Callable[..., T],
    *args: Any,
    profile_name: str = "torch_sim_profile",
    output_dir: str | Path = ".",
    config: ProfilerConfig | None = None,
    print_summary: bool = True,
    **kwargs: Any,
) -> tuple[T, Profiler]:
    """Profile a function call and export results.

    Convenience function that wraps any callable with profiling.

    Example::

        from torch_sim.profiling import profile_run

        result, prof = profile_run(
            ts.optimize,
            system, model,
            optimizer="fire",
            max_steps=100,
            profile_name="optimization_profile"
        )

        # Print detailed stats
        prof.print_summary(sort_by="cuda_time_total")

    Args:
        fn: The function to profile.
        *args: Positional arguments to pass to fn.
        profile_name: Name for the profiling session.
        output_dir: Directory for output files.
        config: ProfilerConfig with detailed settings.
        print_summary: Whether to print summary after profiling.
        **kwargs: Keyword arguments to pass to fn.

    Returns:
        Tuple of (function result, Profiler instance).
    """
    profiler = Profiler(name=profile_name, output_dir=output_dir, config=config)

    with profiler:
        result = fn(*args, **kwargs)

    if print_summary:
        profiler.print_summary()

    return result, profiler


@contextmanager
def profiling_section(name: str) -> Generator[None, None, None]:
    """Context manager to mark a section of code for profiling.

    Use this within a Profiler context to create labeled regions in the
    flame graph.

    Example::

        with Profiler("my_simulation") as prof:
            with profiling_section("initialization"):
                state = ts.fire_init(state, model)

            with profiling_section("optimization_loop"):
                for step in range(100):
                    state = ts.fire_step(state, model)

    Args:
        name: Label for this section in the profile.

    Yields:
        None
    """
    with record_function(name):
        yield
