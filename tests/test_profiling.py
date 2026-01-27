"""Tests for the profiling module."""

import tempfile

import pytest
import torch

import torch_sim as ts
from torch_sim.models.interface import ModelInterface
from torch_sim.optimizers import Optimizer
from torch_sim.profiling import Profiler, ProfilerConfig, profile_run, profiling_section


DEVICE = torch.device("cpu")
DTYPE = torch.float64


@pytest.fixture
def ar_state(ar_supercell_sim_state: ts.SimState) -> ts.SimState:
    """Get argon state from conftest."""
    return ar_supercell_sim_state


class TestProfiler:
    """Tests for the Profiler class."""

    def test_basic_profiling(
        self, ar_state: ts.SimState, lj_model: ModelInterface
    ) -> None:
        """Test basic profiling context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with Profiler("test_profile", output_dir=tmpdir) as prof:
                # Run a simple operation
                _ = lj_model(ar_state)

            # Check that trace file was created
            assert prof.trace_path is not None
            assert prof.trace_path.exists()
            assert prof.trace_path.suffix == ".json"

    def test_profiler_without_auto_export(
        self, ar_state: ts.SimState, lj_model: ModelInterface
    ) -> None:
        """Test profiler with auto_export disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with Profiler("test_profile", output_dir=tmpdir, auto_export=False) as prof:
                _ = lj_model(ar_state)

            # Trace path should be None since we didn't auto-export
            assert prof.trace_path is None

            # Manual export should work
            trace_path = prof.export_chrome_trace()
            assert trace_path.exists()

    def test_key_averages(
        self, ar_state: ts.SimState, lj_model: ModelInterface
    ) -> None:
        """Test accessing key averages from profiler."""
        with Profiler("test_profile", auto_export=False) as prof:
            _ = lj_model(ar_state)

        averages = prof.key_averages()
        assert averages is not None
        # Should have at least some events
        assert len(averages) > 0

    def test_export_stacks(
        self, ar_state: ts.SimState, lj_model: ModelInterface
    ) -> None:
        """Test exporting stack traces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProfilerConfig(with_stack=True)
            with Profiler(
                "test_profile", output_dir=tmpdir, config=config, auto_export=False
            ) as prof:
                _ = lj_model(ar_state)

            stacks_path = prof.export_stacks()
            assert stacks_path.exists()
            assert stacks_path.suffix == ".txt"

    def test_custom_config(
        self, ar_state: ts.SimState, lj_model: ModelInterface
    ) -> None:
        """Test profiler with custom configuration."""
        config = ProfilerConfig(
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            row_limit=10,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with Profiler("test_profile", output_dir=tmpdir, config=config) as prof:
                _ = lj_model(ar_state)

            assert prof.trace_path is not None

    def test_profiler_not_run_error(self) -> None:
        """Test error when accessing results before running profiler."""
        prof = Profiler("test_profile")
        with pytest.raises(RuntimeError, match="Profiler has not been run"):
            prof.key_averages()

        with pytest.raises(RuntimeError, match="Profiler has not been run"):
            prof.export_chrome_trace()


class TestProfileRun:
    """Tests for the profile_run convenience function."""

    def test_profile_run_basic(
        self, ar_state: ts.SimState, lj_model: ModelInterface
    ) -> None:
        """Test basic usage of profile_run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result, prof = profile_run(
                lj_model,
                ar_state,
                profile_name="test_run",
                output_dir=tmpdir,
                print_summary=False,
            )

            # Check result is returned
            assert "energy" in result
            assert "forces" in result

            # Check profiler worked
            assert prof.trace_path is not None
            assert prof.trace_path.exists()


class TestProfilingSection:
    """Tests for the profiling_section context manager."""

    def test_profiling_section_labeling(
        self, ar_state: ts.SimState, lj_model: ModelInterface
    ) -> None:
        """Test that profiling sections create labeled regions."""
        with Profiler("test_sections", auto_export=False) as prof:
            with profiling_section("model_forward"):
                _ = lj_model(ar_state)

            with profiling_section("second_forward"):
                _ = lj_model(ar_state)

        # Check that sections appear in the profile
        averages = prof.key_averages()
        event_names = [event.key for event in averages]
        assert "model_forward" in event_names
        assert "second_forward" in event_names


class TestProfilerWithOptimization:
    """Test profiling with actual optimization loops."""

    def test_profile_fire_optimization(
        self, ar_state: ts.SimState, lj_model: ModelInterface
    ) -> None:
        """Test profiling a FIRE optimization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with Profiler("fire_optimization", output_dir=tmpdir) as prof:
                state = ts.fire_init(state=ar_state, model=lj_model, dt_start=0.005)
                for _step in range(5):
                    state = ts.fire_step(state=state, model=lj_model, dt_max=0.01)

            assert prof.trace_path is not None
            assert prof.trace_path.exists()

            # Verify we can read the profile data
            averages = prof.key_averages()
            assert len(averages) > 0

    def test_profile_optimize_runner(
        self, ar_state: ts.SimState, lj_model: ModelInterface
    ) -> None:
        """Test profiling the high-level optimize runner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with Profiler("optimize_runner", output_dir=tmpdir) as prof:
                _ = ts.optimize(
                    system=ar_state,
                    model=lj_model,
                    optimizer=Optimizer.fire,
                    max_steps=10,
                    pbar=False,
                )

            assert prof.trace_path is not None
            assert prof.trace_path.exists()
