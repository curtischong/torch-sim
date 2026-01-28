"""Script to run TorchSim static on MgO rocksalt structures, time runs, and plot scaling.

To run:

uv sync --extra mace --extra test
uv run 0.static.py

"""

# pyright: basic

import time
import typing
import warnings

from torch.profiler import ProfilerActivity, profile, record_function

warnings.filterwarnings(
    "ignore",
    message="The TorchScript type system doesn't support instance-level annotations on empty non-base types",
    category=UserWarning,
    module="torch.jit._check",
)

import plotly.graph_objects as go
import torch
import torch_sim as ts
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp

from pymatgen.io.ase import AseAtomsAdaptor
from torch_sim.models.mace import MaceModel, MaceUrls

MEMORY_SCALES_WITH = "n_atoms_x_density"
MAX_MEMORY_SCALER = 400_000


def make_speed_figure(
    n_structures: list[int],
    times: list[float],
    title: str,
    y_title: str = "Time (s)",
) -> go.Figure:
    """Build one speed-plot figure (single series)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=n_structures,
            y=times,
            mode="lines+markers",
            marker={"size": 10},
            line={"width": 2},
            name=title,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="n_structures",
        yaxis_title=y_title,
    )
    return fig


# def load_mace_mpa_model(device: torch.device) -> tuple[torch.nn.Module, str, int]:
#     """Load MACE-MPA model via mace-mp and wrap for torch_sim (no atlas)."""
#     loaded_model = mace_mp(
#         model=MaceUrls.mace_mpa_medium,
#         return_raw_model=True,
#         default_dtype="float64",
#         device=str(device),
#     )

#     model = MaceModel(
#         model=typing.cast(torch.nn.Module, loaded_model),
#         device=device,
#         compute_forces=True,
#         compute_stress=True,
#         dtype=torch.float64,
#         enable_cueq=False,
#     )

#     return model, MEMORY_SCALES_WITH, MAX_MEMORY_SCALER

def load_mace_mpa_model(device: torch.device) -> tuple[torch.nn.Module, str, int]:
    """
    Load MACE-MPA model and apply AOT (Ahead-of-Time) compilation 
    to prevent on-the-fly overhead during timing.
    """
    # 1. Load the raw model
    loaded_model = mace_mp(
        model=MaceUrls.mace_mpa_medium,
        return_raw_model=True,
        default_dtype="float64",
        device=str(device),
    )
    raw_model = typing.cast(torch.nn.Module, loaded_model)

    # 2. Pre-compile the model
    # 'reduce-overhead' is ideal for static-ish workloads as it uses CUDA Graphs.
    # 'dynamic=True' ensures that changing n_structures doesn't trigger a re-compile.
    print("Pre-compiling MACE model with torch.compile...")
    compiled_model = torch.compile(
        raw_model, 
        mode="reduce-overhead", 
        dynamic=True
    )

    # 3. Wrap in TorchSim model container
    model = MaceModel(
        model=compiled_model,
        device=device,
        compute_forces=True,
        compute_stress=True,
        dtype=torch.float64,
        enable_cueq=False,
    )

    return model, MEMORY_SCALES_WITH, MAX_MEMORY_SCALER


def load_ase_mace_calculator(device: torch.device) -> typing.Any:
    """Load MACE-MPA as an ASE calculator (same model as TorchSim)."""
    return mace_mp(
        model=MaceUrls.mace_mpa_medium,
        default_dtype="float64",
        device=str(device),
    )


def ts_warmup(
    base_structure: typing.Any,
    model: typing.Any,
    memory_scales_with: str,
    max_memory_scaler: int,
    n_warmup_list: list[int],
    nskip: int,
) -> None:
    """Warm up model with largest batch so CUDA/JIT warmup does not affect timings."""
    for n_warmup in n_warmup_list:
        structures = [base_structure] * n_warmup
        state = ts.initialize_state(structures, model.device, model.dtype)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
        batcher = ts.BinningAutoBatcher(  # pyright: ignore[reportAttributeAccessIssue]
            model=model,  # pyright: ignore[reportArgumentType]
            max_memory_scaler=max_memory_scaler,
            memory_scales_with=memory_scales_with,  # pyright: ignore[reportArgumentType]
        )
        batcher.load_states(state)
        for _ in range(nskip):
            ts.static(  # pyright: ignore[reportAttributeAccessIssue]
                system=structures,
                model=model,  # pyright: ignore[reportArgumentType]
                autobatcher=batcher,
            )
    print("Warmup done.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, memory_scales_with, max_memory_scaler = load_mace_mpa_model(device)
    ase_calc = load_ase_mace_calculator(device)

    # MgO rocksalt structure
    mgo_ase = bulk(name="MgO", crystalstructure="rocksalt", a=4.21, cubic=True)
    mgo_struct = AseAtomsAdaptor.get_structure(atoms=mgo_ase)  # pyright: ignore[reportArgumentType]
    base_structure = mgo_struct
    nskip = 5

    # Statistics sizes
    iterations = 1
    n_structures_list: list[int] = [1, 100, 200, 300, 400, 500, 1000, 1500]

    # Warmup all batch configurations -> this ensures that the timings below are accurate within TorchSim
    ts_warmup(
        base_structure=base_structure,
        model=model,
        memory_scales_with=memory_scales_with,
        max_memory_scaler=max_memory_scaler,
        n_warmup_list=n_structures_list,
        nskip=nskip,
    )

    # Run the same analysis multiple times with PyTorch profiler
    static_times: list[float] = []
    ase_times: list[float] = []
    batch_info_list: list[dict[str, typing.Any]] = []

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(iterations):
            print("Iteration", _ + 1)
            for n in n_structures_list:
                with record_function(f"n_structures={n}"):
                    structures = [base_structure] * n

                    with record_function("initialize_state"):
                        state = ts.initialize_state(structures, model.device, model.dtype)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

                    with record_function("create_batcher"):
                        batcher = ts.BinningAutoBatcher(  # pyright: ignore[reportAttributeAccessIssue]
                            model=model,  # pyright: ignore[reportArgumentType]
                            max_memory_scaler=max_memory_scaler,
                            memory_scales_with=memory_scales_with,  # pyright: ignore[reportArgumentType]
                        )

                    with record_function("load_states"):
                        batcher.load_states(state)

                    n_batches = len(batcher.index_bins)
                    batch_sizes = [len(b) for b in batcher.index_bins]
                    batch_info_list.append({"n": n, "n_batches": n_batches, "batch_sizes": batch_sizes})

                    t0 = time.perf_counter()
                    with record_function("ts.static"):
                        ts.static(  # pyright: ignore[reportAttributeAccessIssue]
                            system=structures,
                            model=model,  # pyright: ignore[reportArgumentType]
                            autobatcher=batcher,
                        )
                    elapsed = time.perf_counter() - t0
                    static_times.append(elapsed)

                    # ASE static: one calculator, n copies of atoms, sequential single-point runs
                    with record_function("ase_static"):
                        ase_atoms_list = [mgo_ase.copy() for _ in range(n)]
                        for at in ase_atoms_list:
                            at.calc = ase_calc
                        t0_ase = time.perf_counter()
                        for at in ase_atoms_list:
                            at.get_potential_energy()
                            at.get_forces()
                        elapsed_ase = time.perf_counter() - t0_ase
                    ase_times.append(elapsed_ase)

                    print(
                        f"n={n} n_batches={n_batches} batch_sizes={batch_sizes} "
                        f"static_time={elapsed:.6f}s ase_static_time={elapsed_ase:.6f}s"
                    )

    # # Print profiler results
    # print("\n" + "=" * 80)
    # print("PROFILER RESULTS - Sorted by CPU time")
    # print("=" * 80)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    # if torch.cuda.is_available():
    #     print("\n" + "=" * 80)
    #     print("PROFILER RESULTS - Sorted by CUDA time")
    #     print("=" * 80)
    #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # Export Chrome trace for detailed visualization
    prof.export_chrome_trace("profiler_trace2.json")
    print("\nProfiler trace exported to profiler_trace2.json")

    # # Plot ASE and TorchSim curves for the latest iteration
    # n_per_iter = len(n_structures_list)
    # latest_ts = static_times[-n_per_iter:]
    # latest_ase = ase_times[-n_per_iter:]
    # fig = go.Figure()
    # fig.add_trace(
    #     go.Scatter(
    #         x=n_structures_list,
    #         y=latest_ts,
    #         mode="lines+markers",
    #         marker={"size": 10},
    #         line={"width": 2},
    #         name="TorchSim",
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=n_structures_list,
    #         y=latest_ase,
    #         mode="lines+markers",
    #         marker={"size": 10},
    #         line={"width": 2},
    #         name="ASE",
    #     )
    # )
    # fig.update_layout(
    #     title=f"Static timing — latest iteration (iteration {iterations})",
    #     xaxis_title="n_structures",
    #     yaxis_title="Time (s)",
    #     legend={"x": 0.01, "y": 0.99},
    # )
    # fig.write_html("static_timing.html")