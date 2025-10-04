from typing import (
    Optional, 
    List, 
    Dict, 
    TypedDict,
)
import os
from pathlib import Path
import numpy as np
from ase.io import read, write
from ase.md.andersen import Andersen
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from ase import units
from .calculator import CalculatorWrapper
from pfd_agent_tool.init_mcp import mcp

import logging


class OptimizationResult(TypedDict):
    """Result structure for structure optimization"""
    optimized_structure: Path
    optimization_traj: Optional[Path]
    final_energy: float
    message: str


@mcp.tool()
def list_calculators() -> List[str]:
    """List all available calculator types. You should check the required input parameters for initializing each calculator type.
    Available calculator types are:
    - 'mattersim': MatterSim calculator for biomolecular systems. Requires `model_path` to the MatterSim model checkpoint.
    - 'deepmd' or 'dpa': DeepMD calculator for general materials. Requires `model_path` to the DeepMD model file (.pb or .pt). It might also need `head` parameter to specify the application domain. Only required for multi-head models.
    """
    return CalculatorWrapper.get_all_calculator()



@mcp.tool()
def optimize_structure( 
    input_structure: Path,
    model_style: str = "dpa",
    model_path: Optional[Path]= None,
    force_tolerance: float = 0.01, 
    max_iterations: int = 100, 
    relax_cell: bool = False,
    **kwargs
) -> OptimizationResult:
    """Optimize crystal structure using a Deep Potential (DP) model.

    Args:
        input_structure (Path): Path to the input structure file (e.g., CIF, POSCAR).
        model_path (Path): Path to the trained Deep Potential model directory.
            Default options are {'DPA2.4-7M': "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/13756/27666/store/upload/cd12300a-d3e6-4de9-9783-dd9899376cae/dpa-2.4-7M.pt", "DPA3.1-3M": "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/13756/27666/store/upload/18b8f35e-69f5-47de-92ef-af8ef2c13f54/DPA-3.1-3M.pt"}.
        force_tolerance (float, optional): Convergence threshold for atomic forces in eV/Å.
            Default is 0.01 eV/Å.
        max_iterations (int, optional): Maximum number of geometry optimization steps.
            Default is 100 steps.
        relax_cell (bool, optional): Whether to relax the unit cell shape and volume in addition to atomic positions.
            Default is False.
        **kwargs: Additional keyword arguments passed to the calculator initialization. For example, for DeepMD calculator, you might specify `head` to select the model head for multi-head models.


    Returns:
        dict: A dictionary containing optimization results:
            - optimized_structure (Path): Path to the final optimized structure file.
            - optimization_traj (Optional[Path]): Path to the optimization trajectory file, if available.
            - final_energy (float): Final potential energy after optimization in eV.
            - message (str): Status or error message describing the outcome.
    """
    try:
        base_name = input_structure.stem
        
        logging.info(f"Reading structure from: {input_structure}")
        atoms = read(str(input_structure))
       
        # Setup calculator
        calc=CalculatorWrapper.get_calculator(model_style)
        calc=calc().create(model_path=model_path, **kwargs)
        atoms.calc = calc

        traj_file = f"{base_name}_optimization_traj.extxyz"  
        if Path(traj_file).exists():
            logging.warning(f"Overwriting existing trajectory file: {traj_file}")
            Path(traj_file).unlink()

        logging.info("Starting structure optimization...")

        if relax_cell:
            logging.info("Using cell relaxation (ExpCellFilter)...")
            ecf = ExpCellFilter(atoms)
            optimizer = BFGS(ecf, trajectory=traj_file)
            optimizer.run(fmax=force_tolerance, steps=max_iterations)
        else:
            optimizer = BFGS(atoms, trajectory=traj_file)
            optimizer.run(fmax=force_tolerance, steps=max_iterations)

        output_file = f"{base_name}_optimized.cif"
        write(output_file, atoms)
        final_energy = float(atoms.get_potential_energy())

        logging.info(
            f"Optimization completed in {optimizer.nsteps} steps. "
            f"Final energy: {final_energy:.4f} eV"
        )

        return {
            "optimized_structure": Path(output_file),
            "optimization_traj": Path(traj_file),
            "final_energy": final_energy,
            "message": f"Successfully completed in {optimizer.nsteps} steps"
        }

    except Exception as e:
        logging.error(f"Optimization failed: {str(e)}", exc_info=True)
        return {
            "optimized_structure": Path(""),
            "optimization_traj": None, 
            "final_energy": -1.0,
            "message": f"Optimization failed: {str(e)}"
        }


def _log_progress(atoms, dyn):
    """Log simulation progress"""
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    temp = ekin / (1.5 * len(atoms) * units.kB)
    logging.info(f"Step: {dyn.nsteps:6d}, E_pot: {epot:.3f} eV, T: {temp:.2f} K")

def _run_md_stage(atoms, stage, save_interval_steps, traj_file, seed, stage_id):
    """Run a single MD simulation stage"""
    temperature_K = stage.get('temperature_K', None)
    pressure = stage.get('pressure', None)
    mode = stage['mode']
    runtime_ps = stage['runtime_ps']
    timestep_ps = stage.get('timestep', 0.0005)  # default: 0.5 fs
    tau_t_ps = stage.get('tau_t', 0.01)         # default: 10 fs
    tau_p_ps = stage.get('tau_p', 0.1)          # default: 100 fs

    timestep_fs = timestep_ps * 1000  # convert to fs
    total_steps = int(runtime_ps * 1000 / timestep_fs)

    # Initialize velocities if first stage with temperature
    if stage_id == 1 and temperature_K is not None:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, 
                                rng=np.random.RandomState(seed))
        Stationary(atoms)
        ZeroRotation(atoms)

    # Choose ensemble
    if mode == 'NVT' or mode == 'NVT-NH':
        # Use NoseHooverChain for NVT by default
        dyn = NoseHooverChainNVT(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            tdamp=tau_t_ps * 1000 * units.fs
        )
    elif mode == 'NVT-Berendsen':
        dyn = NVTBerendsen(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            taut=tau_t_ps * 1000 * units.fs
        )
    elif mode == 'NVT-Andersen':
        dyn = Andersen(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            friction=1.0 / (tau_t_ps * 1000 * units.fs),
            rng=np.random.RandomState(seed)
        )
    elif mode == 'NVT-Langevin' or mode == 'Langevin':
        dyn = Langevin(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            friction=1.0 / (tau_t_ps * 1000 * units.fs),
            rng=np.random.RandomState(seed)
        )
    elif mode == 'NPT-aniso' or mode == 'NPT-tri':
        if mode == 'NPT-aniso':
            mask = np.eye(3, dtype=bool)
        elif mode == 'NPT-tri':
            mask = None
        else:
            raise ValueError(f"Unknown NPT mode: {mode}")

        if pressure is None:
            raise ValueError("Pressure must be specified for NPT simulations")

        dyn = NPT(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            externalstress=pressure * units.GPa,
            ttime=tau_t_ps * 1000 * units.fs,
            pfactor=tau_p_ps * 1000 * units.fs,
            mask=mask
        )
    elif mode == 'NVE':
        dyn = VelocityVerlet(
            atoms,
            timestep=timestep_fs * units.fs
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Prepare trajectory file
    os.makedirs(os.path.dirname(traj_file), exist_ok=True)
    if os.path.exists(traj_file):
        os.remove(traj_file)

    def _write_frame():
        """Write current frame to trajectory"""
        results = atoms.calc.results
        energy = results.get("energy", atoms.get_potential_energy())
        forces = results.get("forces", atoms.get_forces())
        stress = results.get("stress", atoms.get_stress(voigt=False))

        if np.isnan(energy).any() or np.isnan(forces).any() or np.isnan(stress).any():
            raise ValueError("NaN detected in simulation outputs. Aborting trajectory write.")

        new_atoms = atoms.copy()
        new_atoms.info["energy"] = energy
        new_atoms.arrays["force"] = forces
        if "occupancy" in atoms.info:
            del atoms.info["occupancy"]
        if "spacegroup" in atoms.info:
            del atoms.info["spacegroup"] 

        write(traj_file, new_atoms, format="extxyz", append=True)

    # Attach callbacks
    dyn.attach(_write_frame, interval=save_interval_steps)
    dyn.attach(lambda: _log_progress(atoms, dyn), interval=100)

    logging.info(f"[Stage {stage_id}] Starting {mode} simulation: T={temperature_K} K"
                + (f", P={pressure} GPa" if pressure is not None else "")
                + f", steps={total_steps}, dt={timestep_ps} ps")

    # Run simulation
    dyn.run(total_steps)
    logging.info(f"[Stage {stage_id}] Finished simulation. Trajectory saved to: {traj_file}\n")

    return atoms

def _run_md_pipeline(atoms, stages, save_interval_steps=100, traj_prefix='traj', seed=None):
    """Run multiple MD stages sequentially"""
    for i, stage in enumerate(stages):
        mode = stage['mode']
        T = stage.get('temperature_K', 'NA')
        P = stage.get('pressure', 'NA')

        tag = f"stage{i+1}_{mode}_{T}K"
        if P != 'NA':
            tag += f"_{P}GPa"
        traj_file = os.path.join("trajs_files", f"{traj_prefix}_{tag}.extxyz")

        atoms = _run_md_stage(
            atoms=atoms,
            stage=stage,
            save_interval_steps=save_interval_steps,
            traj_file=traj_file,
            seed=seed,
            stage_id=i + 1
        )

    return atoms

@mcp.tool()
def run_molecular_dynamics(
    initial_structure: Path,
    stages: List[Dict],
    model_style: str = "dpa",
    model_path: Optional[Path]= None,
    save_interval_steps: int = 100,
    traj_prefix: str = 'traj',
    seed: Optional[int] = 42,
    **kwargs
    #head: Optional[str] = "Omat24",
) -> Dict:
    """
    [Modified from AI4S-agent-tools/servers/DPACalculator] Run a multi-stage molecular dynamics simulation using Deep Potential. 

    This tool performs molecular dynamics simulations with different ensembles (NVT, NPT, NVE)
    in sequence, using the ASE framework with the Deep Potential calculator.

    Args:
        initial_structure (Path): Input atomic structure file (supports .xyz, .cif, etc.)
        model_path (Path): Path to the Deep Potential model file (.pt or .pb)
            Default options are {'DPA2.4-7M': "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/13756/27666/store/upload/cd12300a-d3e6-4de9-9783-dd9899376cae/dpa-2.4-7M.pt", "DPA3.1-3M": "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/13756/27666/store/upload/18b8f35e-69f5-47de-92ef-af8ef2c13f54/DPA-3.1-3M.pt"}.
        stages (List[Dict]): List of simulation stages. Each dictionary can contain:
            - mode (str): Simulation ensemble type. One of:
                * "NVT" or "NVT-NH"- NVT ensemble (constant Particle Number, Volume, Temperature), with Nosé-Hoover (NH) chain thermostat
                * "NVT-Berendsen"- NVT ensemble with Berendsen thermostat. For quick thermalization
                * 'NVT-Andersen- NVT ensemble with Andersen thermostat. For quick thermalization (not rigorous NVT)
                * "NVT-Langevin" or "Langevin"- Langevin dynamics. For biomolecules or implicit solvent systems.
                * "NPT-aniso" - constant Number, Pressure (anisotropic), Temperature
                * "NPT-tri" - constant Number, Pressure (tri-axial), Temperature
                * "NVE" - constant Number, Volume, Energy (no thermostat/barostat, or microcanonical)
            - runtime_ps (float): Simulation duration in picoseconds.
            - temperature_K (float, optional): Temperature in Kelvin (required for NVT/NPT).
            - pressure (float, optional): Pressure in GPa (required for NPT).
            - timestep_ps (float, optional): Time step in picoseconds (default: 0.0005 ps = 0.5 fs).
            - tau_t_ps (float, optional): Temperature coupling time in picoseconds (default: 0.01 ps).
            - tau_p_ps (float, optional): Pressure coupling time in picoseconds (default: 0.1 ps).
        save_interval_steps (int): Interval (in MD steps) to save trajectory frames (default: 100).
        traj_prefix (str): Prefix for trajectory output files (default: 'traj').
        seed (int, optional): Random seed for initializing velocities (default: 42).
        **kwargs: Additional keyword arguments passed to the calculator initialization. For example, for DeepMD calculator, you might specify `head` to select the model head for multi-head models.
            - head (str, optional): Model head to use for multi-head DeepMD models (e.g., "Omat24", "Omat31"). Default is None.

    Returns: A dictionary containing:
            - final_structure (Path): Final atomic structure after all stages.
            - trajectory_dir (Path): The path of output directory of trajectory files generated.
            - log_file (Path): Path to the log file containing simulation output.

    Examples:
        >>> stages = [
        ...     {
        ...         "mode": "NVT",
        ...         "temperature_K": 300,
        ...         "runtime_ps": 5,
        ...         "timestep_ps": 0.0005,
        ...         "tau_t_ps": 0.01
        ...     },
        ...     {
        ...         "mode": "NPT-aniso",
        ...         "temperature_K": 300,
        ...         "pressure": 1.0,
        ...         "runtime_ps": 5,
        ...         "timestep_ps": 0.0005,
        ...         "tau_t_ps": 0.01,
        ...         "tau_p_ps": 0.1
        ...     },
        ...     {
        ...         "mode": "NVE",
        ...         "runtime_ps": 5,
        ...         "timestep_ps": 0.0005
        ...     }
        ... ]

        >>> result = run_molecular_dynamics(
        ...     initial_structure=Path("input.xyz"),
        ...     model_path=Path("model.pb"),
        ...     stages=stages,
        ...     save_interval_steps=50,
        ...     traj_prefix="cu_relax",
        ...     seed=42
        ... )
    """

    # Create output directories
    os.makedirs("trajs_files", exist_ok=True)
    log_file = Path("md_simulation.log")
    
    # Read initial structure
    atoms = read(initial_structure)
    
    # Setup calculator
    calc=CalculatorWrapper.get_calculator(model_style)
    calc=calc().create(model_path=model_path, **kwargs)
    atoms.calc = calc
    
    # Run MD pipeline
    final_atoms = _run_md_pipeline(
        atoms=atoms,
        stages=stages,
        save_interval_steps=save_interval_steps,
        traj_prefix=traj_prefix,
        seed=seed
    )
    
    # Save final structure
    final_structure = Path("final_structure.xyz")
    write(final_structure, final_atoms)
    
    # Collect trajectory files
    trajectory_dir = Path("trajs_files")
    
    result = {
        "final_structure": final_structure,
        "trajectory_dir": trajectory_dir,
        "log_file": log_file
    }
    
    for i, stage in enumerate(stages):
        mode = stage['mode']
        T = stage.get('temperature_K', 'NA')
        P = stage.get('pressure', 'NA')

        tag = f"stage{i+1}_{mode}_{T}K"
        if P != 'NA':
            tag += f"_{P}GPa"
        traj_file = os.path.join("trajs_files", f"{traj_prefix}_{tag}.extxyz")
        
        result[f"stage_{i+1}"] = Path(traj_file)
    
    return result


