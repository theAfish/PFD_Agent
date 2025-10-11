from typing import Optional, Union, Literal, Dict, Any, List, Tuple
import os
from pathlib import Path
from ase.io import read, write
from abacustest.lib_model.model_013_inputs import PrepInput
from abacustest.lib_model.comm import check_abacus_inputs
from abacustest.lib_prepare.abacus import AbacusStru, ReadInput, WriteInput
from pfd_agent_tool.modules.util.common import generate_work_path
from pfd_agent_tool.init_mcp import mcp
from pfd_agent_tool.modules.util.ase2xyz import dpdata2ase_single
import dpdata
from abacusagent.modules.util.comm import generate_work_path, link_abacusjob, run_abacus, collect_metrics
from abacusagent.modules.submodules.abacus import abacus_modify_input as _abacus_modify_input

from abacusagent.modules.abacus import abacus_modify_input, abacus_modify_stru                                                   
import logging
import traceback

@mcp.tool()
def abacus_prepare_batch(
    structure_path: Path,
    job_type: Literal["scf", "relax", "cell-relax", "md"] = "scf",
    lcao: bool = True,
    nspin: Literal[1, 2, 4] = 1,
    soc: bool = False,
    dftu: bool = False,
    dftu_param: Optional[Union[Dict[str, Union[float, Tuple[Literal["p", "d", "f"], float]]], Literal['auto']]] = None,
    init_mag: Optional[Dict[str, float]] = None,
    afm: bool = False,
    extra_input: Optional[Dict[str, Any]] = None,
    frames: Optional[List[int]] = None  # select specific frame indices, default: all
) -> Dict[str, Any]:
    """
    Prepare mandatory input files for ABACUS calculation from a file that contain a list of structures.
    This function does not perform any actual calculation, but is necessary to use this function
    to prepare a list of directories containing necessary input files for ABACUS calculations in batch mode. 
    If user provides ABACUS input files by him/herself, this function should not be used.
    
    Args:
        stru_file (Path): Structure file in cif, poscar, or abacus/stru format.
        stru_type (Literal["cif", "poscar", "abacus/stru"] = "cif"): Type of structure file, can be 'cif', 'poscar', or 'abacus/stru'. 'cif' is the default. 'poscar' is the VASP POSCAR format. 'abacus/stru' is the ABACUS structure format.
        job_type (Literal["scf", "relax", "cell-relax", "md"] = "scf"): The type of job to be performed, can be:
            'scf': Self-consistent field calculation, which is the default. 
            'relax': Geometry relaxation calculation, which will relax the atomic position to the minimum energy configuration.
            'cell-relax': Cell relaxation calculation, which will relax the cell parameters and atomic positions to the minimum energy configuration.
            'md': Molecular dynamics calculation, which will perform molecular dynamics simulation.
        lcao (bool): Whether to use LCAO basis set, default is True.
        nspin (int): The number of spins, can be 1 (no spin), 2 (spin polarized), or 4 (non-collinear spin). Default is 1.
        soc (bool): Whether to use spin-orbit coupling, if True, nspin should be 4.
        dftu (bool): Whether to use DFT+U, default is False.
        dftu_param (dict): The DFT+U parameters, should be 'auto' or a dict
            If dft_param is set to 'auto', hubbard U parameters will be set to d-block and f-block elements automatically. For d-block elements, default U=4eV will
                be set to d orbital. For f-block elements, default U=6eV will be set to f orbital.
            If dft_param is a dict, the keys should be name of elements and the value has two choices:
                - A float number, which is the Hubbard U value of the element. The corrected orbital will be infered from the name of the element.
                - A list containing two elements: the corrected orbital (should be 'p', 'd' or 'f') and the Hubbard U value.
                For example, {"Fe": ["d", 4], "O": ["p", 1]} means applying DFT+U to Fe 3d orbital with U=4 eV and O 2p orbital with U=1 eV.
        init_mag ( dict or None): The initial magnetic moment for magnetic elements, should be a dict like {"Fe": 4, "Ti": 1}, where the key is the element symbol and the value is the initial magnetic moment.
        afm (bool): Whether to use antiferromagnetic calculation, default is False. If True, half of the magnetic elements will be set to negative initial magnetic moment.
        extra_input: Extra input parameters in the prepared INPUT file. 
    
    Returns:
        A dictionary containing the list of job paths.
        - 'abacus_inputs_dir_list': A list of the absolute paths to the generated ABACUS input directory, containing INPUT, STRU, pseudopotential and orbital files.
        - 'input_content_list': A list of the content of the generated INPUT file.
    Raises:
        FileNotFoundError: If the structure file or pseudopotential path does not exist.
        ValueError: If LCAO basis set is selected but no orbital library path is provided.
        RuntimeError: If there is an error preparing input files.
    """
    try:
        structure_path = Path(structure_path).absolute()
        if not structure_path.is_file():
            raise FileNotFoundError(f"Structure file not found: {structure_path}")
        
        # Check if the pseudopotential path exists
        if soc:
            pp_path = os.environ.get("ABACUS_SOC_PP_PATH")
            orb_path = os.environ.get("ABACUS_SOC_ORB_PATH")
        else:
            pp_path = os.environ.get("ABACUS_PP_PATH")
            orb_path = os.environ.get("ABACUS_ORB_PATH")

        if not os.path.exists(pp_path):
            raise FileNotFoundError(f"Pseudopotential path {pp_path} does not exist.")

        if lcao and not os.path.exists(orb_path):
            raise FileNotFoundError(f"Orbital library path {orb_path} does not exist.")

        pwd = os.getcwd()
        work_path = Path(generate_work_path())
        os.chdir(work_path)

        results: List[Dict[str, Any]] = []
        cif_path_ls=[]
        cif_dir = Path("cif_dir")
        cif_dir.mkdir(exist_ok=True)
        for idx, atoms in enumerate(read(str(structure_path), index=':',format="extxyz")):
            if frames is not None and idx not in frames:
                continue
            cif_path = cif_dir / f"frame_{idx:04d}.cif"
            try:
                write(str(cif_path), atoms, format="cif")
            except Exception as e:
                results.append({"message": f"Failed to write CIF for frame {idx}: {e}", "frame_index": idx})
                continue
            cif_path_ls.append(str(cif_path))
            #logging.INFO(f"Written CIF for frame {idx} to {cif_path}")                                                             
        try:
            extra_input_file = None
            if extra_input is not None:
                if 'out_chg' not in extra_input.keys():
                    extra_input['out_chg'] = -1
                # write extra input to the input file
                extra_input_file = Path("INPUT.tmp").absolute()
                WriteInput(extra_input, extra_input_file)
            
            # return a list of abacus input job dir
            _, abacus_inputs_dir_ls = PrepInput(
                files=cif_path_ls,
                filetype="cif",
                jobtype=job_type,
                pp_path=pp_path,
                orb_path=orb_path,
                input_file=extra_input_file,
                lcao=lcao,
                nspin=nspin,
                soc=soc,
                dftu=dftu,
                dftu_param=dftu_param,
                init_mag=init_mag,
                afm=afm,
                copy_pp_orb=True
            ).run() 
        except Exception as e:
            os.chdir(pwd)
            raise RuntimeError(f"Error preparing input files: {e}")

        if len(abacus_inputs_dir_ls) == 0:
            os.chdir(pwd)
            raise RuntimeError("No job path returned from PrepInput.")

        input_content_ls = [ReadInput(os.path.join(abacus_inputs_dir, "INPUT")) for abacus_inputs_dir in abacus_inputs_dir_ls]
        abacus_inputs_dir_ls = [Path(abacus_inputs_dir).absolute() for abacus_inputs_dir in abacus_inputs_dir_ls]
        os.chdir(pwd)
        
        for abacus_inputs_dir in abacus_inputs_dir_ls:
            is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
            if not is_valid:
                raise RuntimeError(f"Invalid ABACUS input files at {abacus_inputs_dir}: {msg}")
        return {"abacus_inputs_dir_list": abacus_inputs_dir_ls,
                "input_content_list": input_content_ls}
    except Exception as e:
        
        return {"message": f"Batch prepare failed: {e}",
                "traceback": traceback.format_exc(),}

@mcp.tool()
def abacus_calculation_scf_batch(
    abacus_inputs_dir_ls: Union[List[Path], Path],
) -> Dict[str, Any]:
    """
    Run ABACUS SCF calculation.

    Args:
        abacus_inputs_dir (Path): Path to the directory containing the ABACUS input files.
    Returns:
        A dictionary containing the path to output file of ABACUS calculation, and a dictionary containing whether the SCF calculation
        finished normally, the SCF is converged or not, the converged SCF energy and total time used.
    """
    try:
        if isinstance(abacus_inputs_dir_ls, Path):
            abacus_inputs_dir_ls = [abacus_inputs_dir_ls]
            
        # base work directory
        work_path_base = Path(generate_work_path()).absolute()
        work_path_ls=[]
        for abacus_inputs_dir in abacus_inputs_dir_ls:
            is_valid, msg = check_abacus_inputs(abacus_inputs_dir)
            if not is_valid:
                raise RuntimeError(f"Invalid ABACUS input files: {msg}")
            _abacus_modify_input(abacus_inputs_dir,extra_input={'cal_force':True})
            work_path = work_path_base / abacus_inputs_dir.name
            work_path.mkdir(parents=True, exist_ok=True)
            link_abacusjob(src=abacus_inputs_dir, dst=work_path, copy_files=['INPUT', 'STRU'])
            input_params = ReadInput(os.path.join(work_path, "INPUT"))

            input_params['calculation'] = 'scf'
            WriteInput(input_params, os.path.join(work_path, "INPUT"))
            work_path_ls.append(work_path)
        
        # running abacus calculation
        run_abacus(work_path_ls)

        return_dict = {'scf_work_dir_list': [Path(work_path).absolute() for work_path in work_path_ls],}
        
        return_dict.update(
            {'scf_work_metrics_list':[collect_metrics(work_path,
                                                      metrics_names=['normal_end', 'converge', 'energy', 'total_time']) for work_path in work_path_ls]}
        )

        return return_dict
    except Exception as e:
        return {"message": f"Performing SCF calculation failed: {e}"}
    
@mcp.tool()
def collect_abacus_scf_results_batch(
    scf_work_dir_ls: Union[List[Path], Path],
) -> Dict[str, Any]:
    """
    Collect results from ABACUS SCF calculation.

    Args:
        scf_work_dir_ls (List[Path]): A list of path to the directories containing the ABACUS SCF calculation output files.
    Returns:
        A dictionary containing the path to output file of ABACUS calculation in extxyz format. The extxyz file contains the atomic structure and the total energy, atomic forces, etc., from the SCF calculation.
    """
    try:
        if isinstance(scf_work_dir_ls, Path):
            scf_work_dir_ls = [scf_work_dir_ls]
        atoms_ls=[]
        for scf_work_dir in scf_work_dir_ls:
            system=dpdata.LabeledSystem(str(scf_work_dir.absolute()),fmt='abacus/scf') 
            atoms=dpdata2ase_single(system)
            atoms_ls.append(atoms)
        work_path = Path(generate_work_path()).absolute()
        work_path.mkdir(parents=True, exist_ok=True)
        scf_result = work_path / "scf_result.extxyz"
        write(scf_result, atoms_ls, format="extxyz")
        return {"scf_result": scf_result}
    except Exception as e:
        return {"message": f"Collecting SCF results failed: {e}","traceback": traceback.format_exc()}