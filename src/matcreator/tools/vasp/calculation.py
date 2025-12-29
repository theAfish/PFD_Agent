import os
import time
import subprocess
import shutil
from pymatgen.core import Element, Structure
from pymatgen.io.vasp import VaspInput, Vasprun, Kpoints, Poscar, Chgcar, Potcar, Outcar
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine
import math
import pathlib
from typing import Optional, Dict, Any, Union
import numpy as np
from .common import run_vasp
import traceback
from dpdispatcher import Machine, Resources, Task, Submission

def vasp_relaxation(calculation_id: str, work_dir: str, struct: Structure, 
                   kpoints: Kpoints, incar_dict: dict, potcar_map: Optional[Dict] = None) -> Dict[str, Any]:
    """
    提交VASP结构优化计算任务
    
    参数:
        calculation_id: 计算ID
        work_dir: 工作目录
        struct: 晶体结构
        kpoints: K点设置
        incar_dict: 额外的INCAR参数，会与默认设置合并。除非用户指定，不要擅自修改。
        
    返回:
        Dict包含success、error等信息
    """
    if potcar_map is None:
        potcar_map = {}
    try:
        Name = calculation_id
        calc_dir = os.path.abspath(f'{work_dir}/{Name}')
        calc_dir_1 = (f'tmp/vasp_server/{Name}')
        # 创建VASP输入文件
        # 手动获取元素列表，确保顺序与POSCAR一致
        poscar = Poscar(struct)
        unique_species = []
        for species in poscar.structure.species:
            species: Element
            if unique_species:
                if species.symbol != unique_species[-1]:
                    if species.symbol not in potcar_map:
                        potcar_map[species.symbol] = species.symbol
                    unique_species.append(species.symbol)
            else:
                if species.symbol not in potcar_map:
                    potcar_map[species.symbol] = species.symbol
                unique_species.append(species.symbol)
        potcar_symbols = []
        for symbol in unique_species:
            potcar_symbols.append(potcar_map[symbol])

        vasp_input = VaspInput(
            poscar=poscar,
            incar=incar_dict,
            kpoints=kpoints,
            potcar=Potcar(potcar_symbols)
        )
        
        # 准备结构优化目录
        vasp_input.write_input(calc_dir)
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Performing vasp calculation failed: {e}",
            "traceback": traceback.format_exc()}

    submit_type = os.environ.get("VASPAGENT_SUBMIT_TYPE", "local").lower()

    if submit_type == "local":
        run_vasp(calc_dir) 

    elif submit_type == "bohrium":
        task=Task(
            command="source /opt/intel/oneapi/setvars.sh && mpirun -n 32 vasp_std",
            # task_work_path=calc_dir_1,
            task_work_path=Name,
            forward_files=["POSCAR","INCAR","POTCAR","KPOINTS"],
            backward_files=["OSZICAR","CONTCAR","OUTCAR","vasprun.xml"]
        )
        return task
    
    else:
        raise ValueError("Invalid VASPAGENT_SUBMIT_TYPE. Must be 'local' or 'bohrium'.")


def vasp_scf(calculation_id: str, work_dir: str, struct: Structure, 
            kpoints: Kpoints, incar_dict: dict, chgcar_path: Optional[str] = None, 
            wavecar_path: Optional[str] = None, potcar_map: Optional[Dict] = None):
    """
    提交VASP自洽场计算任务
    
    参数:
        calculation_id: 计算ID
        work_dir: 工作目录
        struct: 晶体结构
        kpoints: K点设置
        incar_dict: 额外的INCAR参数，会与默认设置合并。除非用户指定，不要擅自修改。
        chgcar_path: CHGCAR文件路径
        wavecar_path: WAVECAR文件路径
        
    返回:
        Dict包含success、error等信息
    """
    if potcar_map is None:
        potcar_map = {}
    try:
        Name = calculation_id
        calc_dir = os.path.abspath(f'{work_dir}/{Name}')
        calc_dir_1 = (f'tmp/vasp_server/{Name}')
        # 创建VASP输入文件
        # 手动获取元素列表，确保顺序与POSCAR一致
        poscar = Poscar(struct)
        unique_species = []
        for species in poscar.structure.species:
            species: Element
            if unique_species:
                if species.symbol != unique_species[-1]:
                    if species.symbol not in potcar_map:
                        potcar_map[species.symbol] = species.symbol
                    unique_species.append(species.symbol)
            else:
                if species.symbol not in potcar_map:
                    potcar_map[species.symbol] = species.symbol
                unique_species.append(species.symbol)
        potcar_symbols = []
        for symbol in unique_species:
            potcar_symbols.append(potcar_map[symbol])

        vasp_input = VaspInput(
            poscar=poscar,
            incar=incar_dict,
            kpoints=kpoints,
            potcar=Potcar(potcar_symbols)
        )

        # 准备自洽场计算目录
        vasp_input.write_input(calc_dir)
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Performing vasp calculation failed: {e}",
            "traceback": traceback.format_exc()}


    submit_type = os.environ.get("VASPAGENT_SUBMIT_TYPE", "local").lower()

    if submit_type == "local":
        run_vasp(calc_dir) 

    elif submit_type == "bohrium":
        task=Task(
            command="source /opt/intel/oneapi/setvars.sh && mpirun -n 32 vasp_std",
            # task_work_path=calc_dir_1,
            task_work_path=Name,
            forward_files=["POSCAR","INCAR","POTCAR","KPOINTS"],
            backward_files=["OSZICAR","CONTCAR","OUTCAR","vasprun.xml"]
        )
        return task
    
    else:
        raise ValueError("Invalid VASPAGENT_SUBMIT_TYPE. Must be 'local' or 'bohrium'.")

