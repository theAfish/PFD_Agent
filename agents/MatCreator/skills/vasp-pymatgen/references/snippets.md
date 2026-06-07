# vasp-pymatgen Python snippets

Detailed Python code for all `vasp-pymatgen` skill commands. Load with:
```
load_skill_resource(skill_name="vasp-pymatgen", path="references/snippets.md")
```

---

## Common header

Include at the top of every preparation snippet.

```python
import json, os, shutil
from datetime import datetime
from pathlib import Path

from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure

def _work_dir(tag: str) -> Path:
    base = Path(os.environ.get("MATCLAW_SESSION_DIR", ".")) / "vasp"
    d = base / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    d.mkdir(parents=True, exist_ok=True)
    return d
```

---

## prepare_relaxation

```python
# ── inputs ──────────────────────────────────────────────────────────
STRUCTURE_FILE  = "<path.extxyz|path.vasp>"
FRAMES          = None   # list of int indices, or None for all (extxyz only)
USER_INCAR      = {
    "NCORE": 4,
    # "ENCUT": 600,
}

# ── execution ────────────────────────────────────────────────────────
from pymatgen.io.vasp.sets import MPRelaxSet

atoms_list = ase_read(STRUCTURE_FILE, index=":" if FRAMES is None else FRAMES)
if not isinstance(atoms_list, list):
    atoms_list = [atoms_list]

calc_dirs = []
for atoms in atoms_list:
    structure = AseAtomsAdaptor.get_structure(atoms)
    vis = MPRelaxSet(structure, user_incar_settings=USER_INCAR)
    d = _work_dir("relax")
    vis.write_input(str(d))
    calc_dirs.append(str(d.resolve()))

print(json.dumps({"status": "success", "calc_dir_list": calc_dirs}))
```

**MPRelaxSet key defaults:** IBRION=2, ISIF=3, NSW=99, ISMEAR=0, SIGMA=0.05, LWAVE=False, LCHARG=False. k-points: `reciprocal_density=64`.

---

## prepare_scf

```python
# ── inputs ──────────────────────────────────────────────────────────
RELAX_DIR      = "<relax_calc_dir>"   # set to None if no prior relaxation
STRUCTURE_FILE = "<path.extxyz|path.vasp>"  # used only if RELAX_DIR is None
FRAMES         = None
SOC            = False
USER_INCAR     = {"NCORE": 4}

if SOC:
    USER_INCAR.update({
        "LSORBIT": True, "ISPIN": 2,
        "GGA": "PE", "GGA_COMPAT": False, "ISYM": 0,
        "LWAVE": True,   # WAVECAR needed for SOC NSCF
    })

# ── execution ────────────────────────────────────────────────────────
from pymatgen.io.vasp.sets import MPStaticSet

calc_dirs = []
if RELAX_DIR is not None:
    vis = MPStaticSet.from_prev_calc(prev_calc_dir=RELAX_DIR, user_incar_settings=USER_INCAR)
    d = _work_dir("scf")
    vis.write_input(str(d))
    calc_dirs.append(str(d.resolve()))
else:
    atoms_list = ase_read(STRUCTURE_FILE, index=":" if FRAMES is None else FRAMES)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    for atoms in atoms_list:
        structure = AseAtomsAdaptor.get_structure(atoms)
        vis = MPStaticSet(structure, user_incar_settings=USER_INCAR)
        d = _work_dir("scf")
        vis.write_input(str(d))
        calc_dirs.append(str(d.resolve()))

print(json.dumps({"status": "success", "calc_dir_list": calc_dirs}))
```

**MPStaticSet key defaults:** NSW=0, IBRION=-1, LCHARG=True, LORBIT=11, LASPH=True. k-points: `reciprocal_density=100`.

> Include `CHGCAR` (and `WAVECAR` for SOC) in both `forward_files` and `backward_files` when submitting.

---

## prepare_nscf_kpath

```python
# ── inputs ──────────────────────────────────────────────────────────
SCF_DIRS   = ["<scf_dir1>", "<scf_dir2>"]
SOC        = False
USER_INCAR = {"NCORE": 4}   # add "NBANDS": N if more bands needed

if SOC:
    USER_INCAR.update({
        "LSORBIT": True, "ISPIN": 2,
        "GGA": "PE", "GGA_COMPAT": False, "ISYM": 0,
    })

# ── execution ────────────────────────────────────────────────────────
from pymatgen.io.vasp.sets import MPNonSCFSet

calc_dirs = []
for scf_dir in SCF_DIRS:
    vis = MPNonSCFSet.from_prev_calc(
        prev_calc_dir=scf_dir, mode="line", user_incar_settings=USER_INCAR,
    )
    d = _work_dir("nscf_kpath")
    vis.write_input(str(d))   # copies CHGCAR automatically
    if SOC and (Path(scf_dir) / "WAVECAR").exists():
        shutil.copy2(Path(scf_dir) / "WAVECAR", d / "WAVECAR")
    calc_dirs.append(str(d.resolve()))

print(json.dumps({"status": "success", "calc_dir_list": calc_dirs}))
```

**MPNonSCFSet line-mode defaults:** ICHARG=11, LCHARG=False, LWAVE=False, LORBIT=11. k-path from `HighSymmKpath` (64 points per segment).

---

## prepare_nscf_uniform

```python
# ── inputs ──────────────────────────────────────────────────────────
SCF_DIRS   = ["<scf_dir1>"]
SOC        = False
USER_INCAR = {"NCORE": 4, "NEDOS": 2000}

if SOC:
    USER_INCAR.update({
        "LSORBIT": True, "ISPIN": 2,
        "GGA": "PE", "GGA_COMPAT": False, "ISYM": 0,
    })

# ── execution ────────────────────────────────────────────────────────
from pymatgen.io.vasp.sets import MPNonSCFSet

calc_dirs = []
for scf_dir in SCF_DIRS:
    vis = MPNonSCFSet.from_prev_calc(
        prev_calc_dir=scf_dir, mode="uniform", user_incar_settings=USER_INCAR,
    )
    d = _work_dir("nscf_uniform")
    vis.write_input(str(d))
    if SOC and (Path(scf_dir) / "WAVECAR").exists():
        shutil.copy2(Path(scf_dir) / "WAVECAR", d / "WAVECAR")
    calc_dirs.append(str(d.resolve()))

print(json.dumps({"status": "success", "calc_dir_list": calc_dirs}))
```

**MPNonSCFSet uniform-mode defaults:** ICHARG=11, ISMEAR=-5 (tetrahedron), reciprocal_density=1000.

---

## read_results

```python
# ── inputs ──────────────────────────────────────────────────────────
CALC_TYPE = "<relaxation|scf|nscf>"
CALC_DIR  = "<calc_dir>"

# ── execution ────────────────────────────────────────────────────────
import json
from pathlib import Path
from pymatgen.io.vasp.outputs import Vasprun

d  = Path(CALC_DIR)
vr = Vasprun(str(d / "vasprun.xml"), parse_dos=False, parse_eigen=(CALC_TYPE == "nscf"))
result = {"status": "success", "calc_type": CALC_TYPE}

if CALC_TYPE == "relaxation":
    forces = vr.ionic_steps[-1]["forces"]
    result.update({
        "structure":    vr.final_structure.as_dict(),
        "total_energy": vr.final_energy,
        "max_force":    max(sum(f**2 for f in force)**0.5 for force in forces),
        "stress":       vr.ionic_steps[-1].get("stress"),
        "ionic_steps":  len(vr.ionic_steps),
    })
elif CALC_TYPE == "scf":
    gap, cbm, vbm, is_metal = vr.eigenvalue_band_properties
    result.update({
        "structure":    vr.final_structure.as_dict(),
        "total_energy": vr.final_energy,
        "efermi":       vr.efermi,
        "band_gap":     gap,
        "is_metal":     is_metal,
    })
elif CALC_TYPE == "nscf":
    bs = vr.get_band_structure(line_mode=True)
    gap, cbm, vbm, is_metal = vr.eigenvalue_band_properties
    result.update({
        "structure": vr.final_structure.as_dict(),
        "efermi":    vr.efermi,
        "band_gap":  gap,
        "cbm":       cbm,
        "vbm":       vbm,
        "is_metal":  is_metal,
        "band_structure_summary": {
            "n_kpoints": len(bs.kpoints),
            "n_bands":   bs.nb_bands,
            "labels":    {str(k): list(v) for k, v in bs.labels_dict.items()},
        },
    })

print(json.dumps(result))
```

---

## collect_results

```python
# ── inputs ──────────────────────────────────────────────────────────
CALC_DIRS   = ["<calc_dir1>", "<calc_dir2>"]
OUTPUT_FILE = "vasp_collected.extxyz"

# ── execution ────────────────────────────────────────────────────────
import json
import numpy as np
from pathlib import Path
from ase.io import write as ase_write
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.ase import AseAtomsAdaptor

frames = []
for calc_dir in [Path(p) for p in CALC_DIRS]:
    vr    = Vasprun(str(calc_dir / "vasprun.xml"), parse_dos=False, parse_eigen=False)
    atoms = AseAtomsAdaptor.get_atoms(vr.final_structure)
    atoms.info["energy"] = vr.final_energy
    atoms.arrays["forces"] = np.array(vr.ionic_steps[-1]["forces"])
    frames.append(atoms)

output = Path(CALC_DIRS[0]).parent / OUTPUT_FILE
ase_write(str(output), frames, format="extxyz")
print(json.dumps({"status": "success", "scf_result": str(output.resolve())}))
```
