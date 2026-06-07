# SCF — Python snippet

```python
import json, os
from datetime import datetime
from pathlib import Path

from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.sets import MPStaticSet

def _work_dir(tag: str) -> Path:
    base = Path(os.environ.get("MATCLAW_SESSION_DIR", ".")) / "vasp"
    d = base / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    d.mkdir(parents=True, exist_ok=True)
    return d

# ── inputs ──────────────────────────────────────────────────────────
RELAX_DIR      = "<relax_calc_dir>"          # set to None if no prior relaxation
STRUCTURE_FILE = "<path.extxyz|path.vasp>"   # used only if RELAX_DIR is None
FRAMES         = None
SOC            = False
USER_INCAR     = {"NCORE": 4}  # Set NCORE ≈ sqrt(number of cores) for optimal performance

if SOC:
    USER_INCAR.update({
        "LSORBIT": True, "ISPIN": 2,
        "GGA": "PE", "GGA_COMPAT": False, "ISYM": 0,
        "LWAVE": True,   # WAVECAR needed for SOC NSCF
    })

# ── execution ────────────────────────────────────────────────────────
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

## Performance tuning

**NCORE**: Set to approximately **√(number of cores)** for optimal performance:
- 8 cores → NCORE=4
- 16 cores → NCORE=4
- 32 cores → NCORE=4
- 64 cores → NCORE=8

This balances band parallelization across cores and minimizes communication overhead.
