# Relaxation — Python snippet

```python
import json, os
from datetime import datetime
from pathlib import Path

from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.sets import MPRelaxSet

def _work_dir(tag: str) -> Path:
    base = Path(os.environ.get("MATCLAW_SESSION_DIR", ".")) / "vasp"
    d = base / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    d.mkdir(parents=True, exist_ok=True)
    return d

# ── inputs ──────────────────────────────────────────────────────────
STRUCTURE_FILE = "<path.extxyz|path.vasp>"
FRAMES         = None   # list of int indices, or None for all (extxyz only)
USER_INCAR     = {
    "NCORE": 4,
    # "ENCUT": 600,
}

# ── execution ────────────────────────────────────────────────────────
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
