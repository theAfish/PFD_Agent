# Collect results — Python snippet

Parse multiple calculation directories and consolidate into a single extxyz file.

```python
import json
import numpy as np
from pathlib import Path
from ase.io import write as ase_write
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.ase import AseAtomsAdaptor

# ── inputs ──────────────────────────────────────────────────────────
CALC_DIRS   = ["<calc_dir1>", "<calc_dir2>"]
OUTPUT_FILE = "vasp_collected.extxyz"

# ── execution ────────────────────────────────────────────────────────
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
