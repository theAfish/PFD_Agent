---
name: vasp
description: Skills for VASP DFT calculations — input preparation, remote submission to Bohrium, and result collection.
tags: [vasp, dft, relaxation, scf, band-structure, dos, bohrium]
tools: [run_bash,run_python_file]
dependent_skills: []
---

# VASP DFT Skill

Two scripts work together to run VASP calculations:

| Script | Role |
|---|---|
| `vasp_tools.py` | Prepare input files; collect / read results |
| `bohrium_submit.py` | Submit prepared directories to Bohrium via dpdispatcher |

Both scripts live in the same directory alongside `config.yaml` and `.env`.
Every command prints JSON to stdout and exits 0 on success, 1 on error.

---

## Mandatory workflow sequence

1. **Obtain a structure** — if the user has not supplied one, generate it first.
2. **Prepare inputs** — run the appropriate `vasp_tools.py prepare_*` command.
3. **Submit to Bohrium** — pass the returned `calc_dir_list` to `bohrium_submit.py submit`.
4. **Collect / read results** — after the job finishes, run `collect_results` or `read_results`.

Run exactly **one property step at a time**. Do not chain relaxation + SCF in a single step.

---

## vasp_tools.py — Command reference

### Common flags (all prepare_* commands)

| Flag | Default | Description |
|---|---|---|
| `--config PATH` | `config.yaml` next to the script | Path to config.yaml |
| `--incar_tags JSON` | `{}` | Extra INCAR overrides, e.g. `'{"ENCUT": 600}'` |
| `--potcar_map JSON` | `{}` | Element → POTCAR label, e.g. `'{"Bi": "Bi_d"}'` |

All `prepare_*` commands return:
```json
{ "status": "success", "calc_dir_list": ["<abs_path>", ...] }
```

---

### prepare_relaxation

Prepare structural relaxation (IBRION=2, ISIF=3, NSW=200).

```bash
python vasp_tools.py prepare_relaxation \
    --structure <path.extxyz|path.vasp> \
    [--frames 0 1 2] \
    [--kpoints NX NY NZ] \
    [--incar_tags '{"ENCUT": 600}'] \
    [--potcar_map '{"Bi": "Bi_d"}']
```

- `--structure`: structure file to read. Supported formats:
  - `*.extxyz` — may contain multiple frames; all frames are processed by default.
  - `*.vasp` (POSCAR format) — always contains exactly one frame.
- `--frames`: integer indices to process (extxyz only); default is all frames.
- `--kpoints`: explicit Gamma-centred mesh. Default: auto KPPRA density 40.

---

### prepare_scf

Prepare a self-consistent field calculation (NSW=0, IBRION=-1).

```bash
python vasp_tools.py prepare_scf \
    --structure <path.extxyz|path.vasp> \
    [--frames 0 1 2] \
    [--kpoints NX NY NZ] \
    [--soc] \
    [--incar_tags '{"ENCUT": 600}']
```

- `--structure`: structure file to read. Supported formats:
  - `*.extxyz` — may contain multiple frames; all frames are processed by default.
  - `*.vasp` (POSCAR format) — always contains exactly one frame.
- `--frames`: integer indices to process (extxyz only); default is all frames.
- `--soc`: use `scf_soc` INCAR preset (ISPIN=2, LSORBIT=.TRUE.); default: `scf_nsoc`.
- Default k-mesh: auto KPPRA density 40.
- SCF outputs CHGCAR and WAVECAR when `LCHARG=true` / `LWAVE=true` (SOC preset only by default).

---

### prepare_nscf_kpath

Prepare a non-self-consistent band-structure calculation along a k-path.
Requires completed SCF directories that contain `CONTCAR` and `CHGCAR`.

```bash
python vasp_tools.py prepare_nscf_kpath \
    --scf_dirs <scf_dir1> [<scf_dir2> ...] \
    [--kpath GMKG] \
    [--n_kpoints 16] \
    [--soc] \
    [--incar_tags '{"NBANDS": 48}']
```

- `--kpath`: explicit path string (e.g. `GMKG`). Default: auto from pymatgen `HighSymmKpath`.
- `--n_kpoints`: points per segment (default 16).
- Copies `CHGCAR` (and `WAVECAR` if present) from the SCF directory.

---

### prepare_nscf_uniform

Prepare a non-self-consistent uniform-mesh calculation (for DOS).
Requires completed SCF directories.

```bash
python vasp_tools.py prepare_nscf_uniform \
    --scf_dirs <scf_dir1> [<scf_dir2> ...] \
    [--kpoints NX NY NZ] \
    [--soc] \
    [--incar_tags '{"NEDOS": 2000}']
```

- Default k-mesh: auto KPPRA density 100.
- Copies `CHGCAR` (and `WAVECAR` if present) from the SCF directory.

---

### collect_results

Parse `OUTCAR` files and write all frames into a single extxyz (via dpdata).

```bash
python vasp_tools.py collect_results \
    --dirs <calc_dir1> [<calc_dir2> ...]
```

Returns:
```json
{ "status": "success", "scf_result": "<abs_path_to_scf_result.extxyz>" }
```

---

### read_results

Read `vasprun.xml` / `OUTCAR` and return key scalar results as JSON.

```bash
python vasp_tools.py read_results \
    --calc_type <relaxation|scf|nscf> \
    --calc_dir  <calc_dir>
```

| calc_type | Returned fields |
|---|---|
| `relaxation` | `structure`, `total_energy`, `max_force`, `stress`, `ionic_steps` |
| `scf` | `structure`, `total_energy`, `efermi`, `band_gap`, `is_metal` |
| `nscf` | `structure`, `efermi`, `is_metal`, `band_gap`, `cbm`, `vbm` |

---

## bohrium_submit.py — Command reference

### Environment variables (`.env` or shell export)

| Variable | Description |
|---|---|
| `BOHRIUM_USERNAME` | Bohrium account email |
| `BOHRIUM_PASSWORD` | Bohrium account password |
| `BOHRIUM_PROJECT_ID` | Bohrium project ID (integer) |
| `BOHRIUM_VASP_MACHINE` | Machine type, e.g. `c32_m128_cpu` |
| `BOHRIUM_VASP_IMAGE` | Container image for VASP |

### submit

```bash
python bohrium_submit.py submit \
    --calc_dirs <dir1> [<dir2> ...] \
    --calc_type <relaxation|scf|nscf> \
    [--group_size 4] \
    [--vasp_command "source /opt/intel/oneapi/setvars.sh && mpirun -n 32 vasp_std"]
```

- All `--calc_dirs` **must share the same parent directory** (used as dpdispatcher `local_root`).
- `--group_size`: jobs per Bohrium submission group (default 4).
- `--vasp_command`: shell command executed on the remote (default shown above).

**File transfer per calc type:**

| calc_type | Forward (upload) | Backward (download) |
|---|---|---|
| `relaxation` | POSCAR INCAR POTCAR KPOINTS | OSZICAR CONTCAR OUTCAR vasprun.xml |
| `scf` | POSCAR INCAR POTCAR KPOINTS | OSZICAR CONTCAR OUTCAR vasprun.xml CHGCAR WAVECAR |
| `nscf` | POSCAR INCAR POTCAR KPOINTS CHGCAR WAVECAR | OSZICAR CONTCAR OUTCAR vasprun.xml |

Returns:
```json
{ "status": "success", "calc_type": "scf", "calc_dir_list": ["<abs_path>", ...] }
```

---

## End-to-end example: relaxation → SCF → band structure

```bash
# 1. Prepare relaxation
RELAX=$(python vasp_tools.py prepare_relaxation --structure Al.extxyz)
RELAX_DIRS=$(echo $RELAX | python -c "import sys,json; print(' '.join(json.load(sys.stdin)['calc_dir_list']))")

# 2. Submit relaxation to Bohrium
python bohrium_submit.py submit --calc_dirs $RELAX_DIRS --calc_type relaxation

# 3. Read relaxation results
python vasp_tools.py read_results --calc_type relaxation --calc_dir <relax_dir>

# 4. Prepare SCF from relaxed structure (CONTCAR → extxyz conversion needed, or pass CONTCAR directly)
SCF=$(python vasp_tools.py prepare_scf --structure Al_relaxed.extxyz)
SCF_DIRS=$(echo $SCF | python -c "import sys,json; print(' '.join(json.load(sys.stdin)['calc_dir_list']))")

# 5. Submit SCF
python bohrium_submit.py submit --calc_dirs $SCF_DIRS --calc_type scf

# 6. Prepare NSCF k-path from SCF output
NSCF=$(python vasp_tools.py prepare_nscf_kpath --scf_dirs $SCF_DIRS)
NSCF_DIRS=$(echo $NSCF | python -c "import sys,json; print(' '.join(json.load(sys.stdin)['calc_dir_list']))")

# 7. Submit NSCF
python bohrium_submit.py submit --calc_dirs $NSCF_DIRS --calc_type nscf

# 8. Read NSCF results (band gap, CBM/VBM)
python vasp_tools.py read_results --calc_type nscf --calc_dir <nscf_dir>
```

---

## Configuration file (config.yaml)

`config.yaml` lives in the same directory as `vasp_tools.py`. It controls:

- `work_dir` — where all calculation subdirectories are created (default `/tmp/vasp_server`).
- `VASP_default_INCAR` — one sub-key per preset (`relaxation`, `scf_nsoc`, `scf_soc`, `nscf_nsoc`, `nscf_soc`).

**When asked about default INCAR settings, read `config.yaml` directly** — do not guess from memory, as the user may have edited it.

```bash
# Show the full config
cat "$(dirname $(which python))"/../skills/vasp/config.yaml
# Or read it in Python
python -c "import yaml; print(yaml.dump(yaml.safe_load(open('config.yaml'))))"
```

To override individual tags for a single run without editing the file, use `--incar_tags`:
```bash
--incar_tags '{"ENCUT": 600, "EDIFF": 1e-6}'
```

To permanently change a default, edit the relevant preset in `config.yaml` directly.
