#!/usr/bin/env python3
"""
bohrium_submit.py  —  Submit prepared VASP calc directories to Bohrium via dpdispatcher.

Usage
-----
  python bohrium_submit.py submit \\
      --calc_dirs /tmp/vasp_server/20240323_001 /tmp/vasp_server/20240323_002 \\
      --calc_type relaxation

  # Pipe calc_dirs from vasp_tools.py
  DIRS=$(python vasp_tools.py prepare_scf --structure Al.extxyz | python -c \\
      "import sys,json; print(' '.join(json.load(sys.stdin)['calc_dir_list']))")
  python bohrium_submit.py submit --calc_dirs $DIRS --calc_type scf

File transfer per calc type
---------------------------
  relaxation   forward : POSCAR INCAR POTCAR KPOINTS
               backward: OSZICAR CONTCAR OUTCAR vasprun.xml

  scf          forward : POSCAR INCAR POTCAR KPOINTS
               backward: OSZICAR CONTCAR OUTCAR vasprun.xml CHGCAR WAVECAR

  nscf         forward : POSCAR INCAR POTCAR KPOINTS CHGCAR WAVECAR
               backward: OSZICAR CONTCAR OUTCAR vasprun.xml

All --calc_dirs must share the same parent directory (used as dpdispatcher local_root).

Required environment variables (or .env file next to this script)
-----------------------------------------------------------------
  BOHRIUM_USERNAME       Bohrium account email
  BOHRIUM_PASSWORD       Bohrium account password
  BOHRIUM_PROJECT_ID     Bohrium project ID (integer)
  BOHRIUM_VASP_MACHINE   Bohrium machine type, e.g. "c32_m128_cpu"
  BOHRIUM_VASP_IMAGE     Bohrium container image for VASP
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
from dpdispatcher import Machine, Resources, Task, Submission

_SCRIPT_DIR = Path(__file__).parent

# ── file manifests per calc type ──────────────────────────────────────────────

_FORWARD_FILES = {
    "relaxation": ["POSCAR", "INCAR", "POTCAR", "KPOINTS"],
    "scf":        ["POSCAR", "INCAR", "POTCAR", "KPOINTS"],
    "nscf":       ["POSCAR", "INCAR", "POTCAR", "KPOINTS", "CHGCAR", "WAVECAR"],
}

_BACKWARD_FILES = {
    "relaxation": ["OSZICAR", "CONTCAR", "OUTCAR", "vasprun.xml"],
    "scf":        ["OSZICAR", "CONTCAR", "OUTCAR", "vasprun.xml", "CHGCAR", "WAVECAR"],
    "nscf":       ["OSZICAR", "CONTCAR", "OUTCAR", "vasprun.xml"],
}

_DEFAULT_VASP_COMMAND = (
    "source /opt/intel/oneapi/setvars.sh && mpirun -n 32 vasp_std"
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_env() -> None:
    env_file = _SCRIPT_DIR / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Set it in {_SCRIPT_DIR / '.env'} or export it before running."
        )
    return val


# ── command ───────────────────────────────────────────────────────────────────

def cmd_submit(args) -> dict:
    calc_dirs = [Path(d).resolve() for d in args.calc_dirs]
    calc_type = args.calc_type

    # All calc dirs must share a common parent — used as dpdispatcher local_root
    parents = {d.parent for d in calc_dirs}
    if len(parents) > 1:
        raise ValueError(
            "All --calc_dirs must share the same parent directory "
            "(required by dpdispatcher as local_root).\n"
            f"Found multiple parents: {parents}"
        )
    work_dir = str(next(iter(parents)))

    machine_dict = {
        "batch_type": "Bohrium",
        "context_type": "BohriumContext",
        "local_root": work_dir,
        "remote_profile": {
            "email": _require_env("BOHRIUM_USERNAME"),
            "password": _require_env("BOHRIUM_PASSWORD"),
            "program_id": int(_require_env("BOHRIUM_PROJECT_ID")),
            "keep_backup": True,
            "input_data": {
                "job_type": "container",
                "grouped": True,
                "job_name": f"vasp_{calc_type}",
                "scass_type": _require_env("BOHRIUM_VASP_MACHINE"),
                "platform": "ali",
                "image_name": _require_env("BOHRIUM_VASP_IMAGE"),
            },
        },
    }

    machine = Machine.load_from_dict(machine_dict)
    resources = Resources.load_from_dict({"group_size": args.group_size})

    task_list = [
        Task(
            command=args.vasp_command,
            task_work_path=d.name,
            forward_files=_FORWARD_FILES[calc_type],
            backward_files=_BACKWARD_FILES[calc_type],
        )
        for d in calc_dirs
    ]

    submission = Submission(
        work_base="./",
        machine=machine,
        resources=resources,
        task_list=task_list,
        forward_common_files=[],
        backward_common_files=[],
    )
    submission.run_submission()

    return {
        "status": "success",
        "calc_type": calc_type,
        "calc_dir_list": [str(d) for d in calc_dirs],
    }


# ── argparse ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Submit prepared VASP directories to Bohrium via dpdispatcher.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("submit", help="Submit calc dirs to Bohrium.")
    p.add_argument(
        "--calc_dirs", nargs="+", required=True,
        help="Paths to prepared VASP calculation directories (all must share a parent).",
    )
    p.add_argument(
        "--calc_type", required=True, choices=["relaxation", "scf", "nscf"],
        help="Calculation type — determines which files are uploaded and downloaded.",
    )
    p.add_argument(
        "--group_size", type=int, default=4,
        help="dpdispatcher group_size: jobs per Bohrium submission group (default: 4).",
    )
    p.add_argument(
        "--vasp_command", type=str, default=_DEFAULT_VASP_COMMAND,
        help=f"Shell command to execute VASP on the remote (default: '{_DEFAULT_VASP_COMMAND}').",
    )
    p.set_defaults(func=cmd_submit)

    args = parser.parse_args()
    _load_env()

    try:
        result = args.func(args)
        print(json.dumps(result, indent=2))
        sys.exit(0)
    except Exception as e:
        print(
            json.dumps({"status": "error", "message": str(e),
                        "traceback": traceback.format_exc()}),
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
