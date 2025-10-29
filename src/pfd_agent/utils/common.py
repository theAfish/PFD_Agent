import traceback
import time
import uuid
import os
from pathlib import Path

def generate_work_path(create: bool = True) -> str:
    """
    Generate a unique working directory path based on call function and current time.

    directory = calling function name + current time + random string.

    Returns:
        str: The path to the working directory.
    """
    calling_function = traceback.extract_stack(limit=2)[-2].name
    current_time = time.strftime("%Y%m%d%H%M%S")
    random_string = str(uuid.uuid4())[:8]

    # Get the base directory (PFD_Agent/src/output)
    base_dir = Path(__file__).parent.parent.parent / "output"
    work_path = base_dir / f"{current_time}.{calling_function}.{random_string}"

    if create:
        work_path.mkdir(parents=True, exist_ok=True)

    return str(work_path)