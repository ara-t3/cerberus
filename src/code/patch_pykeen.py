import pykeen.losses
import shutil
from pathlib import Path

def patch_pykeen():
    # 1. Find the path to the original PyKEEN losses.py file
    target_path = Path(pykeen.losses.__file__)
    
    # 2. Calculate the exact path of the source file
    # Path(__file__).parent gets the 'code/' folder
    # .parent goes up one level to the root folder ('src/')
    current_dir = Path(__file__).parent
    source_path = current_dir.parent / "patches" / "losses.py"
    
    if not source_path.exists():
        raise FileNotFoundError(f"Unable to find the source file in: {source_path}")

    # 3. Override the original losses.py with the patched version
    shutil.copy(source_path, target_path)
    print(f"[SUCCESS] The file {target_path.name} of PyKEEN has been overwritten.")
    print(f"Patched file in: {target_path}")

if __name__ == "__main__":
    patch_pykeen()