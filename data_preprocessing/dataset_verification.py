# generate with gemini
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path

PDMX_PREPROCESSED_ROOT = "../dataset/PDMX_preprocessed/"
if not PDMX_PREPROCESSED_ROOT.endswith("/"): 
    PDMX_PREPROCESSED_ROOT += "/"

def verify_dataset():
    csv_path = os.path.join(PDMX_PREPROCESSED_ROOT, "dataset_info.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: Log file not found: {csv_path}")
        return

    print("Reading dataset_info.csv...")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} processing records.")

    errors = {
        "missing_files": [],
        "token_mismatch": [],
        "empty_files": [],
        "orphan_files": []
    }

    # Used to record absolute paths of all files expected from the CSV
    expected_files = set()

    print("\nPhase 1: Comparing CSV records with physical files (Top-Down)...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Construct actual file paths
        lmx_path = os.path.join(PDMX_PREPROCESSED_ROOT, str(row['lmx']).lstrip('/'))
        mxl_path = os.path.join(PDMX_PREPROCESSED_ROOT, str(row['mxl']).lstrip('/'))
        midi_path = os.path.join(PDMX_PREPROCESSED_ROOT, str(row['midi']).lstrip('/'))

        expected_files.update([os.path.abspath(lmx_path), os.path.abspath(mxl_path), os.path.abspath(midi_path)])

        # 1. Check if files exist
        if not os.path.exists(lmx_path): errors["missing_files"].append(f"LMX Missing: {lmx_path}")
        if not os.path.exists(mxl_path): errors["missing_files"].append(f"MXL Missing: {mxl_path}")
        if not os.path.exists(midi_path): errors["missing_files"].append(f"MIDI Missing: {midi_path}")

        # 2. Verify LMX Token count consistency
        if os.path.exists(lmx_path):
            with open(lmx_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tokens = content.split()  # Split by whitespace to count tokens
                expected_tokens = int(row['n_tokens'])
                if len(tokens) != expected_tokens:
                    errors["token_mismatch"].append(
                        f"LMX length mismatch [{row['part_id']}]: Expected {expected_tokens}, Actual {len(tokens)} ({lmx_path})"
                    )
        
        # 3. Check if MXL or MIDI files are empty (0 bytes)
        if os.path.exists(mxl_path) and os.path.getsize(mxl_path) == 0:
            errors["empty_files"].append(f"Empty file (MXL): {mxl_path}")
        if os.path.exists(midi_path) and os.path.getsize(midi_path) == 0:
            errors["empty_files"].append(f"Empty file (MIDI): {midi_path}")

    print("\nPhase 2: Scanning physical directories for orphan files (Bottom-Up)...")
    # Scan lmx, mxl, midi directories under the preprocessed root
    check_dirs = ['lmx', 'mxl', 'midi']
    for d in check_dirs:
        dir_path = os.path.join(PDMX_PREPROCESSED_ROOT, d)
        if not os.path.exists(dir_path):
            continue
            
        for root, _, files in os.walk(dir_path):
            for file in files:
                # Exclude hidden files
                if file.startswith('.'): continue
                
                full_path = os.path.abspath(os.path.join(root, file))
                if full_path not in expected_files:
                    errors["orphan_files"].append(f"File not recorded in CSV: {full_path}")

    # === Summary Output ===
    print("\n" + "="*40)
    print("Verification Results Summary")
    print("="*40)
    
    total_errors = sum(len(v) for v in errors.values())
    if total_errors == 0:
        print("Success! Dataset is perfectly aligned with no errors found.")
    else:
        print(f"Warning: Found {total_errors} anomalies:\n")
        
        if errors["missing_files"]:
            print(f"  - Missing Files ({len(errors['missing_files'])}):")
            for e in errors["missing_files"]: print(f"    - {e}")
            
        if errors["token_mismatch"]:
            print(f"\n  - Token Mismatch ({len(errors['token_mismatch'])}):")
            for e in errors["token_mismatch"]: print(f"    - {e}")
            
        if errors["empty_files"]:
            print(f"\n  - Empty Files ({len(errors['empty_files'])}):")
            for e in errors["empty_files"]: print(f"    - {e}")
            
        if errors["orphan_files"]:
            print(f"\n  - Orphan Files (On disk but not in CSV) ({len(errors['orphan_files'])}):")
            for e in errors["orphan_files"]: print(f"    - {e}")
            
        print("\nTip: You can export the anomaly list to a .txt file for detailed inspection.")

if __name__ == "__main__":
    verify_dataset()