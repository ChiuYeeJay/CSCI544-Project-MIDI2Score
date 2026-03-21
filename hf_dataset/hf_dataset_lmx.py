import os
import sys
import pandas as pd
from datasets import Dataset, DatasetDict
import multiprocessing

sys.path.append(".")
from tokenizer.musicxml_tokenizer import MusicXMLTokenizer

# ==========================================
# 1. Parameter and Path Configuration
# ==========================================
PDMX_PREPROCESSED_ROOT = "dataset/PDMX_preprocessed"
CSV_PATH = os.path.join(PDMX_PREPROCESSED_ROOT, "dataset_info_with_partitions.csv")
HF_OUTPUT_DIR = "dataset/huggingface"
BPE_MODEL_PATH = "tokenizer/tokenizer.json"
MAX_SEQ_LENGTH = 8192

def main():
    print("=== Initializing Tokenizer ===")
    tokenizer = MusicXMLTokenizer()
    if not os.path.exists(BPE_MODEL_PATH):
        raise FileNotFoundError(f"BPE model not found: {BPE_MODEL_PATH}. Please run the tokenizer training script first.")
    
    tokenizer.load_bpe_model(BPE_MODEL_PATH)

    print(f"\n=== Reading and Filtering Metadata ===")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Partition table not found: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    
    # Filter out entries missing lmx paths or partition info
    initial_count = len(df)
    df = df.dropna(subset=['lmx', 'partition'])
    
    # Verify that files actually exist on disk
    df['lmx_full_path'] = df['lmx'].apply(lambda x: os.path.join(PDMX_PREPROCESSED_ROOT, str(x)))
    df = df[df['lmx_full_path'].apply(os.path.exists)]
    
    print(f"Valid entries: {len(df)} / {initial_count} (Excluded missing or invalid LMX items)")

    # ==========================================
    # 2. Create Hugging Face DatasetDict
    # ==========================================
    print("\n=== Creating Hugging Face Dataset Object ===")
    
    # Split pandas DataFrame based on partition
    train_df = df[df['partition'] == 'training']
    valid_df = df[df['partition'] == 'validation']
    test_df = df[df['partition'] == 'test']

    # Convert to HF Dataset
    hf_datasets = DatasetDict({
        "training": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(valid_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False)
    })

    print(f"Dataset Size - Train: {len(hf_datasets['training'])}, Val: {len(hf_datasets['validation'])}, Test: {len(hf_datasets['test'])}")

    # ==========================================
    # 3. Define Map Function for Tokenization and Truncation
    # ==========================================
    def process_and_tokenize(examples, dropout_rate):
        """
        This function is executed in batches.
        Read LMX files from disk -> Tokenize -> Truncate -> Return input_ids
        """
        # Change tokenizer mode dynamically for process safety
        if dropout_rate > 0.0:
            tokenizer.train_mode(dropout_rate=dropout_rate)
        else:
            tokenizer.eval_mode()

        input_ids_batch = []
        for file_path in examples['lmx_full_path']:
            # Read LMX text
            with open(file_path, 'r', encoding='utf-8') as f:
                lmx_content = f.read().strip()
            
            # Convert LMX to BPE token IDs
            token_ids = tokenizer.encode_lmx_to_bpe(lmx_content)
            input_ids_batch.append(token_ids[:MAX_SEQ_LENGTH])
            
        return {"input_ids": input_ids_batch}

    # ==========================================
    # 4. Execute High-Efficiency Processing
    # ==========================================
    print("\n=== Starting Processing and Tokenization (This may take a while) ===")
    num_proc = max(1, multiprocessing.cpu_count() - 1)
    
    processed_splits = {}
    
    for split_name, dataset in hf_datasets.items():
        bpe_dropout = 0.1 if split_name == "training" else 0.0
        print(f"\n-> Processing '{split_name}' split (BPE Dropout: {bpe_dropout})")
        
        columns_to_remove = dataset.column_names
        
        processed_splits[split_name] = dataset.map(
            process_and_tokenize,
            fn_kwargs={"dropout_rate": bpe_dropout},
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
            remove_columns=columns_to_remove,
            desc=f"Tokenizing {split_name}"
        )

    tokenized_datasets = DatasetDict(processed_splits)

    # ==========================================
    # 5. Save Cache to Disk
    # ==========================================
    print(f"\n=== Saving processed dataset to {HF_OUTPUT_DIR} ===")
    os.makedirs(HF_OUTPUT_DIR, exist_ok=True)

    # save_to_disk preserves the Arrow format for fast loading later via load_from_disk
    tokenized_datasets.save_to_disk(HF_OUTPUT_DIR)
    
    print("=== Done! ===")
    from datasets import load_from_disk
    ds = load_from_disk('dataset/huggingface')
    print(f"training set size: {len(ds['training'])}")
    print(f"validation set size: {len(ds['validation'])}")
    print(f"test set size: {len(ds['test'])}")
    print(f"sample from train: \n{ds['training'][0]}")

if __name__ == "__main__":
    main()