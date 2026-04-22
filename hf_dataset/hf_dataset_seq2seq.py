import os
import sys
import pandas as pd
from datasets import Dataset, DatasetDict
import multiprocessing
import miditok
import symusic

sys.path.append(".")
from tokenizer.musicxml_tokenizer import MusicXMLTokenizer
from tokenizer.midi_augmentation import apply_midi_augmentation

# ==========================================
# 1. Parameter and Path Configuration
# ==========================================
PDMX_PREPROCESSED_ROOT = "dataset/PDMX_preprocessed_rd"
CSV_PATH = os.path.join(PDMX_PREPROCESSED_ROOT, "dataset_info_with_partitions.csv")
HF_OUTPUT_DIR = "dataset/huggingface_seq2seq_v3"
BPE_MODEL_PATH = "tokenizer/tokenizer_rd.json"
MAX_SEQ_LENGTH = 10000

def main():
    print("=== Initializing Tokenizer ===")
    # LMX tokenizer
    tokenizer = MusicXMLTokenizer()
    if not os.path.exists(BPE_MODEL_PATH):
        raise FileNotFoundError(f"BPE model not found: {BPE_MODEL_PATH}. Please run the tokenizer training script first.")
    
    tokenizer.load_bpe_model(BPE_MODEL_PATH)

    # CPWord tokenizer
    miditok_config = miditok.TokenizerConfig(
        pitch_range=(0, 127),
        beat_res={(0, 8): 8, (8, 17): 4},
        use_velocities=False,
        use_chords=False,
        use_rests=False,
        use_tempos=True,
        use_time_signatures=True,
        use_sustain_pedals=False,
        use_programs=True,
        use_pitch_bends=False,
        use_pitchdrum_tokens=False,
        one_token_stream_for_programs=True,
        time_signature_range={beat_type: list(range(1, 17)) for beat_type in [2, 4, 8, 16]},
        special_tokens=["PAD", "BOS", "EOS"],
    )

    cpword_tokenizer = miditok.CPWord(miditok_config)

    print(f"\n=== Reading and Filtering Metadata ===")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Partition table not found: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    
    # Filter out entries missing lmx paths or partition info
    initial_count = len(df)
    df = df.dropna(subset=['lmx', 'partition', 'midi'])
    
    # Verify that files actually exist on disk
    df['lmx_full_path'] = df['lmx'].apply(lambda x: os.path.join(PDMX_PREPROCESSED_ROOT, str(x)))
    df['midi_full_path'] = df['midi'].apply(lambda x: os.path.join(PDMX_PREPROCESSED_ROOT, str(x)))
    df = df[df['lmx_full_path'].apply(os.path.exists) & df['midi_full_path'].apply(os.path.exists)]
    
    print(f"Valid entries: {len(df)} / {initial_count} (Excluded missing or invalid LMX/MIDI items)")

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
    # 3. Define Map Function for Tokenization, Augmentation and Truncation
    # ==========================================
    
    def process_and_tokenize(examples, dropout_rate):
        """
        This function is executed in batches.
        1. Read and tokenize MusicXML to LMX tokens
        2. Read MIDI and generate CPWord tokens with 'Clean', 'Light Noise', 'Heavy Noise'
        """
        if dropout_rate > 0.0:
            tokenizer.train_mode(dropout_rate=dropout_rate)
        else:
            tokenizer.eval_mode()

        batch_lmx_ids = []
        batch_midi_clean = []
        batch_midi_light = []
        batch_midi_heavy = []
        batch_lmx_lengths = []

        for lmx_path, midi_path in zip(examples['lmx_full_path'], examples['midi_full_path']):
            try:
                # Read LMX
                with open(lmx_path, 'r', encoding='utf-8') as f:
                    lmx_content = f.read().strip()
                lmx_token_ids = tokenizer.encode_lmx_to_bpe(lmx_content)[:MAX_SEQ_LENGTH]
                lmx_length = len(lmx_token_ids)

                # Read MIDI
                score = symusic.Score(midi_path)
                
                # 1. Clean
                clean_seq = cpword_tokenizer.encode(score)
                clean_ids = clean_seq.ids if hasattr(clean_seq, 'ids') else clean_seq
                
                # 2. Light Noise
                score_light = apply_midi_augmentation(score, heavy_noise=False)
                light_seq = cpword_tokenizer.encode(score_light)
                light_ids = light_seq.ids if hasattr(light_seq, 'ids') else light_seq
                
                # 3. Heavy Noise
                score_heavy = apply_midi_augmentation(score, heavy_noise=True)
                heavy_seq = cpword_tokenizer.encode(score_heavy)
                heavy_ids = heavy_seq.ids if hasattr(heavy_seq, 'ids') else heavy_seq

                batch_lmx_ids.append(lmx_token_ids)
                batch_midi_clean.append(clean_ids[:MAX_SEQ_LENGTH])
                batch_midi_light.append(light_ids[:MAX_SEQ_LENGTH])
                batch_midi_heavy.append(heavy_ids[:MAX_SEQ_LENGTH])
                batch_lmx_lengths.append(lmx_length)

            except Exception as e:
                print(f"Skipping {midi_path} due to error: {e}")
                continue
                
        return {
            "lmx_ids": batch_lmx_ids,
            "midi_clean_ids": batch_midi_clean,
            "midi_light_ids": batch_midi_light,
            "midi_heavy_ids": batch_midi_heavy,
            "lmx_length": batch_lmx_lengths
        }

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
            desc=f"Tokenizing & Augmenting {split_name}"
        )

    tokenized_datasets = DatasetDict(processed_splits)

    # ==========================================
    # 5. Save to Disk
    # ==========================================
    print(f"\n=== Saving processed dataset to {HF_OUTPUT_DIR} ===")
    os.makedirs(HF_OUTPUT_DIR, exist_ok=True)

    # save_to_disk preserves the Arrow format for fast loading later via load_from_disk
    tokenized_datasets.save_to_disk(HF_OUTPUT_DIR)
    
    print("=== Done! ===")
    from datasets import load_from_disk
    ds = load_from_disk(HF_OUTPUT_DIR)
    print(f"Training set size: {len(ds['training'])}")
    print(f"Validation set size: {len(ds['validation'])}")
    print(f"Test set size: {len(ds['test'])}")
    
    print(f"\nDataset Features Preview (Keys available in each sample):")
    print(list(ds['training'][0].keys()))

if __name__ == "__main__":
    main()