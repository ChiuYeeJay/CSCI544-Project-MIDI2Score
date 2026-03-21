import os
import pandas as pd
from multiprocessing import Pool
from musicxml_tokenizer import MusicXMLTokenizer

# Parameters
BPE_MODEL_PATH = "tokenizer.json"
PDMX_PREPROCEESSED_ROOT = "../dataset/PDMX_preprocessed"
INFO_CSV_PATH = os.path.join(PDMX_PREPROCEESSED_ROOT, "dataset_info_with_partitions.csv")

def process_lmx_file(args):
    """Worker function to read and map an LMX file to bytes."""
    lmx_path, byte_offset, vocab = args
    try:
        # Avoid passing the whole tokenizer object through multiprocessing by mapping manually
        vocab2bytes = {token: chr(byte_offset + i) for i, token in enumerate(vocab)}
        
        with open(lmx_path, 'r') as f:
            lmx_str = f.read().strip()
            
        lmx_tokens = lmx_str.split()
        lmx_len = len(lmx_tokens)
        lmx_bytes = ''.join(vocab2bytes[token] for token in lmx_tokens if token in vocab2bytes)
        return lmx_bytes, lmx_len
    except Exception as e:
        print(f"Error processing {lmx_path}: {e}")
        return None, 0

if __name__ == "__main__":
    print("Initializing tokenizer...")
    tokenizer = MusicXMLTokenizer()
    tokenizer.load_bpe_model(BPE_MODEL_PATH)
    tokenizer.eval_mode() # Ensure dropout is off for consistent evaluation stats

    print(f"Loading dataset info from {INFO_CSV_PATH}...")
    info_csv = pd.read_csv(INFO_CSV_PATH)
    
    # Filter valid files
    valid_rows = info_csv[(info_csv["partition"] == "train") & (info_csv["lmx"].notna())]
    file_paths = [os.path.join(PDMX_PREPROCEESSED_ROOT, str(path)) for path in valid_rows["lmx"]]
    
    print(f"Found {len(file_paths)} files. Reading and converting to bytes...")
    
    # Prepare arguments for multiprocessing
    vocab = tokenizer.mapper.vocab
    byte_offset = tokenizer.config.unicode_byte_begin
    worker_args = [(path, byte_offset, vocab) for path in file_paths]

    # Multiprocessing file reading & byte mapping
    corpus_bytes = []
    total_lmx_tokens = 0
    
    with Pool() as pool:
        results = pool.map(process_lmx_file, worker_args)
        for byte_str, lmx_len in results:
            if byte_str:
                corpus_bytes.append(byte_str)
                total_lmx_tokens += lmx_len

    print("Batch encoding the entire corpus (this uses Rust under the hood and is very fast)...")
    # encode_batch is highly optimized for lists of strings
    batch_encodings = tokenizer.bpe.bpe_tokenizer.backend_tokenizer.encode_batch(corpus_bytes)
    
    # Calculate Total BPE Tokens
    total_bpe_tokens = sum(len(encoding.ids) for encoding in batch_encodings)
    
    print("\n" + "="*40)
    print("📊 Corpus Tokenization Statistics")
    print("="*40)
    print(f"Total files processed:       {len(corpus_bytes)}")
    print(f"Total Original LMX Tokens:   {total_lmx_tokens:,}")
    print(f"Total Encoded BPE Tokens:    {total_bpe_tokens:,}")
    
    if total_lmx_tokens > 0:
        inflation_factor = total_bpe_tokens / total_lmx_tokens
        print(f"Inflation Factor:            {inflation_factor:.4f} (BPE / LMX)")
    print("="*40)