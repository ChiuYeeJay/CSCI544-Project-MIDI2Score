import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterator

import tokenizers
import tokenizers.models
import tokenizers.trainers
import tokenizers.processors
import tokenizers.decoders
import tokenizers.pre_tokenizers
from transformers import PreTrainedTokenizerFast

from lmx.linearization.Linearizer import Linearizer
from lmx.linearization.Delinearizer import Delinearizer
from lmx.linearization.vocabulary import ALL_TOKENS
from lmx.symbolic.part_to_score import part_to_score
from lmx.symbolic.MxlFile import MxlFile


# ==========================================
# 1. Configuration
# ==========================================
@dataclass
class TokenizerConfig:
    unicode_byte_begin: int = 33
    bpe_vocab_size: int = 5000
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"
    pad_token: str = "[PAD]"


# ==========================================
# 2. Format Converter (MusicXML <-> LMX)
# ==========================================
class MusicXMLLMXConverter:
    """Handles conversion between MusicXML and LMX formats."""
    
    @staticmethod
    def linearize(musicxml_content: str) -> List[str]:
        """Linearizes a MusicXML string into a list of LMX tokens."""
        mxl = MxlFile(ET.ElementTree(ET.fromstring(musicxml_content)))
        
        parts = mxl.tree.findall("part")
        if not parts:
            raise ValueError("No <part> element found in the MusicXML file.")
        if len(parts) > 1: 
            print("Multiple parts found. Only the first part will be processed.", file=sys.stderr)
            
        linearizer = Linearizer(errout=None)
        linearizer.process_part(parts[0])

        return linearizer.output_tokens

    @staticmethod
    def delinearize(lmx_content: str) -> str:
        """Delinearizes an LMX string back into MusicXML format."""
        delinearizer = Delinearizer(errout=sys.stderr)
        delinearizer.process_text(lmx_content)
        score_etree = part_to_score(delinearizer.part_element)
        
        return str(ET.tostring(
            score_etree.getroot(),
            encoding="utf-8",
            xml_declaration=True
        ), "utf-8")


# ==========================================
# 3. Byte Mapper (LMX <-> Bytes)
# ==========================================
class ByteMapper:
    """Handles bidirectional mapping between LMX symbols and Unicode Bytes."""
    
    def __init__(self, vocab: List[str], byte_offset: int):
        self.vocab = vocab
        self.vocab2bytes = {token: chr(byte_offset + i) for i, token in enumerate(vocab)}
        self.bytes2vocab = {chr(byte_offset + i): token for i, token in enumerate(vocab)}

    def encode_to_bytes(self, lmx_tokens: List[str]) -> str:
        return ''.join(self.vocab2bytes[token] for token in lmx_tokens if token in self.vocab2bytes)

    def decode_to_lmx(self, byte_str: str) -> str:
        return ' '.join([self.bytes2vocab[byte] for byte in byte_str if byte in self.bytes2vocab])
    
    def get_token_byte(self, token: str) -> str:
        return self.vocab2bytes.get(token, "")


# ==========================================
# 4. BPE Manager (BPE Training & Wrapping)
# ==========================================
class BPEManager:
    """Handles the loading, training, and inference of the HuggingFace Tokenizer."""
    
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.bpe_tokenizer: Optional[PreTrainedTokenizerFast] = None

    @property
    def is_trained(self) -> bool:
        return self.bpe_tokenizer is not None

    def load(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BPE model file not found at {model_path}")
            
        self.bpe_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizers.Tokenizer.from_file(model_path),
            pad_token=self.config.pad_token,
            bos_token=self.config.bos_token,
            eos_token=self.config.eos_token,
        )
        print(f"BPE tokenizer loaded from {model_path}")

    def train(self, model_path: str, corpus_iterator: Iterator[str], measure_byte: str):
        if self.is_trained:
            print("BPE tokenizer is already loaded/trained.")
            return

        # Initialize base Tokenizer
        raw_tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
        
        # Pre-tokenizer (using the 'measure' byte as the split pattern)
        raw_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Split(
            pattern=measure_byte, behavior="merged_with_next"
        )
        
        # Trainer configuration
        trainer = tokenizers.trainers.BpeTrainer(
            vocab_size=self.config.bpe_vocab_size, 
            special_tokens=[self.config.pad_token, self.config.bos_token, self.config.eos_token], 
            show_progress=True
        )
        raw_tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)
        
        # Post-processor & Decoder configuration
        raw_tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
            single=f"{self.config.bos_token} $A {self.config.eos_token}",
            special_tokens=[
                (self.config.bos_token, raw_tokenizer.token_to_id(self.config.bos_token)),
                (self.config.eos_token, raw_tokenizer.token_to_id(self.config.eos_token)),
            ],
        )
        raw_tokenizer.decoder = tokenizers.decoders.BPEDecoder()
        
        # Save and wrap
        raw_tokenizer.save(model_path)
        self.bpe_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw_tokenizer,
            pad_token=self.config.pad_token,
            bos_token=self.config.bos_token,
            eos_token=self.config.eos_token,
        )
        print(f"BPE tokenizer trained and saved to {model_path}")

    def set_dropout(self, dropout_rate: float):
        """Sets the BPE dropout rate on the underlying Rust tokenizer model."""
        self._check_ready()
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError("Dropout rate must be between 0.0 and 1.0")
        self.bpe_tokenizer.backend_tokenizer.model.dropout = dropout_rate

    def encode(self, byte_str: str) -> List[int]:
        self._check_ready()
        return self.bpe_tokenizer.encode(byte_str)

    def decode(self, token_ids: List[int]) -> str:
        self._check_ready()
        return self.bpe_tokenizer.decode(token_ids, skip_special_tokens=True)

    def _check_ready(self):
        if not self.is_trained:
            raise RuntimeError("BPE tokenizer is not trained or loaded yet.")


# ==========================================
# 5. Facade API (High-level Interface)
# ==========================================
class MusicXMLTokenizer:
    """
    High-level Pipeline class for external usage.
    Integrates MusicXML conversion, Byte mapping, and BPE encoding/decoding.
    """
    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self.converter = MusicXMLLMXConverter()
        self.mapper = ByteMapper(ALL_TOKENS, self.config.unicode_byte_begin)
        self.bpe = BPEManager(self.config)

    # --- Pipeline Operations ---

    def encode_musicxml_to_bpe(self, musicxml_content: str) -> List[int]:
        """End-to-End: MusicXML -> LMX -> Bytes -> BPE Tokens"""
        lmx_tokens = self.converter.linearize(musicxml_content)
        byte_str = self.mapper.encode_to_bytes(lmx_tokens)
        return self.bpe.encode(byte_str)

    def decode_bpe_to_musicxml(self, bpe_tokens: List[int]) -> str:
        """End-to-End: BPE Tokens -> Bytes -> LMX -> MusicXML"""
        byte_str = self.bpe.decode(bpe_tokens)
        lmx_content = self.mapper.decode_to_lmx(byte_str)
        return self.converter.delinearize(lmx_content)
        
    def encode_lmx_to_bpe(self, lmx_content: str) -> List[int]:
        """LMX string -> BPE Tokens"""
        byte_str = self.mapper.encode_to_bytes(lmx_content.split())
        return self.bpe.encode(byte_str)

    # --- Mode Toggles ---
    
    def train_mode(self, dropout_rate: float = 0.1):
        """Enables BPE dropout. Recommended to be called before training loop."""
        self.bpe.set_dropout(dropout_rate)
        
    def eval_mode(self):
        """Disables BPE dropout. Recommended to be called before evaluation/inference."""
        self.bpe.set_dropout(0.0)

    # --- Delegation for convenience ---

    def load_bpe_model(self, path: str):
        self.bpe.load(path)

    def train_bpe_model(self, path: str, corpus_iterator: Iterator[str]):
        measure_byte = self.mapper.get_token_byte("measure")
        self.bpe.train(path, corpus_iterator, measure_byte)


if __name__ == "__main__":
    import pandas as pd
    
    # Parameters
    BPE_MODEL_PATH = "tokenizer.json"
    PDMX_PREPROCEESSED_ROOT = "../dataset/PDMX_preprocessed"
    
    tokenizer = MusicXMLTokenizer()

    # 1. Prepare Corpus
    print("Preparing corpus...")
    info_csv_path = os.path.join(PDMX_PREPROCEESSED_ROOT, "dataset_info_with_partitions.csv")
    try:
        info_csv = pd.read_csv(info_csv_path)
        corpus = []
        for i, row in info_csv.iterrows():
            if row["partition"] == "training" and pd.notna(row["lmx"]):
                lmx_path = os.path.join(PDMX_PREPROCEESSED_ROOT, str(row["lmx"]))
                with open(lmx_path, 'r') as f:
                    lmx_str = f.read().strip()
                lmx_bytes = tokenizer.mapper.encode_to_bytes(lmx_str.split())
                corpus.append(lmx_bytes)
    except FileNotFoundError:
        print(f"Dataset not found at {info_csv_path}. Skipping training pipeline for demo.")
        corpus = []

    # 2. Load or Train BPE
    if os.path.exists(BPE_MODEL_PATH):
        tokenizer.load_bpe_model(BPE_MODEL_PATH)
    elif corpus:
        print(f"Training new BPE tokenizer...")
        corpus_iterator = (lmx for lmx in corpus if lmx)
        tokenizer.train_bpe_model(BPE_MODEL_PATH, corpus_iterator)
    else:
        raise RuntimeError("BPE tokenizer is not trained and corpus is empty.")
    
    # 3. Testing and Validation Pipeline
    if tokenizer.bpe.is_trained:
        # Example to show Dropout functionality
        tokenizer.train_mode()
        print("Training mode (Dropout ON) activated.")
        
        # Disable for deterministic evaluation
        tokenizer.eval_mode()
        print("Evaluation mode (Dropout OFF) activated.")

        try:
            example_lmx_path = info_csv.sample(1, random_state=42)["lmx"].values[0]
            example_lmx_path = os.path.join(PDMX_PREPROCEESSED_ROOT, str(example_lmx_path))
            
            if os.path.exists(example_lmx_path):
                with open(example_lmx_path, 'r') as f:
                    example_lmx = f.read().strip()
                    
                print(f"Example LMX (Preview): {example_lmx[:50]}...")
                
                # Test Mapper
                example_bytes = tokenizer.mapper.encode_to_bytes(example_lmx.split())
                print(f"Length in bytes: {len(example_bytes)}")
                
                # Test Encode / Decode
                encoded_ids = tokenizer.encode_lmx_to_bpe(example_lmx)
                print(f"Example encoding length: {len(encoded_ids)}")
                
                # Restoration Test
                decoded_bytes = tokenizer.bpe.decode(encoded_ids)
                decoded_lmx = tokenizer.mapper.decode_to_lmx(decoded_bytes)
                
                print(f"Is Consistent: {decoded_lmx == example_lmx}")
        except NameError:
            print("No CSV loaded. Skipping file tests.")