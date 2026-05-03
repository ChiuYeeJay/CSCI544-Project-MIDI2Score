# CSCI544-Project-MIDI2Score

## Enviroment Setup

## Dataset and Tokenizer Preparation

### PDMX Download

To train/evaluate the model, we need to first obtain the source dataset. We use PDMX dataset as the source of MusicXML data, and can be downloaded on the [Zendo](https://zenodo.org/records/15571083) website. The only needed files are [mxl.tar.gz](https://zenodo.org/records/15571083/files/mxl.tar.gz?download=1) and [PDMX.csv](https://zenodo.org/records/15571083/files/PDMX.csv?download=1), please extract them and put them under `dataset/PDMX` directory.

### MIDI-MusicXML Paired Dataset Generation

Then we generate MIDI-MusicXML paired dataset from it. The logic is scripted in `pdmx_data_preprocessing.py`, and it contains some path variables at the top that can be changed to fit the need. Execute the following command:

```sh
cd PROJECT_ROOT/data_preprocessing
python3 pdmx_data_preprocessing.py
```

This will extract the `rated_deduplicated` subset of the PDMX dataset and generate `PDMX_preprocessed` dataset with paired MIDI and MusicXML (including non-BPE LMX tokens) with some metadata.

### Dataset Splitting

Then split the dataset for training/validating/testing purposes with the following command:

```sh
cd PROJECT_ROOT/data_preprocessing
python3 data_splitting.py
```

The `data_splitting.py` script will split the dataset by total token length while considering source file and the length of a piece, and output the split information to the `dataset/PDMX_preprocessed/dataset_info_with_partitions.csv` file.

### LMX BPE Tokenizer Training

Before further processing dataset, the BPE tokenizer should be trained first. Execute the following command:

```sh
cd PROJECT_ROOT/tokenizer
python3 musicxml_tokenizer.py
```

This will save a BPE tokenizer record file at `tokenizer/tokenizer.json`.

### Generating HuggingFace Dataset

At this step, we tokenize the MusicXML data to BPE LMX tokens, and MIDI to CPWord tokens. The result is processed and stored with huggingface `datasets` library and `Arrow` format. Since decoder pre-training and seq2seq training need different data, so we have different pipeline for each.

#### For Decoder Pre-training

```sh
cd PROJECT_ROOT
python3 hf_dataset/hf_dataset_lmx.py
```

This will This will generate a dataset with only LMX sequences (without MIDI data) at `dataset/huggingface`.

#### For Seq2Seq Training

```sh
cd PROJECT_ROOT
python3 hf_dataset/hf_dataset_seq2seq.py

# bar-awared truncation
python3 hf_dataset/hf_dataset_seq2seq_truncate.py --input-path dataset/huggingface_seq2seq --output-path dataset/huggingface_seq2seq_truncated --tokenizer-path tokenizer/tokenizer.json --max-source-length 1022 --max-target-length 1022 --lookahead-tokens 0
```

This will generate a dataset without truncation at `dataset/huggingface_seq2seq` and one with truncation at `dataset/huggingface_seq2seq_truncated`.

## Seq2Seq Model Training

## Evaluation

### Running Prediction

### Our Model

### External Models

## Reproducing Our Results
