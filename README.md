# CSCI544-Project-MIDI2Score

## Enviroment Setup

1. Install MuseScore for evaluation: https://musescore.org/en/download

2. Install requirements
```sh
pip install -r requirements.txt
```

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

This will generate a dataset with only LMX sequences (without MIDI data) at `dataset/huggingface`.

#### For Seq2Seq Training

```sh
cd PROJECT_ROOT
python3 hf_dataset/hf_dataset_seq2seq.py

# bar-awared truncation
python3 hf_dataset/hf_dataset_seq2seq_truncate.py --input-path dataset/huggingface_seq2seq --output-path dataset/huggingface_seq2seq_truncated --tokenizer-path tokenizer/tokenizer.json --max-source-length 1022 --max-target-length 1022 --lookahead-tokens 0
```

This will generate a dataset without truncation at `dataset/huggingface_seq2seq` and one with truncation at `dataset/huggingface_seq2seq_truncated`.

#### Generating Dataset for Evaluation

Since external models or applications usually don't use our tokenization scheme, so we need a dataset that contains raw MIDI and MusicXML that is aligned to our truncated dataset for comparison. The `eval_dataset`  can be generated using the following command:

```sh
cd PROJECT_ROOT
python3 hf_dataset/hf_dataset_eval_generate.py --input-path dataset/huggingface_seq2seq_truncated --output-root dataset/eval_dataset --tokenizer-path DATA/tokenizer.json --split test  --ratio-clean 1 --ratio-light 1 --ratio-heavy 1 --seed 42 --overwrite
```

This will produce a dataset with aligned MIDI, CPWord, MusicXML, LMX data, and some meta data at `dataset/eval_dataset`.

## Seq2Seq Model Training

We use config files to manage model hyper-parameters and other configuration of training. The configs that were used in experiments and debugging are in `configs` directory.

In the config files, there are some specified paths for dataset location, tokenizer record, pre-trained decoder weights, and optionally other paths. The paths need to be changed to the real path or the corresponding files should be moved to the specified location, so that the training can be started.

The seq2seq training can be started by executing the following command:

```sh
python3 run_seq2seq.py --config configs/experiments/seq2seq_fullft_32.yaml
```

This will start the training process, and finally generate the best/latest model weight and the weight for the last 4 epochs at the specified location in the config file (for the example above, they are saved in the `artifacts/seq2seq` directory). Also, the log data will be saved to the specified path, with default tensorboard and CSV logs, as well as optional wandb data (for the example above, they are stored in `logs` directory).

## Evaluation

### Running Prediction

### Our Model

### External Models
To convert your midi files using external models, run this code
```sh
cd PROJECT_ROOT
python3 convert.py --input <input_folder> --output <output_folder> --method <music21|musescore>
```
For example if you have midi files in a folder called "midi_folder" and wanted to output them to "music21_folder" using music21 to convert them, run :
```sh
cd PROJECT_ROOT
python3 convert.py --input midi_folder --output music21_folder --method music21
```
If you have a different path for musescore, and want to convert your midi files to musicxml using musescore, run this
```sh
cd PROJECT_ROOT
python3 convert.py --input midi_folder --output musescore_folder --method musescore --musescore-path PATH_TO_MUSESCORE
```

After you have your midis converted and want to get evaluation statistics on your musecore folder, run this:

```sh
cd PROJECT_ROOT
python3 evaluation.py --pred_xml_dir musescore_folder --gt_xml_dir ground_truth_folder
```
If you want statistics based on clean, light or heavy, find the jsonl file and run this:
 ```sh
 cd PROJECT_ROOT
 python3 evaluation.py --pred_xml_dir musescore_folder --gt_xml_dir ground_truth_folder --manifest_jsonl manifest.jsonl
 ```

 To replicate the evaluation we did on our examples, use this command
 ```sh
 cd PROJECT_ROOT
 python3 evaluation.py --pred_xml_dir musescore_folder --gt_xml_dir ground_truth_folder --manifest_jsonl manifest.jsonl --onset_tol 0.25 --duration_tol 0.25
 ```

## Reproducing Our Results

The training configs we use for experiment and the final result are as the followings:

| **Title**                              | **Used in**                | **config name (in `configs/experiments`)** |
| -------------------------------------- | -------------------------- | ----------------------------------------------------- |
| LoRA without Curriculum Learning       | Experiment 1               | `seq2seq_lora_no_curriculum.yaml`                     |
| LoRA with Curriculum Learning          | Experiment 1, 2            | `seq2seq_lora_16.yaml`                                |
| Full Fine-tuning                       | Experiment 2, 3            | `seq2seq_fullft.yaml`                                 |
| End-to-End                             | Experiment 3, 4            | `seq2seq_e2e.yaml`                                    |
| End-to-End with Symmetric Architecture | Experiment 4               | `seq2seq_e2e_symmetric.yaml`                          |
| Main Result                            | Main Result and Comparison | `seq2seq_fullft_32.yaml`                              |

The necessary "DATA" directory can be found on the [Google Drive](https://drive.google.com/file/d/1M_ztdOLgMSm0FcB7N67u9xk7owL8ZFqn/view?usp=sharing), including datasets, pre-trained decoder weights and the tokenizer record file.

The seq2seq training process are all done with A40 instances on the CARC.
