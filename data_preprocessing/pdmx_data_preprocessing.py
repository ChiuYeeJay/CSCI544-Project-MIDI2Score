import os
import sys
import pandas as pd
import music21
import tqdm
import time
import multiprocessing


from lmx.linearization.Linearizer import Linearizer
from lmx.linearization.Delinearizer import Delinearizer
from lmx.symbolic.MxlFile import MxlFile
import xml.etree.ElementTree as ET

PDMX_ROOT = "../dataset/PDMX/"
PDMX_PREPROCESSED_ROOT = "../dataset/PDMX_preprocessed/"
MAX_TOKEN_LENGTH = 20000  # Maximum number of tokens in a single LMX file

if not PDMX_ROOT.endswith("/"): PDMX_ROOT += "/"
if not PDMX_PREPROCESSED_ROOT.endswith("/"): PDMX_PREPROCESSED_ROOT += "/"
os.makedirs(PDMX_PREPROCESSED_ROOT, exist_ok=True)

def _error_linearizer(self, *values):
    header = f"[ERROR][P:{self._part_id} M:{self._measure_number}]:"
    print(header, *values, file=self._errout)
    self.error_record.append(header + " " + " ".join(map(str, values)))

Linearizer._error = _error_linearizer

def _error_delinearizer(self, token, *values):
    header = f"[ERROR][Token '{token.terminal}' at position {token.position}]:"
    print(header, *values, file=self._errout)
    self.error_record.append(header + " " + " ".join(map(str, values)))

Delinearizer._error = _error_delinearizer

# ### Process MXL

def part_to_score(part: ET.Element, part_id="P2", score_part=None, musicxml_version="3.1") -> ET.ElementTree:
    """Embeds a <part> element within a MusicXML <score-partwise> file"""
    
    root = ET.Element("score-partwise")
    root.attrib["version"] = musicxml_version

    part_list_element = ET.Element("part-list")
    part_list_element.append(score_part)

    root.append(part_list_element)

    part.attrib["id"] = part_id
    root.append(part)

    return ET.ElementTree(root)


def match_part_and_scorepart(parts: list[ET.Element], score_parts: list[ET.Element])->list[tuple[ET.Element, ET.Element, str]]:
    pairs = []
    for part in parts:
        part_id = part.attrib.get("id")
        if part_id is None:
            print(f"Part without id: {ET.tostring(part, encoding='unicode')}")
            continue
        for score_part in score_parts:
            if part_id == score_part.attrib.get("id"):
                pairs.append((part, score_part, part_id))
                break
    return pairs

def truncate_lmx_tokens(tokens, max_token_length):
    bound = 1
    for i in range(max_token_length, 0, -1):
        if tokens[i] == "measure": 
            bound = i
            break
    return tokens[:bound]

def produce_lmx_file(part,  lmx_output_path, max_token_length=MAX_TOKEN_LENGTH):
    linearizer = Linearizer(errout=None)
    linearizer.error_record = []
    linearizer.process_part(part)

    tokens = linearizer.output_tokens
    is_truncated = False
    
    if len(tokens) > max_token_length:
        is_truncated = True
        tokens = truncate_lmx_tokens(tokens, max_token_length)
        assert len(tokens) > 0, "Error: no token left after truncating"

    output_lmx = " ".join(tokens)
    error_record = "|".join(linearizer.error_record) if linearizer.error_record else None
    info = {"is_truncated": is_truncated, "n_tokens": len(tokens)}

    os.makedirs(os.path.dirname(lmx_output_path), exist_ok=True)
    with open(lmx_output_path, "w") as f:
        f.write(output_lmx)
    
    return output_lmx, error_record, info

def delinearize_lmx_tokens(lmx_input, part_id, score_part, lmx_output_path):
    delinearizer = Delinearizer(errout=sys.stderr)
    delinearizer.error_record = []
    delinearizer.process_text(lmx_input)
    score_etree = part_to_score(delinearizer.part_element, part_id, score_part)
    output_xml = str(ET.tostring(
        score_etree.getroot(),
        encoding="utf-8",
        xml_declaration=True
    ), "utf-8")
    error_record = "|".join(delinearizer.error_record) if delinearizer.error_record else None

    os.makedirs(os.path.dirname(lmx_output_path), exist_ok=True)
    with open(lmx_output_path, "w") as f:
        f.write(output_xml)
    
    return output_xml, error_record

def generate_midi_and_analysis(musicxml_str, midi_output_path):
    start_time = time.time()
    # Parse the MusicXML string
    score = music21.converter.parse(musicxml_str)
    parse_time = time.time() - start_time
    
    # Generate MIDI file
    os.makedirs(os.path.dirname(midi_output_path), exist_ok=True)
    score.write("midi", fp=midi_output_path)
    midi_time = time.time() - start_time - parse_time

    # Analyze the score
    if not score.parts:
        raise ValueError("Parsed score has no parts.")
    first_part = score.parts[0]
    n_notes = len(score.flatten().notes)
    n_measure = len(first_part.getElementsByClass(music21.stream.Measure))
    part_instrument = first_part.getInstrument().instrumentName
    part_midi_program = first_part.getInstrument().midiProgram
    analysis_time = time.time() - start_time - parse_time - midi_time

    info = {
        "n_notes": n_notes,
        "n_measure": n_measure,
        "part_instrument": part_instrument,
        "part_midi_program": part_midi_program,
        "parse_time": parse_time,
        "midi_time": midi_time,
        "analysis_time": analysis_time
    }
    
    return info

def process_mxl(entry: tuple[int, pd.DataFrame]):
    try:
        entry = entry[1]
        mxl_path = entry["mxl"]
        input_path = os.path.join(PDMX_ROOT, mxl_path)

        #  Load the MXL file
        if input_path.endswith(".mxl"):
            mxl = MxlFile.load_mxl(input_path)
        if input_path.endswith(".musicxml"):
            with open(input_path, "r") as f:
                mxl = MxlFile(ET.ElementTree(ET.fromstring(f.read())))
        
        # Extract parts and score-part elements
        parts = mxl.tree.findall("part")
        score_parts = mxl.tree.findall("part-list/score-part")
        part_pairs = match_part_and_scorepart(parts, score_parts)
    except Exception as e:
        print(f"Error: XML parsing failed {mxl_path}: {e}")
        return [], []
    
    part_infos = []
    error_part_infos = []

    for part, score_part, part_id in part_pairs:
        error_from = ""
        try:
            # Obtain output paths
            lmx_output_path = PDMX_PREPROCESSED_ROOT + "lmx/" + mxl_path.removeprefix("./mxl/").removesuffix(".mxl") + f"-{part_id}.lmx"
            mxl_output_path = lmx_output_path.replace(".lmx", ".musicxml").replace("/lmx/", "/mxl/")
            midi_output_path = lmx_output_path.replace(".lmx", ".mid").replace("/lmx/", "/midi/")
            # print(lmx_output_path)

            part_start_time = time.time()
            # Produce LMX file
            try:
                output_lmx, lmx_error_record, lmx_info = produce_lmx_file(part, lmx_output_path, MAX_TOKEN_LENGTH)
                lmx_time = time.time() - part_start_time
            except Exception as e:
                error_from = "produce_lmx_file"
                raise
            
            # Delinearize LMX tokens to MusicXML
            try:
                output_xml, mxl_error_record = delinearize_lmx_tokens(output_lmx, part_id, score_part, mxl_output_path)
                mxl_time = time.time() - part_start_time - lmx_time
            except Exception as e:
                error_from = "delinearize_lmx_tokens"
                raise

            # Generate MIDI and analysis information
            try:
                analysis_info = generate_midi_and_analysis(output_xml, midi_output_path)
                total_time = time.time() - part_start_time
            except Exception as e:
                error_from = "generate_midi_and_analysis"
                raise
            
            info = {
                "origin": mxl_path,                                             # origin musicxml Path
                "part_id": part_id,                                             # part id from the origin
                "lmx": lmx_output_path.removeprefix(PDMX_PREPROCESSED_ROOT),    # processed lmx file path
                "mxl": mxl_output_path.removeprefix(PDMX_PREPROCESSED_ROOT),    # processed musicxml flie path
                "midi": midi_output_path.removeprefix(PDMX_PREPROCESSED_ROOT),  # processed midi file path
                "subset_deduplicated": entry["subset:deduplicated"],            # is in deduplicated subset
                "subset_rated": entry["subset:rated"],                          # is in rated subset
                "subset_rated_deduplicated": entry["subset:rated_deduplicated"],# is in both subset
                "part_instrument": analysis_info["part_instrument"],            # text of part instrument
                "part_midi_program": analysis_info["part_midi_program"],        # midi program num of part instrument
                "is_truncated": lmx_info["is_truncated"],                       # the part is truncated or not
                "n_tokens": lmx_info["n_tokens"],                               # number of lmx token
                "n_notes": analysis_info["n_notes"],                            # number of note
                "n_measure": analysis_info["n_measure"],                        # number of measure
                "origin_n_part": len(parts),                                    # number of parts in origin
                "mxl_time": mxl_time,                                           # time for delinearizing lmx
                "lmx_time": lmx_time,                                           # time for linearizing mxl
                "music21_parse_time": analysis_info["parse_time"],              # time for music21 parsing
                "music21_midi_time": analysis_info["midi_time"],                # time for music21 midi conversion
                "music21_analysis_time": analysis_info["analysis_time"],        # time for music21 analysis
                "total_time": total_time,                                       # total time
                "lmx_error_record": lmx_error_record,                           # error record of mxl linearization
                "mxl_error_record": mxl_error_record,                           # error record of lmx delinearization
            }
            part_infos.append(info)
        
        except Exception as e:
            # source_text = f"(from {error_from})" if error_from else ""
            # print(f"Error processing {mxl_path} part {part_id} {source_text}: {e}")
            error_part_infos.append({
                "origin": mxl_path,
                "part_id": part_id,
                "error": str(e),
                "error_from": error_from
            })
            continue
    
    return part_infos, error_part_infos

# ### Parallel processing

if __name__ == "__main__":
    main_start_time = time.time()

    df = pd.read_csv(os.path.join(PDMX_ROOT, "PDMX.csv"))
    df_filtered = df[df["subset:rated_deduplicated"]]
    # df_filtered = df
    df_filtered = df_filtered.loc[:, ["mxl", "subset:deduplicated", "subset:rated", "subset:rated_deduplicated"]]
    df_filtered = df_filtered[~df_filtered["mxl"].isna()]
    df_filtered = df_filtered.reset_index(drop=True)
    print(df_filtered.shape)
    # df_filtered = df_filtered.sample(1000,  random_state=42)

    DATASET_INFO_PATH = os.path.join(PDMX_PREPROCESSED_ROOT, "dataset_info.csv")
    ERROR_DATASET_INFO_PATH = os.path.join(PDMX_PREPROCESSED_ROOT, "error_dataset_info.csv")
    if os.path.exists(DATASET_INFO_PATH): os.remove(DATASET_INFO_PATH)
    if os.path.exists(ERROR_DATASET_INFO_PATH): os.remove(ERROR_DATASET_INFO_PATH)

    DATASET_INFO_COLUMN = [
        "origin", "part_id", "lmx", "mxl", "midi", "subset_deduplicated", "subset_rated", "subset_rated_deduplicated", 
        "part_instrument", "part_midi_program", "is_truncated", "n_tokens", "n_notes", "n_measure", "origin_n_part", 
        "mxl_time", "lmx_time", "music21_parse_time", "music21_midi_time", "music21_analysis_time", "total_time", 
        "lmx_error_record", "mxl_error_record"
    ]
    ERROR_DATASET_INFO_COLUMN = ["origin", "part_id", "error", "error_from"]
    pd.DataFrame(columns=DATASET_INFO_COLUMN).to_csv(DATASET_INFO_PATH, header=True, index=False)
    pd.DataFrame(columns=ERROR_DATASET_INFO_COLUMN).to_csv(ERROR_DATASET_INFO_PATH, header=True, index=False)

    with multiprocessing.Manager() as manager:
        print("Process number:", multiprocessing.cpu_count())
        with multiprocessing.Pool(maxtasksperchild=100) as pool:
            all_part_infos = []
            all_error_infos = []

            for part_infos, error_part_infos in tqdm.tqdm(pool.imap_unordered(process_mxl, df_filtered.iterrows()), total=len(df_filtered)):
                if len(part_infos) > 0: all_part_infos.extend(part_infos)
                if len(error_part_infos) > 0: all_error_infos.extend(error_part_infos)

                # Append info data
                if len(all_part_infos) >= 1000:
                    pd.DataFrame(all_part_infos).to_csv(DATASET_INFO_PATH, mode="a", header=False,index=False)
                    all_part_infos = []
                if len(all_error_infos) >= 1000:
                    pd.DataFrame(all_error_infos).to_csv(ERROR_DATASET_INFO_PATH, mode="a", header=False, index=False)
                    all_error_infos = []

            # remaining parts
            if all_part_infos:
                pd.DataFrame(all_part_infos).to_csv(
                    os.path.join(PDMX_PREPROCESSED_ROOT, "dataset_info.csv"), 
                    mode="a", 
                    header=not os.path.exists(os.path.join(PDMX_PREPROCESSED_ROOT, "dataset_info.csv")),
                    index=False
                )
                all_part_infos = []
            if all_error_infos:
                pd.DataFrame(all_error_infos).to_csv(
                    os.path.join(PDMX_PREPROCESSED_ROOT, "error_dataset_info.csv"),
                    mode="a", 
                    header=not os.path.exists(os.path.join(PDMX_PREPROCESSED_ROOT, "error_dataset_info.csv")),
                    index=False
                )
                all_error_infos = []

    elapsed_time = time.time() - main_start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f'Total time taken: {elapsed_time_str}')