import argparse
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ============================================================
# Basic data structures
# ============================================================

@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    onset: int
    duration: int
    measure_idx: int
    staff: Optional[int]
    voice: Optional[str]


# ============================================================
# Generic helpers
# ============================================================

STEP_TO_SEMITONE = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}


def safe_int(text: Optional[str], default: int = 0) -> int:
    if text is None:
        return default
    try:
        return int(text.strip())
    except Exception:
        return default


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def find_child(elem: ET.Element, child_name: str) -> Optional[ET.Element]:
    for child in elem:
        if local_name(child.tag) == child_name:
            return child
    return None


def find_children(elem: ET.Element, child_name: str) -> List[ET.Element]:
    return [child for child in elem if local_name(child.tag) == child_name]


def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        a = seq1[i - 1]
        for j in range(1, m + 1):
            b = seq2[j - 1]
            cost = 0 if a == b else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[n][m]


# ============================================================
# LMX evaluation
# ============================================================

def read_lmx_tokens(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return text.split() if text else []


def split_lmx_by_measure(tokens: List[str]) -> List[List[str]]:
    """
    Split a flat LMX token sequence into measures.
    Assumption:
      - each measure starts with token 'measure'
    """
    measures: List[List[str]] = []
    current: List[str] = []

    for tok in tokens:
        if tok == "measure":
            if current:
                measures.append(current)
            current = [tok]
        else:
            current.append(tok)

    if current:
        measures.append(current)

    return measures


def normalized_edit_distance(seq_pred: List[str], seq_gt: List[str]) -> Dict[str, float]:
    dist = levenshtein_distance(seq_pred, seq_gt)
    denom = max(len(seq_gt), 1)
    return {
        "edit_distance": dist,
        "normalized_edit_distance": dist / denom,
    }


def evaluate_lmx_pair(pred_path: str, gt_path: str) -> Dict:
    pred_tokens = read_lmx_tokens(pred_path)
    gt_tokens = read_lmx_tokens(gt_path)

    exact_match = int(pred_tokens == gt_tokens)

    seq_scores = normalized_edit_distance(pred_tokens, gt_tokens)

    pred_measures = split_lmx_by_measure(pred_tokens)
    gt_measures = split_lmx_by_measure(gt_tokens)

    total_measures = max(len(pred_measures), len(gt_measures))
    strict_correct = 0
    measure_edit_values = []

    for i in range(total_measures):
        pred_m = pred_measures[i] if i < len(pred_measures) else []
        gt_m = gt_measures[i] if i < len(gt_measures) else []

        if pred_m == gt_m:
            strict_correct += 1

        med = normalized_edit_distance(pred_m, gt_m)["normalized_edit_distance"]
        measure_edit_values.append(med)

    strict_measure_accuracy = (
        strict_correct / total_measures if total_measures > 0 else 0.0
    )
    avg_measure_edit_distance = (
        sum(measure_edit_values) / len(measure_edit_values)
        if measure_edit_values
        else 0.0
    )

    return {
        "file": os.path.basename(gt_path),
        "lmx_num_pred_tokens": len(pred_tokens),
        "lmx_num_gt_tokens": len(gt_tokens),
        "lmx_exact_match": exact_match,
        "lmx_edit_distance": seq_scores["edit_distance"],
        "lmx_normalized_edit_distance": seq_scores["normalized_edit_distance"],
        "lmx_num_pred_measures": len(pred_measures),
        "lmx_num_gt_measures": len(gt_measures),
        "lmx_measure_strict_accuracy": strict_measure_accuracy,
        "lmx_measure_avg_edit_distance": avg_measure_edit_distance,
    }


def aggregate_lmx_results(file_results: List[Dict]) -> Dict:
    if not file_results:
        return {}

    num_files = len(file_results)

    return {
        "num_files": num_files,
        "lmx_exact_match_rate": sum(r["lmx_exact_match"] for r in file_results) / num_files,
        "avg_lmx_normalized_edit_distance": sum(r["lmx_normalized_edit_distance"] for r in file_results) / num_files,
        "avg_lmx_measure_strict_accuracy": sum(r["lmx_measure_strict_accuracy"] for r in file_results) / num_files,
        "avg_lmx_measure_edit_distance": sum(r["lmx_measure_avg_edit_distance"] for r in file_results) / num_files,
    }


# ============================================================
# MusicXML parsing
# ============================================================

def pitch_to_midi(note_elem: ET.Element) -> Optional[int]:
    pitch_elem = find_child(note_elem, "pitch")
    if pitch_elem is None:
        return None

    step_elem = find_child(pitch_elem, "step")
    octave_elem = find_child(pitch_elem, "octave")
    alter_elem = find_child(pitch_elem, "alter")

    if step_elem is None or octave_elem is None or step_elem.text is None or octave_elem.text is None:
        return None

    step = step_elem.text.strip()
    octave = safe_int(octave_elem.text, default=0)
    alter = safe_int(alter_elem.text, default=0) if alter_elem is not None else 0

    if step not in STEP_TO_SEMITONE:
        return None

    return 12 * (octave + 1) + STEP_TO_SEMITONE[step] + alter


def note_is_rest(note_elem: ET.Element) -> bool:
    return find_child(note_elem, "rest") is not None


def note_is_chord(note_elem: ET.Element) -> bool:
    return find_child(note_elem, "chord") is not None


def get_note_duration(note_elem: ET.Element) -> int:
    dur_elem = find_child(note_elem, "duration")
    return safe_int(dur_elem.text if dur_elem is not None else None, default=0)


def get_note_staff(note_elem: ET.Element) -> Optional[int]:
    staff_elem = find_child(note_elem, "staff")
    if staff_elem is None or staff_elem.text is None:
        return None
    return safe_int(staff_elem.text, default=0)


def get_note_voice(note_elem: ET.Element) -> Optional[str]:
    voice_elem = find_child(note_elem, "voice")
    if voice_elem is None or voice_elem.text is None:
        return None
    return voice_elem.text.strip()


def parse_musicxml_file(xml_path: str) -> Dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    part_elem = None
    for elem in root.iter():
        if local_name(elem.tag) == "part":
            part_elem = elem
            break

    if part_elem is None:
        raise ValueError(f"No <part> found in {xml_path}")

    all_notes: List[NoteEvent] = []
    measure_note_keys_staff: List[List[Tuple[int, int, int, Optional[int]]]] = []
    measure_note_keys_basic: List[List[Tuple[int, int, int]]] = []
    structure_tokens: List[str] = []

    absolute_time = 0
    measures = [m for m in part_elem if local_name(m.tag) == "measure"]

    for measure_idx, measure_elem in enumerate(measures, start=1):
        measure_start_time = absolute_time
        measure_time = measure_start_time
        last_non_chord_onset = None

        measure_tokens: List[str] = []
        current_measure_notes_staff: List[Tuple[int, int, int, Optional[int]]] = []
        current_measure_notes_basic: List[Tuple[int, int, int]] = []

        for child in measure_elem:
            tag = local_name(child.tag)
            if tag == "attributes":
                divisions_elem = find_child(child, "divisions")
                if divisions_elem is not None and divisions_elem.text is not None:
                    div = safe_int(divisions_elem.text)
                    measure_tokens.append(f"DIV_{div}")

                time_elem = find_child(child, "time")
                if time_elem is not None:
                    beats_elem = find_child(time_elem, "beats")
                    beat_type_elem = find_child(time_elem, "beat-type")
                    beats = safe_int(beats_elem.text if beats_elem is not None else None, default=0)
                    beat_type = safe_int(beat_type_elem.text if beat_type_elem is not None else None, default=0)
                    measure_tokens.append(f"TIME_{beats}_{beat_type}")

                key_elem = find_child(child, "key")
                if key_elem is not None:
                    fifths_elem = find_child(key_elem, "fifths")
                    if fifths_elem is not None and fifths_elem.text is not None:
                        measure_tokens.append(f"KEY_{safe_int(fifths_elem.text)}")

                clef_elems = find_children(child, "clef")
                for clef in clef_elems:
                    sign_elem = find_child(clef, "sign")
                    line_elem = find_child(clef, "line")
                    if sign_elem is not None and line_elem is not None and sign_elem.text and line_elem.text:
                        measure_tokens.append(f"CLEF_{sign_elem.text.strip()}_{line_elem.text.strip()}")

        for child in measure_elem:
            tag = local_name(child.tag)

            if tag == "note":
                duration = get_note_duration(child)
                is_rest = note_is_rest(child)
                is_chord = note_is_chord(child)
                staff = get_note_staff(child)
                voice = get_note_voice(child)

                if is_chord:
                    onset = last_non_chord_onset if last_non_chord_onset is not None else measure_time
                else:
                    onset = measure_time
                    last_non_chord_onset = onset

                if is_rest:
                    measure_tokens.append(f"REST_D{duration}_S{staff}")
                else:
                    pitch = pitch_to_midi(child)
                    if pitch is not None:
                        all_notes.append(
                            NoteEvent(
                                pitch=pitch,
                                onset=onset,
                                duration=duration,
                                measure_idx=measure_idx,
                                staff=staff,
                                voice=voice,
                            )
                        )
                        current_measure_notes_staff.append((pitch, onset, duration, staff))
                        current_measure_notes_basic.append((pitch, onset, duration))
                        measure_tokens.append(f"NOTE_P{pitch}_D{duration}_S{staff}")

                if not is_chord:
                    measure_time += duration

            elif tag == "backup":
                dur_elem = find_child(child, "duration")
                dur = safe_int(dur_elem.text if dur_elem is not None else None, default=0)
                measure_time -= dur
                measure_tokens.append(f"BACKUP_{dur}")

            elif tag == "forward":
                dur_elem = find_child(child, "duration")
                dur = safe_int(dur_elem.text if dur_elem is not None else None, default=0)
                measure_time += dur
                measure_tokens.append(f"FORWARD_{dur}")

        measure_note_keys_staff.append(current_measure_notes_staff)
        measure_note_keys_basic.append(current_measure_notes_basic)
        structure_tokens.extend(["<MEASURE_START>"] + measure_tokens + ["<MEASURE_END>"])

        measure_span = max(0, measure_time - measure_start_time)
        absolute_time = measure_start_time + measure_span

    return {
        "notes": all_notes,
        "measure_note_keys_staff": measure_note_keys_staff,
        "measure_note_keys_basic": measure_note_keys_basic,
        "structure_tokens": structure_tokens,
        "num_measures": len(measure_note_keys_staff),
    }


# ============================================================
# XML note metrics
# ============================================================

def counter_from_notes(notes: List[NoteEvent], key_mode: str) -> Counter:
    keys = []
    for n in notes:
        if key_mode == "pitch":
            keys.append((n.pitch,))
        elif key_mode == "pitch_onset":
            keys.append((n.pitch, n.onset))
        elif key_mode == "pitch_onset_duration":
            keys.append((n.pitch, n.onset, n.duration))
        elif key_mode == "pitch_onset_duration_staff":
            keys.append((n.pitch, n.onset, n.duration, n.staff))
        else:
            raise ValueError(f"Unsupported key_mode: {key_mode}")
    return Counter(keys)


def precision_recall_f1_from_mode(pred_notes: List[NoteEvent], gt_notes: List[NoteEvent], key_mode: str) -> Dict[str, float]:
    pred_counter = counter_from_notes(pred_notes, key_mode)
    gt_counter = counter_from_notes(gt_notes, key_mode)

    tp = 0
    for k, pred_count in pred_counter.items():
        tp += min(pred_count, gt_counter.get(k, 0))

    total_pred = sum(pred_counter.values())
    total_gt = sum(gt_counter.values())

    fp = total_pred - tp
    fn = total_gt - tp

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def measure_accuracy_and_ser(pred_measures: List[List[Tuple]], gt_measures: List[List[Tuple]]) -> Dict[str, float]:
    total = max(len(pred_measures), len(gt_measures))
    if total == 0:
        return {
            "measure_accuracy": 0.0,
            "ser": 0.0,
            "correct_measures": 0,
            "wrong_measures": 0,
            "total_measures": 0,
        }

    correct = 0
    for i in range(total):
        pred_notes = pred_measures[i] if i < len(pred_measures) else []
        gt_notes = gt_measures[i] if i < len(gt_measures) else []

        pred_counter = Counter(pred_notes)
        gt_counter = Counter(gt_notes)

        if pred_counter == gt_counter:
            correct += 1

    wrong = total - correct

    return {
        "measure_accuracy": correct / total,
        "ser": wrong / total,
        "correct_measures": correct,
        "wrong_measures": wrong,
        "total_measures": total,
    }


def average_measure_note_f1(pred_measures: List[List[Tuple]], gt_measures: List[List[Tuple]]) -> Dict[str, float]:
    total = max(len(pred_measures), len(gt_measures))
    if total == 0:
        return {
            "avg_measure_note_f1": 0.0
        }

    per_measure_f1 = []

    for i in range(total):
        pred_notes = pred_measures[i] if i < len(pred_measures) else []
        gt_notes = gt_measures[i] if i < len(gt_measures) else []

        pred_counter = Counter(pred_notes)
        gt_counter = Counter(gt_notes)

        tp = 0
        for k, pred_count in pred_counter.items():
            tp += min(pred_count, gt_counter.get(k, 0))

        total_pred = sum(pred_counter.values())
        total_gt = sum(gt_counter.values())

        precision = tp / total_pred if total_pred > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_measure_f1.append(f1)

    return {
        "avg_measure_note_f1": sum(per_measure_f1) / len(per_measure_f1)
    }


def evaluate_xml_pair(pred_path: str, gt_path: str) -> Dict:
    pred_data = parse_musicxml_file(pred_path)
    gt_data = parse_musicxml_file(gt_path)

    pitch_scores = precision_recall_f1_from_mode(pred_data["notes"], gt_data["notes"], "pitch")
    po_scores = precision_recall_f1_from_mode(pred_data["notes"], gt_data["notes"], "pitch_onset")
    pod_scores = precision_recall_f1_from_mode(pred_data["notes"], gt_data["notes"], "pitch_onset_duration")
    pods_scores = precision_recall_f1_from_mode(pred_data["notes"], gt_data["notes"], "pitch_onset_duration_staff")

    strict_measure_scores = measure_accuracy_and_ser(
        pred_data["measure_note_keys_staff"],
        gt_data["measure_note_keys_staff"],
    )

    soft_measure_scores = average_measure_note_f1(
        pred_data["measure_note_keys_basic"],
        gt_data["measure_note_keys_basic"],
    )

    struct_scores = normalized_edit_distance(
        pred_data["structure_tokens"],
        gt_data["structure_tokens"],
    )

    return {
        "file": os.path.basename(gt_path),
        "num_pred_notes": len(pred_data["notes"]),
        "num_gt_notes": len(gt_data["notes"]),

        "pitch_precision": pitch_scores["precision"],
        "pitch_recall": pitch_scores["recall"],
        "pitch_f1": pitch_scores["f1"],

        "pitch_onset_precision": po_scores["precision"],
        "pitch_onset_recall": po_scores["recall"],
        "pitch_onset_f1": po_scores["f1"],

        "pitch_onset_duration_precision": pod_scores["precision"],
        "pitch_onset_duration_recall": pod_scores["recall"],
        "pitch_onset_duration_f1": pod_scores["f1"],

        "pitch_onset_duration_staff_precision": pods_scores["precision"],
        "pitch_onset_duration_staff_recall": pods_scores["recall"],
        "pitch_onset_duration_staff_f1": pods_scores["f1"],

        "measure_accuracy": strict_measure_scores["measure_accuracy"],
        "ser": strict_measure_scores["ser"],
        "correct_measures": strict_measure_scores["correct_measures"],
        "wrong_measures": strict_measure_scores["wrong_measures"],
        "total_measures": strict_measure_scores["total_measures"],

        "avg_measure_note_f1": soft_measure_scores["avg_measure_note_f1"],

        "xml_structure_edit_distance": struct_scores["edit_distance"],
        "xml_normalized_structure_edit_distance": struct_scores["normalized_edit_distance"],
    }


def aggregate_xml_results(file_results: List[Dict]) -> Dict:
    if not file_results:
        return {}

    n = len(file_results)

    total_correct_measures = sum(r["correct_measures"] for r in file_results)
    total_measures = sum(r["total_measures"] for r in file_results)
    total_wrong_measures = sum(r["wrong_measures"] for r in file_results)

    return {
        "num_files": n,

        "pitch_f1": sum(r["pitch_f1"] for r in file_results) / n,
        "pitch_onset_f1": sum(r["pitch_onset_f1"] for r in file_results) / n,
        "pitch_onset_duration_f1": sum(r["pitch_onset_duration_f1"] for r in file_results) / n,
        "pitch_onset_duration_staff_f1": sum(r["pitch_onset_duration_staff_f1"] for r in file_results) / n,

        "strict_measure_accuracy": (total_correct_measures / total_measures) if total_measures > 0 else 0.0,
        "ser": (total_wrong_measures / total_measures) if total_measures > 0 else 0.0,
        "avg_measure_note_f1": sum(r["avg_measure_note_f1"] for r in file_results) / n,
        "avg_xml_normalized_structure_edit_distance": sum(r["xml_normalized_structure_edit_distance"] for r in file_results) / n,
    }


# ============================================================
# Validity helpers
# ============================================================

def list_files_with_ext(folder: Optional[str], exts: Tuple[str, ...]) -> List[str]:
    if folder is None or not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])


# ============================================================
# Main evaluation runner
# ============================================================

def evaluate_lmx_dirs(pred_dir: str, gt_dir: str) -> Dict:
    gt_files = list_files_with_ext(gt_dir, (".lmx", ".txt"))
    file_results = []
    missing_preds = []
    parse_failures = []

    for filename in gt_files:
        gt_path = os.path.join(gt_dir, filename)
        pred_path = os.path.join(pred_dir, filename)

        if not os.path.exists(pred_path):
            missing_preds.append(filename)
            continue

        try:
            result = evaluate_lmx_pair(pred_path, gt_path)
            file_results.append(result)
            print(
                f"[LMX][OK] {filename} | "
                f"Exact={result['lmx_exact_match']} | "
                f"ED={result['lmx_normalized_edit_distance']:.4f} | "
                f"MeasureAcc={result['lmx_measure_strict_accuracy']:.4f}"
            )
        except Exception as e:
            parse_failures.append({"file": filename, "error": str(e)})
            print(f"[LMX][ERROR] {filename}: {e}")

    summary = aggregate_lmx_results(file_results)
    validity = {
        "total_gt_files": len(gt_files),
        "evaluated_files": len(file_results),
        "missing_prediction_files": len(missing_preds),
        "parse_failure_files": len(parse_failures),
        "prediction_available_rate": (len(file_results) + len(parse_failures)) / len(gt_files) if gt_files else 0.0,
        "lmx_eval_success_rate": len(file_results) / len(gt_files) if gt_files else 0.0,
    }

    return {
        "summary": summary,
        "per_file": file_results,
        "missing_predictions": missing_preds,
        "failures": parse_failures,
        "validity": validity,
    }


def evaluate_xml_dirs(pred_dir: str, gt_dir: str) -> Dict:
    gt_files = list_files_with_ext(gt_dir, (".xml",))
    file_results = []
    missing_preds = []
    parse_failures = []

    for filename in gt_files:
        gt_path = os.path.join(gt_dir, filename)
        pred_path = os.path.join(pred_dir, filename)

        if not os.path.exists(pred_path):
            missing_preds.append(filename)
            continue

        try:
            result = evaluate_xml_pair(pred_path, gt_path)
            file_results.append(result)
            print(
                f"[XML][OK] {filename} | "
                f"PitchF1={result['pitch_f1']:.4f} | "
                f"PitchOnDurStaffF1={result['pitch_onset_duration_staff_f1']:.4f} | "
                f"MeasureAcc={result['measure_accuracy']:.4f} | "
                f"SER={result['ser']:.4f} | "
                f"StructEdit={result['xml_normalized_structure_edit_distance']:.4f}"
            )
        except Exception as e:
            parse_failures.append({"file": filename, "error": str(e)})
            print(f"[XML][ERROR] {filename}: {e}")

    summary = aggregate_xml_results(file_results)
    validity = {
        "total_gt_files": len(gt_files),
        "evaluated_files": len(file_results),
        "missing_prediction_files": len(missing_preds),
        "parse_failure_files": len(parse_failures),
        "prediction_available_rate": (len(file_results) + len(parse_failures)) / len(gt_files) if gt_files else 0.0,
        "xml_parse_success_rate": len(file_results) / len(gt_files) if gt_files else 0.0,
    }

    return {
        "summary": summary,
        "per_file": file_results,
        "missing_predictions": missing_preds,
        "failures": parse_failures,
        "validity": validity,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluation for MIDI2Score project (LMX + MusicXML).")

    parser.add_argument("--pred_lmx_dir", type=str, default=None)
    parser.add_argument("--gt_lmx_dir", type=str, default=None)

    parser.add_argument("--pred_xml_dir", type=str, default=None)
    parser.add_argument("--gt_xml_dir", type=str, default=None)

    parser.add_argument("--save_json", type=str, default=None)

    args = parser.parse_args()

    run_lmx = args.pred_lmx_dir and args.gt_lmx_dir
    run_xml = args.pred_xml_dir and args.gt_xml_dir

    if not run_lmx and not run_xml:
        raise ValueError("You must provide at least one pair of directories: "
                         "(--pred_lmx_dir, --gt_lmx_dir) or (--pred_xml_dir, --gt_xml_dir).")

    payload = {}

    if run_lmx:
        print("\n================ LMX Evaluation ================\n")
        payload["lmx"] = evaluate_lmx_dirs(args.pred_lmx_dir, args.gt_lmx_dir)

        print("\n----- LMX Summary -----")
        if payload["lmx"]["summary"]:
            for k, v in payload["lmx"]["summary"].items():
                if isinstance(v, float):
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")
        else:
            print("No LMX files evaluated.")

        print("\n----- LMX Validity -----")
        for k, v in payload["lmx"]["validity"].items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    if run_xml:
        print("\n================ XML Evaluation ================\n")
        payload["xml"] = evaluate_xml_dirs(args.pred_xml_dir, args.gt_xml_dir)

        print("\n----- XML Summary -----")
        if payload["xml"]["summary"]:
            for k, v in payload["xml"]["summary"].items():
                if isinstance(v, float):
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")
        else:
            print("No XML files evaluated.")

        print("\n----- XML Validity -----")
        for k, v in payload["xml"]["validity"].items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results to: {args.save_json}")


if __name__ == "__main__":
    main()