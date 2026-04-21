import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import partitura as pt


TIME_ROUND_DIGITS = 6
DEFAULT_FRAME_RESOLUTION = 0.25  # 16th-note grid when 1 beat = quarter note
DEFAULT_TIMELINE_DTW_BAND_FRAC = 0.10
VOICE_WILDCARDS = {None, "", "None"}


@dataclass(frozen=True)
class NoteEvent:
    part_idx: int
    measure_idx: int
    pitch: int
    pitch_class: int
    onset_beat: float
    duration_beat: float
    staff: Optional[int]
    voice: Optional[str]


# ============================================================
# generic helpers
# ============================================================


def safe_int(text: Optional[str], default: int = 0) -> int:
    if text is None:
        return default
    try:
        return int(str(text).strip())
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



def round_time(x: float) -> float:
    return round(float(x), TIME_ROUND_DIGITS)



def stem_no_ext(path: str) -> str:
    return Path(path).stem



def list_files_with_ext(folder: Optional[str], exts: Tuple[str, ...]) -> List[str]:
    if folder is None or not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])



def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        a = seq1[i - 1]
        for j in range(1, m + 1):
            b = seq2[j - 1]
            cost = 0 if a == b else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[m]



def normalized_edit_distance(seq_pred: List[str], seq_gt: List[str]) -> Dict[str, float]:
    dist = levenshtein_distance(seq_pred, seq_gt)
    denom = max(len(seq_gt), 1)
    return {
        "edit_distance": dist,
        "normalized_edit_distance": dist / denom,
    }


# ============================================================
# raw XML side: attributes / notation structure
# ============================================================


STEP_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}



def pitch_to_midi_xml(note_elem: ET.Element) -> Optional[int]:
    pitch_elem = find_child(note_elem, "pitch")
    if pitch_elem is None:
        return None

    step_elem = find_child(pitch_elem, "step")
    octave_elem = find_child(pitch_elem, "octave")
    alter_elem = find_child(pitch_elem, "alter")
    if step_elem is None or octave_elem is None or not step_elem.text or not octave_elem.text:
        return None
    step = step_elem.text.strip()
    if step not in STEP_TO_SEMITONE:
        return None
    octave = safe_int(octave_elem.text, default=0)
    alter = safe_int(alter_elem.text, default=0) if alter_elem is not None else 0
    return 12 * (octave + 1) + STEP_TO_SEMITONE[step] + alter



def extract_structure_and_attribute_events(xml_path: str) -> Dict[str, object]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    part_elems = [e for e in root.iter() if local_name(e.tag) == "part"]
    if not part_elems:
        raise ValueError(f"No <part> found in {xml_path}")

    structure_tokens: List[str] = []
    key_events: List[Tuple[int, int, int]] = []
    time_events: List[Tuple[int, int, int, int]] = []
    clef_events: List[Tuple[int, int, int, str, str]] = []
    measure_ids_all_parts: set = set()

    for part_idx, part_elem in enumerate(part_elems):
        for m_order, measure_elem in enumerate([m for m in part_elem if local_name(m.tag) == "measure"], start=1):
            raw_num = measure_elem.attrib.get("number")
            try:
                measure_idx = int(raw_num) if raw_num is not None else m_order
            except Exception:
                measure_idx = m_order
            measure_ids_all_parts.add(measure_idx)

            measure_tokens: List[str] = [f"PART_{part_idx}", f"MEASURE_{measure_idx}"]
            measure_time = 0
            last_non_chord_onset = None

            for child in measure_elem:
                tag = local_name(child.tag)
                if tag == "attributes":
                    divisions_elem = find_child(child, "divisions")
                    if divisions_elem is not None and divisions_elem.text is not None:
                        measure_tokens.append(f"DIV_{safe_int(divisions_elem.text)}")

                    time_elem = find_child(child, "time")
                    if time_elem is not None:
                        beats_elem = find_child(time_elem, "beats")
                        beat_type_elem = find_child(time_elem, "beat-type")
                        beats = safe_int(beats_elem.text if beats_elem is not None else None, default=0)
                        beat_type = safe_int(beat_type_elem.text if beat_type_elem is not None else None, default=0)
                        measure_tokens.append(f"TIME_{beats}_{beat_type}")
                        time_events.append((part_idx, measure_idx, beats, beat_type))

                    key_elem = find_child(child, "key")
                    if key_elem is not None:
                        fifths_elem = find_child(key_elem, "fifths")
                        if fifths_elem is not None and fifths_elem.text is not None:
                            fifths = safe_int(fifths_elem.text)
                            measure_tokens.append(f"KEY_{fifths}")
                            key_events.append((part_idx, measure_idx, fifths))

                    for clef in find_children(child, "clef"):
                        sign_elem = find_child(clef, "sign")
                        line_elem = find_child(clef, "line")
                        number = safe_int(clef.attrib.get("number"), default=1)
                        if sign_elem is not None and line_elem is not None and sign_elem.text and line_elem.text:
                            sign = sign_elem.text.strip()
                            line = line_elem.text.strip()
                            measure_tokens.append(f"CLEF_{number}_{sign}_{line}")
                            clef_events.append((part_idx, measure_idx, number, sign, line))

            for child in measure_elem:
                tag = local_name(child.tag)
                if tag == "note":
                    is_rest = find_child(child, "rest") is not None
                    is_chord = find_child(child, "chord") is not None
                    dur_elem = find_child(child, "duration")
                    duration = safe_int(dur_elem.text if dur_elem is not None else None, default=0)
                    staff_elem = find_child(child, "staff")
                    staff = safe_int(staff_elem.text if staff_elem is not None else None, default=0)
                    voice_elem = find_child(child, "voice")
                    voice = voice_elem.text.strip() if voice_elem is not None and voice_elem.text else ""

                    onset = last_non_chord_onset if is_chord and last_non_chord_onset is not None else measure_time
                    if not is_chord:
                        last_non_chord_onset = onset

                    if is_rest:
                        measure_tokens.append(f"REST_D{duration}_S{staff}_V{voice}")
                    else:
                        pitch = pitch_to_midi_xml(child)
                        if pitch is not None:
                            measure_tokens.append(f"NOTE_P{pitch}_D{duration}_S{staff}_V{voice}")

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

            structure_tokens.extend([f"<PART_{part_idx}_MEASURE_START>"] + measure_tokens + [f"<PART_{part_idx}_MEASURE_END>"])

    return {
        "structure_tokens": structure_tokens,
        "key_events": key_events,
        "time_events": time_events,
        "clef_events": clef_events,
        "measure_ids": sorted(measure_ids_all_parts),
    }


# ============================================================
# partitura side: note events across all parts
# ============================================================


def iter_measures_for_part(part) -> List[object]:
    try:
        if hasattr(part, "measures") and part.measures is not None:
            return list(part.measures)
    except Exception:
        pass
    try:
        import partitura.score as ptscore
        return list(part.iter_all(ptscore.Measure))
    except Exception:
        return []



def get_measure_spans(part) -> List[Tuple[int, float, float]]:
    spans: List[Tuple[int, float, float]] = []
    for i, m in enumerate(iter_measures_for_part(part), start=1):
        start_obj = getattr(m, "start", None)
        end_obj = getattr(m, "end", None)
        if start_obj is None or end_obj is None:
            continue
        number = getattr(m, "number", None)
        try:
            measure_idx = int(number) if number is not None else i
        except Exception:
            measure_idx = i
        spans.append((measure_idx, float(start_obj.t), float(end_obj.t)))
    spans.sort(key=lambda x: (x[1], x[2], x[0]))
    return spans



def find_measure_idx_for_time(start_div: float, measure_spans: List[Tuple[int, float, float]]) -> int:
    if not measure_spans:
        return 0
    for measure_idx, m_start, m_end in measure_spans:
        if m_start <= start_div < m_end:
            return measure_idx
    if math.isclose(start_div, measure_spans[-1][2]):
        return measure_spans[-1][0]
    return 0



def parse_musicxml_file(xml_path: str) -> Dict[str, object]:
    score = pt.load_score(xml_path)
    if not score.parts:
        raise ValueError(f"No part found in {xml_path}")

    structure_data = extract_structure_and_attribute_events(xml_path)

    all_notes: List[NoteEvent] = []
    notes_before_tie_merge = 0
    notes_by_measure: Dict[int, List[NoteEvent]] = defaultdict(list)

    all_measure_ids = set(structure_data["measure_ids"])

    for part_idx, part in enumerate(score.parts):
        beat_map = part.beat_map
        measure_spans = get_measure_spans(part)
        all_measure_ids.update(m for m, _, _ in measure_spans)

        raw_notes = []
        for n in part.notes:
            notes_before_tie_merge += 1
            start_obj = getattr(n, "start", None)
            end_obj = getattr(n, "end", None)
            if start_obj is None or end_obj is None:
                continue
            start_div = float(start_obj.t)
            end_div = float(end_obj.t)
            if end_div < start_div:
                continue
            pitch = int(n.midi_pitch)
            staff = getattr(n, "staff", None)
            voice_raw = getattr(n, "voice", None)
            voice = str(voice_raw) if voice_raw is not None else None

            tie_group = getattr(n, "tie_group", None)
            note_id = getattr(n, "id", None)
            if tie_group is not None:
                tie_key = f"part{part_idx}::tie::{tie_group}"
            elif note_id is not None:
                tie_key = f"part{part_idx}::note::{note_id}"
            else:
                tie_key = f"part{part_idx}::free::{pitch}::{start_div}::{end_div}::{voice}::{staff}"

            raw_notes.append({
                "tie_key": tie_key,
                "pitch": pitch,
                "start_div": start_div,
                "end_div": end_div,
                "staff": staff,
                "voice": voice,
            })

        groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for item in raw_notes:
            groups[item["tie_key"]].append(item)

        for _, group in groups.items():
            group = sorted(group, key=lambda x: (x["start_div"], x["end_div"]))
            first = group[0]
            start_div = min(float(x["start_div"]) for x in group)
            end_div = max(float(x["end_div"]) for x in group)
            onset_beat = round_time(beat_map(start_div))
            end_beat = round_time(beat_map(end_div))
            duration_beat = round_time(max(0.0, end_beat - onset_beat))
            measure_idx = find_measure_idx_for_time(start_div, measure_spans)

            evt = NoteEvent(
                part_idx=part_idx,
                measure_idx=measure_idx,
                pitch=int(first["pitch"]),
                pitch_class=int(first["pitch"]) % 12,
                onset_beat=onset_beat,
                duration_beat=duration_beat,
                staff=first["staff"],
                voice=first["voice"],
            )
            all_notes.append(evt)
            notes_by_measure[measure_idx].append(evt)

    all_notes.sort(key=lambda n: (n.measure_idx, n.part_idx, n.onset_beat, n.pitch, n.duration_beat, str(n.staff), str(n.voice)))

    measure_ids = sorted([m for m in all_measure_ids if m >= 0])
    measure_note_keys_staff: List[List[Tuple[int, int, float, float, Optional[int]]]] = []
    measure_note_keys_basic: List[List[Tuple[int, int, float, float]]] = []
    measure_rhythm_keys: List[List[Tuple[int, float, float]]] = []
    measure_pitch_class_keys: List[List[Tuple[int, int, float, float]]] = []

    for measure_idx in measure_ids:
        cur_notes = sorted(
            notes_by_measure.get(measure_idx, []),
            key=lambda n: (n.part_idx, n.onset_beat, n.pitch, n.duration_beat, str(n.staff), str(n.voice)),
        )
        cur_staff = []
        cur_basic = []
        cur_rhythm = []
        cur_pc = []
        for note in cur_notes:
            q_on = round_time(note.onset_beat)
            q_dur = round_time(note.duration_beat)
            cur_staff.append((note.part_idx, note.pitch, q_on, q_dur, note.staff))
            cur_basic.append((note.part_idx, note.pitch, q_on, q_dur))
            cur_rhythm.append((note.part_idx, q_on, q_dur))
            cur_pc.append((note.part_idx, note.pitch_class, q_on, q_dur))
        measure_note_keys_staff.append(cur_staff)
        measure_note_keys_basic.append(cur_basic)
        measure_rhythm_keys.append(cur_rhythm)
        measure_pitch_class_keys.append(cur_pc)

    return {
        "notes": all_notes,
        "measure_ids": measure_ids,
        "measure_note_keys_staff": measure_note_keys_staff,
        "measure_note_keys_basic": measure_note_keys_basic,
        "measure_rhythm_keys": measure_rhythm_keys,
        "measure_pitch_class_keys": measure_pitch_class_keys,
        "structure_tokens": structure_data["structure_tokens"],
        "key_events": structure_data["key_events"],
        "time_events": structure_data["time_events"],
        "clef_events": structure_data["clef_events"],
        "num_measures": len(measure_ids),
        "notes_before_tie_merge": notes_before_tie_merge,
        "notes_after_tie_merge": len(all_notes),
        "num_tie_merged_notes": max(0, notes_before_tie_merge - len(all_notes)),
    }


# ============================================================
# metrics
# ============================================================


def normalize_voice(voice: Optional[str]) -> Optional[str]:
    if voice in VOICE_WILDCARDS:
        return None
    return str(voice)



def counter_from_notes(notes: List[NoteEvent], key_mode: str) -> Counter:
    keys = []
    for n in notes:
        if key_mode == "pitch":
            keys.append((n.part_idx, n.pitch))
        elif key_mode == "pitch_class":
            keys.append((n.part_idx, n.pitch_class))
        elif key_mode == "measure_pitch_class":
            keys.append((n.measure_idx, n.part_idx, n.pitch_class))
        elif key_mode == "measure_pitch":
            keys.append((n.measure_idx, n.part_idx, n.pitch))
        else:
            raise ValueError(f"Unsupported counter key_mode: {key_mode}")
    return Counter(keys)



def precision_recall_f1_from_counters(pred_counter: Counter, gt_counter: Counter) -> Dict[str, float]:
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
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}



def notes_match_threshold(
    pred: NoteEvent,
    gt: NoteEvent,
    *,
    key_mode: str,
    onset_tol: float = 0.0,
    duration_tol: float = 0.0,
) -> bool:
    if pred.part_idx != gt.part_idx:
        return False

    pred_voice = normalize_voice(pred.voice)
    gt_voice = normalize_voice(gt.voice)

    if key_mode == "pitch":
        return pred.pitch == gt.pitch

    if key_mode == "pitch_class":
        return pred.pitch_class == gt.pitch_class

    if key_mode == "pitch_onset":
        return pred.pitch == gt.pitch and abs(pred.onset_beat - gt.onset_beat) <= onset_tol

    if key_mode == "pitch_onset_duration":
        return (
            pred.pitch == gt.pitch
            and abs(pred.onset_beat - gt.onset_beat) <= onset_tol
            and abs(pred.duration_beat - gt.duration_beat) <= duration_tol
        )

    if key_mode == "pitch_onset_duration_staff":
        return (
            pred.pitch == gt.pitch
            and pred.staff == gt.staff
            and abs(pred.onset_beat - gt.onset_beat) <= onset_tol
            and abs(pred.duration_beat - gt.duration_beat) <= duration_tol
        )

    if key_mode == "pitch_onset_duration_voice":
        return (
            pred.pitch == gt.pitch
            and pred_voice == gt_voice
            and abs(pred.onset_beat - gt.onset_beat) <= onset_tol
            and abs(pred.duration_beat - gt.duration_beat) <= duration_tol
        )

    if key_mode == "rhythm":
        return (
            abs(pred.onset_beat - gt.onset_beat) <= onset_tol
            and abs(pred.duration_beat - gt.duration_beat) <= duration_tol
        )

    if key_mode == "pitch_class_onset_duration":
        return (
            pred.pitch_class == gt.pitch_class
            and abs(pred.onset_beat - gt.onset_beat) <= onset_tol
            and abs(pred.duration_beat - gt.duration_beat) <= duration_tol
        )

    raise ValueError(f"Unsupported threshold key_mode: {key_mode}")



def note_match_cost(pred: NoteEvent, gt: NoteEvent, *, key_mode: str) -> float:
    onset_diff = abs(pred.onset_beat - gt.onset_beat)
    dur_diff = abs(pred.duration_beat - gt.duration_beat)
    pitch_diff = abs(pred.pitch - gt.pitch) / 128.0
    pc_diff = 0.0 if pred.pitch_class == gt.pitch_class else 1.0

    if key_mode == "pitch":
        return pitch_diff + onset_diff * 0.1 + dur_diff * 0.1
    if key_mode == "pitch_class":
        return pc_diff + onset_diff * 0.1 + dur_diff * 0.1
    if key_mode == "pitch_onset":
        return onset_diff + pitch_diff * 0.05
    if key_mode in {"pitch_onset_duration", "pitch_onset_duration_staff", "pitch_onset_duration_voice"}:
        return onset_diff + dur_diff + pitch_diff * 0.05
    if key_mode == "rhythm":
        return onset_diff + dur_diff
    if key_mode == "pitch_class_onset_duration":
        return onset_diff + dur_diff + pc_diff * 0.05
    return onset_diff + dur_diff



def precision_recall_f1_threshold_match(
    pred_notes: List[NoteEvent],
    gt_notes: List[NoteEvent],
    *,
    key_mode: str,
    onset_tol: float = 0.0,
    duration_tol: float = 0.0,
) -> Dict[str, float]:
    candidate_pairs: List[Tuple[float, int, int]] = []

    for i, pred in enumerate(pred_notes):
        for j, gt in enumerate(gt_notes):
            if notes_match_threshold(
                pred,
                gt,
                key_mode=key_mode,
                onset_tol=onset_tol,
                duration_tol=duration_tol,
            ):
                candidate_pairs.append((note_match_cost(pred, gt, key_mode=key_mode), i, j))

    candidate_pairs.sort(key=lambda x: x[0])

    matched_pred = set()
    matched_gt = set()
    tp = 0

    for _, i, j in candidate_pairs:
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(i)
        matched_gt.add(j)
        tp += 1

    total_pred = len(pred_notes)
    total_gt = len(gt_notes)
    fp = total_pred - tp
    fn = total_gt - tp

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}



def precision_recall_f1_from_mode(
    pred_notes: List[NoteEvent],
    gt_notes: List[NoteEvent],
    key_mode: str,
    onset_tol: float = 0.0,
    duration_tol: float = 0.0,
) -> Dict[str, float]:
    if key_mode in {"pitch", "pitch_class"}:
        pred_counter = counter_from_notes(pred_notes, key_mode)
        gt_counter = counter_from_notes(gt_notes, key_mode)
        return precision_recall_f1_from_counters(pred_counter, gt_counter)

    return precision_recall_f1_threshold_match(
        pred_notes,
        gt_notes,
        key_mode=key_mode,
        onset_tol=onset_tol,
        duration_tol=duration_tol,
    )



def measure_accuracy_and_ser(pred_measures: List[List[Tuple]], gt_measures: List[List[Tuple]]) -> Dict[str, float]:
    total = max(len(pred_measures), len(gt_measures))
    if total == 0:
        return {"measure_accuracy": 0.0, "ser": 0.0, "correct_measures": 0, "wrong_measures": 0, "total_measures": 0}
    correct = 0
    for i in range(total):
        pred_notes = pred_measures[i] if i < len(pred_measures) else []
        gt_notes = gt_measures[i] if i < len(gt_measures) else []
        if Counter(pred_notes) == Counter(gt_notes):
            correct += 1
    wrong = total - correct
    return {
        "measure_accuracy": correct / total,
        "ser": wrong / total,
        "correct_measures": correct,
        "wrong_measures": wrong,
        "total_measures": total,
    }



def average_measure_f1(pred_measures: List[List[Tuple]], gt_measures: List[List[Tuple]]) -> Dict[str, float]:
    total = max(len(pred_measures), len(gt_measures))
    if total == 0:
        return {"avg_measure_f1": 0.0}
    per_measure_f1 = []
    for i in range(total):
        pred_notes = pred_measures[i] if i < len(pred_measures) else []
        gt_notes = gt_measures[i] if i < len(gt_measures) else []
        scores = precision_recall_f1_from_counters(Counter(pred_notes), Counter(gt_notes))
        per_measure_f1.append(scores["f1"])
    return {"avg_measure_f1": sum(per_measure_f1) / len(per_measure_f1)}



def average_measure_soft_f1(
    pred_data: Dict[str, object],
    gt_data: Dict[str, object],
    *,
    key_mode: str,
    onset_tol: float,
    duration_tol: float,
) -> Dict[str, float]:
    pred_map: Dict[int, List[NoteEvent]] = defaultdict(list)
    gt_map: Dict[int, List[NoteEvent]] = defaultdict(list)
    for n in pred_data["notes"]:
        pred_map[n.measure_idx].append(n)
    for n in gt_data["notes"]:
        gt_map[n.measure_idx].append(n)

    all_measure_ids = sorted(set(pred_data["measure_ids"]) | set(gt_data["measure_ids"]))
    if not all_measure_ids:
        return {"avg_measure_soft_f1": 0.0}

    vals = []
    for m in all_measure_ids:
        scores = precision_recall_f1_from_mode(
            pred_map.get(m, []),
            gt_map.get(m, []),
            key_mode=key_mode,
            onset_tol=onset_tol,
            duration_tol=duration_tol,
        )
        vals.append(scores["f1"])
    return {"avg_measure_soft_f1": sum(vals) / len(vals)}



def event_accuracy(pred_events: List[Tuple], gt_events: List[Tuple]) -> Dict[str, float]:
    scores = precision_recall_f1_from_counters(Counter(pred_events), Counter(gt_events))
    return {
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": scores["f1"],
    }



def js_divergence_from_counters(pred_counter: Counter, gt_counter: Counter) -> float:
    keys = set(pred_counter.keys()) | set(gt_counter.keys())
    if not keys:
        return 0.0
    pred_total = sum(pred_counter.values())
    gt_total = sum(gt_counter.values())
    if pred_total == 0 and gt_total == 0:
        return 0.0
    p = {k: pred_counter.get(k, 0) / pred_total if pred_total > 0 else 0.0 for k in keys}
    q = {k: gt_counter.get(k, 0) / gt_total if gt_total > 0 else 0.0 for k in keys}
    m = {k: 0.5 * (p[k] + q[k]) for k in keys}

    def kl(a: Dict, b: Dict) -> float:
        out = 0.0
        for k in keys:
            if a[k] > 0 and b[k] > 0:
                out += a[k] * math.log2(a[k] / b[k])
        return out

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)



def pitch_class_histogram_distance(pred_notes: List[NoteEvent], gt_notes: List[NoteEvent]) -> float:
    pred_counter = Counter([n.pitch_class for n in pred_notes])
    gt_counter = Counter([n.pitch_class for n in gt_notes])
    return js_divergence_from_counters(pred_counter, gt_counter)



def duration_histogram_distance(pred_notes: List[NoteEvent], gt_notes: List[NoteEvent]) -> float:
    pred_counter = Counter([round_time(n.duration_beat) for n in pred_notes])
    gt_counter = Counter([round_time(n.duration_beat) for n in gt_notes])
    return js_divergence_from_counters(pred_counter, gt_counter)



def build_pitch_class_timeline(notes: List[NoteEvent], frame_resolution: float = DEFAULT_FRAME_RESOLUTION) -> List[List[int]]:
    if not notes:
        return []
    max_end = max(n.onset_beat + max(n.duration_beat, frame_resolution) for n in notes)
    num_frames = max(1, int(math.ceil(max_end / frame_resolution)))
    timeline = [[0] * 12 for _ in range(num_frames)]
    for n in notes:
        start_idx = max(0, int(math.floor(n.onset_beat / frame_resolution)))
        end_idx = max(start_idx + 1, int(math.ceil((n.onset_beat + max(n.duration_beat, frame_resolution)) / frame_resolution)))
        for idx in range(start_idx, min(end_idx, num_frames)):
            timeline[idx][n.pitch_class] = 1
    return timeline



def build_rhythm_timeline(notes: List[NoteEvent], frame_resolution: float = DEFAULT_FRAME_RESOLUTION) -> List[List[int]]:
    if not notes:
        return []
    max_end = max(n.onset_beat + max(n.duration_beat, frame_resolution) for n in notes)
    num_frames = max(1, int(math.ceil(max_end / frame_resolution)))
    timeline = [[0, 0] for _ in range(num_frames)]  # [onset_activity, sustain_activity]
    for n in notes:
        start_idx = max(0, int(math.floor(n.onset_beat / frame_resolution)))
        end_idx = max(start_idx + 1, int(math.ceil((n.onset_beat + max(n.duration_beat, frame_resolution)) / frame_resolution)))
        if start_idx < num_frames:
            timeline[start_idx][0] = 1
        for idx in range(start_idx, min(end_idx, num_frames)):
            timeline[idx][1] = 1
    return timeline



def cosine_similarity(vec_a: List[int], vec_b: List[int]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    na = math.sqrt(sum(a * a for a in vec_a))
    nb = math.sqrt(sum(b * b for b in vec_b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)



def framewise_f1(binary_seq_pred: List[List[int]], binary_seq_gt: List[List[int]]) -> float:
    length = max(len(binary_seq_pred), len(binary_seq_gt))
    if length == 0:
        return 0.0
    tp = fp = fn = 0
    width = len(binary_seq_pred[0]) if binary_seq_pred else (len(binary_seq_gt[0]) if binary_seq_gt else 0)
    zero = [0] * width
    for i in range(length):
        p = binary_seq_pred[i] if i < len(binary_seq_pred) else zero
        g = binary_seq_gt[i] if i < len(binary_seq_gt) else zero
        for pv, gv in zip(p, g):
            if pv and gv:
                tp += 1
            elif pv and not gv:
                fp += 1
            elif gv and not pv:
                fn += 1
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0



def dtw_average_similarity(seq_pred: List[List[int]], seq_gt: List[List[int]], band_frac: float = DEFAULT_TIMELINE_DTW_BAND_FRAC) -> float:
    n = len(seq_pred)
    m = len(seq_gt)
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0

    band = max(abs(n - m), int(math.ceil(max(n, m) * band_frac)))
    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    steps = [[0] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m, i + band)
        for j in range(j_start, j_end + 1):
            sim = cosine_similarity(seq_pred[i - 1], seq_gt[j - 1])
            cost = 1.0 - sim
            prev_options = [
                (dp[i - 1][j], steps[i - 1][j]),
                (dp[i][j - 1], steps[i][j - 1]),
                (dp[i - 1][j - 1], steps[i - 1][j - 1]),
            ]
            prev_cost, prev_steps = min(prev_options, key=lambda x: x[0])
            dp[i][j] = prev_cost + cost
            steps[i][j] = prev_steps + 1

    if math.isinf(dp[n][m]) or steps[n][m] == 0:
        return 0.0
    avg_cost = dp[n][m] / steps[n][m]
    return max(0.0, 1.0 - avg_cost)



def timeline_similarity_scores(pred_notes: List[NoteEvent], gt_notes: List[NoteEvent]) -> Dict[str, float]:
    pred_pc = build_pitch_class_timeline(pred_notes, DEFAULT_FRAME_RESOLUTION)
    gt_pc = build_pitch_class_timeline(gt_notes, DEFAULT_FRAME_RESOLUTION)
    pred_rhythm = build_rhythm_timeline(pred_notes, DEFAULT_FRAME_RESOLUTION)
    gt_rhythm = build_rhythm_timeline(gt_notes, DEFAULT_FRAME_RESOLUTION)

    pc_frame_f1 = framewise_f1(pred_pc, gt_pc)
    rhythm_frame_f1 = framewise_f1(pred_rhythm, gt_rhythm)
    pc_dtw = dtw_average_similarity(pred_pc, gt_pc)
    rhythm_dtw = dtw_average_similarity(pred_rhythm, gt_rhythm)

    return {
        "pitch_class_timeline_f1": pc_frame_f1,
        "rhythm_timeline_f1": rhythm_frame_f1,
        "pitch_class_timeline_dtw_similarity": pc_dtw,
        "rhythm_timeline_dtw_similarity": rhythm_dtw,
    }


# ============================================================
# top-level pair / aggregate
# ============================================================


def evaluate_xml_pair(pred_path: str, gt_path: str, onset_tol: float = 0.0, duration_tol: float = 0.0) -> Dict[str, object]:
    pred_data = parse_musicxml_file(pred_path)
    gt_data = parse_musicxml_file(gt_path)

    pitch_scores = precision_recall_f1_from_mode(pred_data["notes"], gt_data["notes"], "pitch")
    pitch_class_scores = precision_recall_f1_from_mode(pred_data["notes"], gt_data["notes"], "pitch_class")
    po_scores = precision_recall_f1_from_mode(pred_data["notes"], gt_data["notes"], "pitch_onset", onset_tol=onset_tol)
    pod_scores = precision_recall_f1_from_mode(pred_data["notes"], gt_data["notes"], "pitch_onset_duration", onset_tol=onset_tol, duration_tol=duration_tol)
    pods_scores = precision_recall_f1_from_mode(pred_data["notes"], gt_data["notes"], "pitch_onset_duration_staff", onset_tol=onset_tol, duration_tol=duration_tol)
    podv_scores = precision_recall_f1_from_mode(pred_data["notes"], gt_data["notes"], "pitch_onset_duration_voice", onset_tol=onset_tol, duration_tol=duration_tol)
    rhythm_scores = precision_recall_f1_from_mode(pred_data["notes"], gt_data["notes"], "rhythm", onset_tol=onset_tol, duration_tol=duration_tol)
    pcd_scores = precision_recall_f1_from_mode(pred_data["notes"], gt_data["notes"], "pitch_class_onset_duration", onset_tol=onset_tol, duration_tol=duration_tol)

    strict_measure_scores = measure_accuracy_and_ser(pred_data["measure_note_keys_staff"], gt_data["measure_note_keys_staff"])
    soft_measure_scores = average_measure_f1(pred_data["measure_note_keys_basic"], gt_data["measure_note_keys_basic"])
    rhythm_measure_scores = average_measure_f1(pred_data["measure_rhythm_keys"], gt_data["measure_rhythm_keys"])
    pc_measure_scores = average_measure_f1(pred_data["measure_pitch_class_keys"], gt_data["measure_pitch_class_keys"])
    soft_measure_note_scores = average_measure_soft_f1(pred_data, gt_data, key_mode="pitch_onset_duration", onset_tol=onset_tol, duration_tol=duration_tol)
    soft_measure_pc_scores = average_measure_soft_f1(pred_data, gt_data, key_mode="pitch_class_onset_duration", onset_tol=onset_tol, duration_tol=duration_tol)

    struct_scores = normalized_edit_distance(pred_data["structure_tokens"], gt_data["structure_tokens"])
    key_scores = event_accuracy(pred_data["key_events"], gt_data["key_events"])
    time_scores = event_accuracy(pred_data["time_events"], gt_data["time_events"])
    clef_scores = event_accuracy(pred_data["clef_events"], gt_data["clef_events"])
    timeline_scores = timeline_similarity_scores(pred_data["notes"], gt_data["notes"])

    music_content_score = (
        0.30 * pitch_class_scores["f1"]
        + 0.25 * rhythm_scores["f1"]
        + 0.20 * pcd_scores["f1"]
        + 0.15 * timeline_scores["pitch_class_timeline_dtw_similarity"]
        + 0.10 * timeline_scores["rhythm_timeline_dtw_similarity"]
    )
    notation_score = (
        0.35 * pods_scores["f1"]
        + 0.25 * strict_measure_scores["measure_accuracy"]
        + 0.15 * key_scores["f1"]
        + 0.15 * time_scores["f1"]
        + 0.10 * max(0.0, 1.0 - struct_scores["normalized_edit_distance"] / 10.0)
    )

    return {
        "file": os.path.basename(gt_path),
        "num_pred_notes": len(pred_data["notes"]),
        "num_gt_notes": len(gt_data["notes"]),
        "num_pred_measures": pred_data["num_measures"],
        "num_gt_measures": gt_data["num_measures"],
        "pred_notes_before_tie_merge": pred_data["notes_before_tie_merge"],
        "gt_notes_before_tie_merge": gt_data["notes_before_tie_merge"],
        "pred_notes_after_tie_merge": pred_data["notes_after_tie_merge"],
        "gt_notes_after_tie_merge": gt_data["notes_after_tie_merge"],
        "pred_tie_merged_notes": pred_data["num_tie_merged_notes"],
        "gt_tie_merged_notes": gt_data["num_tie_merged_notes"],
        "onset_tolerance_beats": onset_tol,
        "duration_tolerance_beats": duration_tol,

        "pitch_precision": pitch_scores["precision"],
        "pitch_recall": pitch_scores["recall"],
        "pitch_f1": pitch_scores["f1"],
        "pitch_class_precision": pitch_class_scores["precision"],
        "pitch_class_recall": pitch_class_scores["recall"],
        "pitch_class_f1": pitch_class_scores["f1"],

        "pitch_onset_precision": po_scores["precision"],
        "pitch_onset_recall": po_scores["recall"],
        "pitch_onset_f1": po_scores["f1"],
        "pitch_onset_duration_precision": pod_scores["precision"],
        "pitch_onset_duration_recall": pod_scores["recall"],
        "pitch_onset_duration_f1": pod_scores["f1"],
        "pitch_onset_duration_staff_precision": pods_scores["precision"],
        "pitch_onset_duration_staff_recall": pods_scores["recall"],
        "pitch_onset_duration_staff_f1": pods_scores["f1"],
        "pitch_onset_duration_voice_precision": podv_scores["precision"],
        "pitch_onset_duration_voice_recall": podv_scores["recall"],
        "pitch_onset_duration_voice_f1": podv_scores["f1"],

        "rhythm_precision": rhythm_scores["precision"],
        "rhythm_recall": rhythm_scores["recall"],
        "rhythm_f1": rhythm_scores["f1"],
        "pitch_class_onset_duration_precision": pcd_scores["precision"],
        "pitch_class_onset_duration_recall": pcd_scores["recall"],
        "pitch_class_onset_duration_f1": pcd_scores["f1"],

        "pitch_onset_tp": po_scores["tp"],
        "pitch_onset_duration_tp": pod_scores["tp"],
        "rhythm_tp": rhythm_scores["tp"],
        "pitch_onset_duration_fn": pod_scores["fn"],

        "measure_accuracy": strict_measure_scores["measure_accuracy"],
        "ser": strict_measure_scores["ser"],
        "correct_measures": strict_measure_scores["correct_measures"],
        "wrong_measures": strict_measure_scores["wrong_measures"],
        "total_measures": strict_measure_scores["total_measures"],
        "avg_measure_note_f1": soft_measure_scores["avg_measure_f1"],
        "avg_measure_rhythm_f1": rhythm_measure_scores["avg_measure_f1"],
        "avg_measure_pitch_class_f1": pc_measure_scores["avg_measure_f1"],
        "avg_measure_soft_note_f1": soft_measure_note_scores["avg_measure_soft_f1"],
        "avg_measure_soft_pitch_class_f1": soft_measure_pc_scores["avg_measure_soft_f1"],

        "key_event_f1": key_scores["f1"],
        "time_event_f1": time_scores["f1"],
        "clef_event_f1": clef_scores["f1"],
        "xml_structure_edit_distance": struct_scores["edit_distance"],
        "xml_normalized_structure_edit_distance": struct_scores["normalized_edit_distance"],
        "pitch_class_histogram_jsd": pitch_class_histogram_distance(pred_data["notes"], gt_data["notes"]),
        "duration_histogram_jsd": duration_histogram_distance(pred_data["notes"], gt_data["notes"]),
        "pitch_class_timeline_f1": timeline_scores["pitch_class_timeline_f1"],
        "rhythm_timeline_f1": timeline_scores["rhythm_timeline_f1"],
        "pitch_class_timeline_dtw_similarity": timeline_scores["pitch_class_timeline_dtw_similarity"],
        "rhythm_timeline_dtw_similarity": timeline_scores["rhythm_timeline_dtw_similarity"],
        "music_content_score": music_content_score,
        "notation_score": notation_score,
    }



def _avg(file_results: List[Dict[str, object]], key: str) -> float:
    return sum(float(r[key]) for r in file_results) / len(file_results) if file_results else 0.0



def aggregate_xml_results(file_results: List[Dict[str, object]]) -> Dict[str, object]:
    if not file_results:
        return {}

    total_correct_measures = sum(int(r["correct_measures"]) for r in file_results)
    total_measures = sum(int(r["total_measures"]) for r in file_results)
    total_wrong_measures = sum(int(r["wrong_measures"]) for r in file_results)

    return {
        "num_files": len(file_results),
        "pitch_f1": _avg(file_results, "pitch_f1"),
        "pitch_class_f1": _avg(file_results, "pitch_class_f1"),
        "pitch_onset_f1": _avg(file_results, "pitch_onset_f1"),
        "pitch_onset_duration_f1": _avg(file_results, "pitch_onset_duration_f1"),
        "pitch_onset_duration_staff_f1": _avg(file_results, "pitch_onset_duration_staff_f1"),
        "pitch_onset_duration_voice_f1": _avg(file_results, "pitch_onset_duration_voice_f1"),
        "rhythm_f1": _avg(file_results, "rhythm_f1"),
        "pitch_class_onset_duration_f1": _avg(file_results, "pitch_class_onset_duration_f1"),
        "strict_measure_accuracy": (total_correct_measures / total_measures) if total_measures > 0 else 0.0,
        "ser": (total_wrong_measures / total_measures) if total_measures > 0 else 0.0,
        "avg_measure_note_f1": _avg(file_results, "avg_measure_note_f1"),
        "avg_measure_rhythm_f1": _avg(file_results, "avg_measure_rhythm_f1"),
        "avg_measure_pitch_class_f1": _avg(file_results, "avg_measure_pitch_class_f1"),
        "avg_measure_soft_note_f1": _avg(file_results, "avg_measure_soft_note_f1"),
        "avg_measure_soft_pitch_class_f1": _avg(file_results, "avg_measure_soft_pitch_class_f1"),
        "key_event_f1": _avg(file_results, "key_event_f1"),
        "time_event_f1": _avg(file_results, "time_event_f1"),
        "clef_event_f1": _avg(file_results, "clef_event_f1"),
        "avg_xml_normalized_structure_edit_distance": _avg(file_results, "xml_normalized_structure_edit_distance"),
        "avg_pred_tie_merged_notes": _avg(file_results, "pred_tie_merged_notes"),
        "avg_gt_tie_merged_notes": _avg(file_results, "gt_tie_merged_notes"),
        "pitch_class_histogram_jsd": _avg(file_results, "pitch_class_histogram_jsd"),
        "duration_histogram_jsd": _avg(file_results, "duration_histogram_jsd"),
        "pitch_class_timeline_f1": _avg(file_results, "pitch_class_timeline_f1"),
        "rhythm_timeline_f1": _avg(file_results, "rhythm_timeline_f1"),
        "pitch_class_timeline_dtw_similarity": _avg(file_results, "pitch_class_timeline_dtw_similarity"),
        "rhythm_timeline_dtw_similarity": _avg(file_results, "rhythm_timeline_dtw_similarity"),
        "music_content_score": _avg(file_results, "music_content_score"),
        "notation_score": _avg(file_results, "notation_score"),
        "onset_tolerance_beats": _avg(file_results, "onset_tolerance_beats"),
        "duration_tolerance_beats": _avg(file_results, "duration_tolerance_beats"),
    }


# ============================================================
# grouped evaluation
# ============================================================


def load_manifest_variants(manifest_jsonl: Optional[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not manifest_jsonl:
        return mapping
    with open(manifest_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            variant = obj.get("selected_variant") or obj.get("variant") or obj.get("noise")
            if variant is None:
                continue
            for key in [obj.get("id"), stem_no_ext(obj.get("truncated_musicxml_path", "")), stem_no_ext(obj.get("truncated_lmx_path", ""))]:
                if key:
                    mapping[str(key)] = str(variant)
    return mapping



def infer_variant_for_file(filename: str, variant_map: Dict[str, str]) -> str:
    stem = stem_no_ext(filename)
    return variant_map.get(stem, variant_map.get(filename, "unknown"))


# ============================================================
# directory evaluation
# ============================================================


def evaluate_xml_dirs(
    pred_dir: str,
    gt_dir: str,
    manifest_jsonl: Optional[str] = None,
    onset_tol: float = 0.0,
    duration_tol: float = 0.0,
) -> Dict[str, object]:
    gt_files = list_files_with_ext(gt_dir, (".xml", ".musicxml"))
    file_results = []
    missing_preds = []
    parse_failures = []

    variant_map = load_manifest_variants(manifest_jsonl)
    grouped_results: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for filename in gt_files:
        gt_path = os.path.join(gt_dir, filename)
        pred_path = os.path.join(pred_dir, filename)
        if not os.path.exists(pred_path):
            missing_preds.append(filename)
            continue
        try:
            result = evaluate_xml_pair(pred_path, gt_path, onset_tol=onset_tol, duration_tol=duration_tol)
            result["group"] = infer_variant_for_file(filename, variant_map)
            file_results.append(result)
            grouped_results[result["group"]].append(result)
            print(
                f"[XML][OK] {filename} | "
                f"PitchF1={result['pitch_f1']:.4f} | "
                f"PODS={result['pitch_onset_duration_staff_f1']:.4f} | "
                f"RhythmF1={result['rhythm_f1']:.4f} | "
                f"PC-DTW={result['pitch_class_timeline_dtw_similarity']:.4f} | "
                f"MusicContent={result['music_content_score']:.4f}"
            )
        except Exception as e:
            parse_failures.append({"file": filename, "error": str(e)})
            print(f"[XML][ERROR] {filename}: {e}")

    summary = aggregate_xml_results(file_results)
    grouped_summary = {group: aggregate_xml_results(rows) for group, rows in grouped_results.items()}
    if file_results:
        grouped_summary.setdefault("overall", summary)

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
        "grouped_summary": grouped_summary,
        "per_file": file_results,
        "missing_predictions": missing_preds,
        "failures": parse_failures,
        "validity": validity,
    }


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Flexible MusicXML evaluation for MIDI2Score (partitura-based, multi-part aware, grouped eval support).")
    parser.add_argument("--pred_xml_dir", type=str, required=True)
    parser.add_argument("--gt_xml_dir", type=str, required=True)
    parser.add_argument("--save_json", type=str, default=None)
    parser.add_argument("--manifest_jsonl", type=str, default=None, help="Optional JSONL manifest for grouped clean/light/heavy summaries.")
    parser.add_argument("--onset_tol", type=float, default=0.0, help="Onset tolerance in beats for threshold note matching.")
    parser.add_argument("--duration_tol", type=float, default=0.0, help="Duration tolerance in beats for threshold note matching.")
    args = parser.parse_args()

    print("\n================ XML Evaluation ================\n")
    payload = {
        "xml": evaluate_xml_dirs(
            args.pred_xml_dir,
            args.gt_xml_dir,
            manifest_jsonl=args.manifest_jsonl,
            onset_tol=args.onset_tol,
            duration_tol=args.duration_tol,
        )
    }

    print("\n----- XML Summary -----")
    if payload["xml"]["summary"]:
        for k, v in payload["xml"]["summary"].items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
    else:
        print("No XML files evaluated.")

    if payload["xml"].get("grouped_summary"):
        print("\n----- XML Grouped Summary -----")
        for group, summary in payload["xml"]["grouped_summary"].items():
            print(f"[{group}]")
            for k, v in summary.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

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
