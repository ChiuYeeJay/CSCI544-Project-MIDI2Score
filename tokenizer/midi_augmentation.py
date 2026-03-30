import symusic
from miditok.utils import merge_same_program_tracks
import numpy as np

def get_tick_per_ms_map(score: symusic.core.ScoreTick, note_times: np.ndarray) -> np.ndarray:
    if len(score.tempos) == 0: # if the tempo is not specified
        default_bpm = 120.0
        tick_per_ms = (score.tpq * default_bpm) / 60000.0
        return np.full_like(note_times, tick_per_ms, dtype=np.float64)
    
    # Extract tempo events
    score.tempos.sort(key=lambda x: x.time)
    tempo_ticks = np.array([t.time for t in score.tempos])
    tempo_bpms = np.array([t.qpm for t in score.tempos])

    # find corresbonding tempo index for each note
    idx = np.searchsorted(tempo_ticks, note_times, side='right') - 1
    idx = np.clip(idx, 0, len(tempo_ticks) - 1)
    active_bpms = tempo_bpms[idx]

    return (score.tpq * active_bpms) / 60000.0

def apply_midi_augmentation(score: symusic.core.ScoreTick, heavy_noise=False):
    score = score.copy(deep=True)
    merge_same_program_tracks(score.tracks)
    if len(score.tracks) == 0 or len(score.tracks[0].notes) == 0:
        return score

    track = score.tracks[0]
    length = score.end()

    sample_step_range = (4, 16)
    if heavy_noise:
        drift_radius = score.tpq // 4 # 16th note
        jitter_std_ms = 50 / 2 # under 50ms for most of time
        strech_range = (0.8, 1.2)
    else:
        drift_radius = score.tpq // 8 # 32th note
        jitter_std_ms = 20 / 2 # under 20ms for most of time
        strech_range = (0.9, 1.1)

    sample_step = int(np.random.uniform(*sample_step_range) * score.tpq)
    n_sample_points = max(int(length // sample_step), 2)
    sample_ticks = np.linspace(0, length, n_sample_points)
    sample_values = np.random.uniform(-drift_radius, drift_radius, size=n_sample_points)

    onsets = np.array([n.time for n in track.notes])
    drift = np.interp(onsets, sample_ticks, sample_values)
    stretch = np.random.uniform(*strech_range, size=onsets.shape)

    tick_per_ms_array = get_tick_per_ms_map(score, onsets)
    jitter = np.random.normal(0, jitter_std_ms, size=onsets.shape) * tick_per_ms_array

    last_note_ref: list[symusic.core.NoteTick | None] = [None] * 128
    for i, note in enumerate(track.notes):
        # Note start with random drift and jitter
        note.time = int(max(note.time + drift[i] + jitter[i], 0))

        prev_note = last_note_ref[note.pitch]
        if prev_note is not None and note.time < prev_note.end:
            note.time = max(prev_note.time+1, note.time)
            prev_note.duration = max(note.time-prev_note.time-1, 1)
        
        note.duration = int(max(note.duration * stretch[i], 1))
        last_note_ref[note.pitch] = note
    track.notes.sort()
    return score

if __name__ == "__main__":
    import os
    import pandas as pd
    import miditok
    import tqdm
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
        special_tokens=["PAD"],
    )

    cpword_tokenizer = miditok.CPWord(miditok_config)

    PDMX_ROOT = "../dataset/PDMX_preprocessed_rd/"
    df = pd.read_csv(os.path.join(PDMX_ROOT, "dataset_info_with_partitions.csv"))
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        # if row["n_notes"] == 0:
        #     continue
        try:
            midi_file = os.path.join(PDMX_ROOT, row["midi"])
            score = symusic.Score(midi_file)
            augmented_score = apply_midi_augmentation(score)
            # augmented_score.dump_midi("temp.mid")
            # cpword_tokenizer.encode("temp.mid")  # This will apply the augmentation
            cpword_tokenizer.encode(augmented_score)  # This will apply the augmentation

        except Exception as e:
            print(f"Error processing {row['midi']}: {e}")
            raise
            continue