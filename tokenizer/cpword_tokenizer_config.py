import miditok

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