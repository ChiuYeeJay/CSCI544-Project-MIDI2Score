import os
import glob
import subprocess
import argparse
from music21 import converter


# =========================
# MuseScore conversion
# =========================
def musescore_midi_to_musicxml(
    input_folder,
    output_folder,
    musescore_path,
    max_files=None
):
    os.makedirs(output_folder, exist_ok=True)

    midi_files = glob.glob(os.path.join(input_folder, "*.mid"))

    if not midi_files:
        print("No MIDI files found in:", input_folder)
        return

    if max_files:
        midi_files = midi_files[:max_files]

    print(f"[MuseScore] Converting {len(midi_files)} files...")

    for midi_file in midi_files:
        filename = os.path.basename(midi_file)
        name = os.path.splitext(filename)[0]
        output_file = os.path.join(output_folder, f"{name}.xml")

        try:
            subprocess.run(
                [musescore_path, midi_file, "-o", output_file],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"[OK] {filename}")

        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {filename}")
            print(e.stderr.decode(errors="ignore"))


# =========================
# music21 conversion
# =========================
def music21_midi_to_musicxml(
    input_folder,
    output_folder,
    max_files=None
):
    os.makedirs(output_folder, exist_ok=True)

    midi_files = glob.glob(os.path.join(input_folder, "*.mid"))

    if not midi_files:
        print("No MIDI files found in:", input_folder)
        return

    if max_files:
        midi_files = midi_files[:max_files]

    print(f"[music21] Converting {len(midi_files)} files...")

    for midi_file in midi_files:
        filename = os.path.basename(midi_file)
        name = os.path.splitext(filename)[0]
        output_file = os.path.join(output_folder, f"{name}.xml")

        try:
            score = converter.parse(midi_file)
            score.write("musicxml", output_file)
            print(f"[OK] {filename}")

        except Exception as e:
            print(f"[FAIL] {filename}: {e}")


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="MIDI → MusicXML converter")

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument(
        "--method",
        type=str,
        choices=["music21", "musescore"],
        required=True,
        help="Which converter to use"
    )

    parser.add_argument(
        "--musescore-path",
        type=str,
        default="C:/Program Files/MuseScore 4/bin/MuseScore4.exe",
        help="Path to MuseScore executable"
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files processed"
    )

    args = parser.parse_args()

    if args.method == "music21":
        music21_midi_to_musicxml(
            args.input,
            args.output,
            max_files=args.max_files
        )

    elif args.method == "musescore":
        musescore_midi_to_musicxml(
            args.input,
            args.output,
            args.musescore_path,
            max_files=args.max_files
        )


if __name__ == "__main__":
    main()