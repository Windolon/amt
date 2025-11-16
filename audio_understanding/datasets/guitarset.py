from __future__ import annotations

from pathlib import Path
import random
from typing import Optional

import librosa
import pandas as pd
from audidata.io.audio import load
from audidata.io.crops import RandomCrop
from audidata.io.midi import read_single_track_midi, clip_notes
from audidata.transforms.audio import Mono
from audidata.transforms.midi import PianoRoll
from audidata.utils import call
from torch.utils.data import Dataset
from typing_extensions import Literal


class GuitarSet(Dataset):
    # URL =

    # DURATION =

    def __init__(
        self,
        root: str,
        split: str = Literal["train", "validation", "test"],
        sr: float = 44100,
        crop: Optional[callable] = RandomCrop(clip_duration=10.0, end_pad=9.9),
        transform: Optional[callable] = Mono(),
        load_target: bool = True,
        extend_pedal: bool = True,
        target_transform: Optional[callable] = PianoRoll(fps=100, pitches_num=128),
    ) -> None:

        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.load_target = load_target
        self.extend_pedal = extend_pedal
        self.transform = transform
        self.target_transform = target_transform

        meta_csv = Path(self.root, "metadata.csv")

        self.meta_dict = self.load_meta(meta_csv)

    def __getitem__(self, index: int) -> dict:

        audio_path = self.meta_dict["audio_path"][index]
        midi_path = self.meta_dict["midi_path"][index]

        full_data = {
            "dataset_name": "GuitarSet",
            "audio_path": audio_path,
            "midi_path": midi_path,
        }

        audio_data = self.load_audio_data(path=audio_path)
        full_data.update(audio_data)

    def __len__(self) -> int:

        return len(self.meta_dict["audio_name"])

    # Other helper methods for constructing the three required methods above.
    def load_meta(self, meta_csv: str) -> dict:

        df = pd.read_csv(meta_csv, sep=",")

        # We want to filter out all the data that is not under our split.
        # There is no built-in split in GuitarSet, so we are going to make our
        # own split. 00-04 will be train splits, and 05 will be test splits.
        if self.split == "test":
            df = df[str(df["File Path"]).startswith("05")]
        else:
            df = df[not str(df["File Path"]).startswith("05")]

        audio_names = df["File Path"].values
        midi_names = df["Midi_file_path"].values

        audio_paths = [str(Path(self.root, "data", name)) for name in audio_names]
        midi_paths = [str(Path(self.root, "data", name)) for name in midi_names]

        durations = [
            librosa.get_duration(path=audio_path) for audio_path in audio_paths
        ]

        meta_dict = {
            "audio_name": audio_names,
            "audio_path": audio_paths,
            "midi_name": midi_names,
            "midi_path": midi_paths,
            "duration": durations,
        }

        return meta_dict

    def load_audio_data(self, path: str) -> dict:

        audio_duration = librosa.get_duration(path=path)

        if self.crop:
            start_time, clip_duration = self.crop(audio_duration=audio_duration)
        else:
            start_time = 0.0
            clip_duration = audio_duration

        audio = load(path=path, sr=self.sr, offset=start_time, duration=clip_duration)

        if self.transform is not None:
            audio = call(transform=self.transform, x=audio)

        data = {"audio": audio, "start_time": start_time, "duration": clip_duration}

        return data

    def load_question_data(self) -> dict:

        questions = [
            "Music transcription.",
            "Convert audio music into MIDI data format.",
            "Transcribe music recordings into MIDI note sequences.",
            "Automatically generate MIDI file from audio music.",
            "Extract music elements and convert to MIDI notes.",
        ]

        question = random.choice(questions)

        data = {"question": question}
        return data

    def load_target_data(
        self, midi_path: str, start_time: float, duration: float
    ) -> dict:

        notes, pedals = read_single_track_midi(
            midi_path=midi_path,
            extend_pedal=self.extend_pedal,
        )

        notes = clip_notes(notes, start_time, duration)
        pedals = clip_notes(pedals, start_time, duration)

        target = {
            "note": notes,
            "pedal": pedals,
            "start_time": start_time,
            "duration": duration,
            "midi_path": midi_path,
        }

        # Transform target
        if self.target_transform:
            target = call(transform=self.target_transform, x=target)

        return target
