from __future__ import annotations
from pathlib import Path
from typing import Optional

import yaml
import os
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate_fn_map

from audidata.io.audio import load
from audidata.io.crops import RandomCrop
from audidata.transforms.audio import Mono
from audidata.transforms.midi import PianoRoll, MultiTrackPianoRoll
from audidata.io.midi import read_single_track_midi, read_midi_beat
from audidata.collate.base import collate_list_fn

from typing_extensions import Literal
import random

default_collate_fn_map.update({list: collate_list_fn})


class Slakh2100(Dataset):
    r"""Slakh2100 [1] is a multiple track MIDI-audio paired dataset containing
    145 hours of 2,100 audio audio files rendered by MIDI files. Audios are
    sampled at 44,100 Hz. After decompression, the dataset is 101 GB.

    [1] E. Manilow, Cutting music source separation some Slakh: A dataset to
    study the impact of training data quality and quantity, WASPAA, 2019

    After decompression, dataset looks like:

        dataset_root (131 GB)
        ├── train (1500 songs)
        │   ├── Track00001
        │   │   ├── all_src.mid
        │   │   ├── metadata.yaml
        │   │   ├── MIDI
        │   │   │   ├── S00.mid
        │   │   │   ├── S01.mid
        │   │   │   └── ...
        │   │   ├── mix.flac
        │   │   └── stems
        │   │       ├── S00.flac
        │   │       ├── S01.flac
        │   │       └── ...
        │   ├── Track00002
        │   └── ...
        ├── validation (375 songs)
        └── test (225 songs)
    """

    url = "https://zenodo.org/records/4599666"

    duration = 521806.45  # Dataset duration (s), 145 hours, including training,
    # validation, and testing.

    def __init__(
        self,
        root: str,
        split: str = Literal["train", "validation", "test"],
        sr: float = 44100,
        crop: Optional[callable] = RandomCrop(clip_duration=10.0, end_pad=9.9),
        transform: Optional[callable] = Mono(),
        load_target: bool = True,
        extend_pedal: bool = True,
        target_transform: Optional[callable] = PianoRoll(
            fps=100, pitches_num=128
        ),
    ):

        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.load_target = load_target
        self.extend_pedal = extend_pedal
        self.transform = transform
        self.target_transform = target_transform

        audios_dir = Path(self.root, self.split)
        self.meta_dict = {"audio_name": sorted(os.listdir(audios_dir))}
        # self.meta_dict = self.load_meta()

    def __getitem__(self, index: int) -> dict:

        prefix = Path(self.root, self.split, self.meta_dict["audio_name"][index])
        audio_path = Path(prefix, "mix.flac")
        meta_yaml = Path(prefix, "metadata.yaml")
        mix_midi_path = Path(prefix, "all_src.mid")
        midis_dir = Path(prefix, "MIDI")

        full_data = {
            "dataset_name": "Slakh2100",
            "audio_path": str(audio_path),
        }

        # Load audio
        audio_data = self.load_audio(path=audio_path)
        full_data.update(audio_data)

        # Load question
        question_data = self.load_question()
        full_data.update(question_data)

        # Load target
        if self.load_target:
            target_data = self.load_target_data(
                meta_yaml=meta_yaml,
                mix_midi_path=mix_midi_path,
                midis_dir=midis_dir,
                start_time=audio_data["start_time"],
                clip_duration=audio_data["duration"],
            )
            full_data.update(target_data)

        return full_data

    def __len__(self):

        audios_num = len(self.meta_dict["audio_name"])

        return audios_num

    # this may be useless
    def load_meta(self) -> dict:
        r"""Load meta dict."""

        meta_dict = {
            "audio_name": [],
            "audio_path": [],
            "midi_name": [],
            "midi_path": [],
        }

        for dir in next(os.walk(str(Path(self.root, self.split))))[1]:

            meta_dict["audio_name"].append("mix.flac")
            meta_dict["audio_path"].append(
                str(Path(self.root, self.split, dir, "mix.flac"))
            )
            meta_dict["midi_name"].append("all_src.mid")
            meta_dict["midi_path"].append(
                str(Path(self.root, self.split, dir, "all_src.mid"))
            )

        return meta_dict

    def load_audio(self, path: str) -> dict:

        audio_duration = librosa.get_duration(path=path)

        if self.crop:
            start_time, clip_duration = self.crop(audio_duration=audio_duration)
        else:
            start_time = 0.0
            clip_duration = None

        audio = load(path=path, sr=self.sr, offset=start_time, duration=clip_duration)
        # shape: (channels, audio_samples)

        data = {
            "audio": audio,
            "start_time": start_time,
            "duration": clip_duration if clip_duration else audio_duration,
        }

        # may be a source of bugs.
        if self.transform is not None:
            data["audio"] = self.transform(data["audio"])

        return data

    def load_question(self) -> dict:

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
        self,
        meta_yaml: str,
        mix_midi_path: str,
        midis_dir: str,
        start_time: float,
        clip_duration: float,
    ) -> dict:

        with open(meta_yaml, "r") as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)

        beats, downbeats = read_midi_beat(mix_midi_path)

        data = {
            "start_time": start_time,
            "clip_duration": clip_duration,
            "beat": beats,
            "downbeat": downbeats,
            "tracks": [],
        }

        for stem_name, stem_data in meta["stems"].items():

            if not stem_data["midi_saved"]:
                continue

            inst_class = stem_data["inst_class"]
            is_drum = stem_data["is_drum"]
            plugin_name = stem_data["plugin_name"]
            program_num = stem_data["program_num"]

            midi_path = Path(midis_dir, "{}.mid".format(stem_name))

            notes, pedals = read_single_track_midi(
                midi_path=str(midi_path), extend_pedal=self.extend_pedal
            )

            track = {
                "track_name": stem_name,
                "inst_class": inst_class,
                "is_drum": is_drum,
                "plugin_name": plugin_name,
                "program_num": program_num,
                "note": notes,
                "pedal": pedals,
            }

            data["tracks"].append(track)

        if self.target_transform:
            data = self.target_transform(data)

        return data
