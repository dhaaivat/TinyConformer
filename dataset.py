import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torch.nn.functional as F

DATASET_DIR = "/content/mini_speech_commands"

def pad_or_trim(mel, max_frames=64):
    """
    Pads or trims mel spectrogram to have exactly `max_frames` in the time dimension.
    mel shape: [1, n_mels, time]
    Returns: [time, mel_bins]
    """
    time_dim = mel.shape[-1]
    if time_dim < max_frames:
        pad_amount = max_frames - time_dim
        mel = F.pad(mel, (0, pad_amount))
    else:
        mel = mel[:, :, :max_frames]
    
    mel = mel.squeeze(0).transpose(0, 1)  # [1, mel_bins, time] -> [time, mel_bins]
    return mel


class MiniSpeechCommandsMelDataset(Dataset):
    def __init__(self, root_dir=DATASET_DIR, split="train", sample_rate=16000, mel_bins=80):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate

        self.mel_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=mel_bins,
            n_fft=1024,
            hop_length=256
        )
        self.db_transform = AmplitudeToDB()

        self.all_files = []
        self.labels = []

        self.label_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_names)}

        for label in self.label_names:
            files = list((self.root_dir / label).glob("*.wav"))
            for f in files:
                self.all_files.append(f)
                self.labels.append(self.label_to_idx[label])

        total = len(self.all_files)
        if split == "train":
            self.all_files = self.all_files[:int(0.8 * total)]
            self.labels = self.labels[:int(0.8 * total)]
        elif split == "val":
            self.all_files = self.all_files[int(0.8 * total):int(0.9 * total)]
            self.labels = self.labels[int(0.8 * total):int(0.9 * total)]
        elif split == "test":
            self.all_files = self.all_files[int(0.9 * total):]
            self.labels = self.labels[int(0.9 * total):]

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file_path = self.all_files[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        mel_spec = self.mel_transform(waveform)      # [1, mel_bins, time]
        mel_spec = self.db_transform(mel_spec)        # log-mel
        mel_spec = pad_or_trim(mel_spec, max_frames=64)  # [time, mel_bins]

        return mel_spec, label


def get_dataloaders(batch_size=32):
    train_set = MiniSpeechCommandsMelDataset(split="train")
    val_set = MiniSpeechCommandsMelDataset(split="val")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    return train_loader, val_loader
