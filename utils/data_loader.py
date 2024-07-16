# data_loader.py

import os
import random
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import numpy as np
from audiomentations import Compose, ApplyImpulseResponse

class AddMusanNoise:
    def __init__(self, noise_path, p=0.5):
        self.noise_path = noise_path
        self.noise_files = self.load_musan_noise_dataset()
        self.p = p

    def load_musan_noise_dataset(self):
        musan_dataset = []
        for root, _, filenames in sorted(os.walk(self.noise_path, followlinks=True)):
            for fn in sorted(filenames):
                name, ext = os.path.splitext(fn)
                if ext == ".wav":
                    musan_dataset.append(os.path.join(root, fn))
        return musan_dataset

    def __call__(self, samples, sample_rate):
        if random.random() < self.p:
            noise_index = torch.randint(0, len(self.noise_files), size=(1,)).item()
            noise, _ = torchaudio.load(self.noise_files[noise_index])
            noise = self._match_length(noise, samples.size(1))
            noise_level = np.random.choice([0, -5, -10])
            noise = noise * 10 ** (noise_level / 20.0)
            samples = samples + noise
        return samples

    def _match_length(self, noise, length):
        if noise.shape[1] < length:
            noise = F.pad(noise, (0, length - noise.shape[1]))
        else:
            offset = torch.randint(0, noise.shape[1] - length + 1, size=(1,)).item()
            noise = noise[:, offset:offset + length]
        return noise

class SpeechCommandDataset(Dataset):
    def __init__(self, dataset_path, json_filename, is_training, class_list, class_encoding, noise_aug=False, musan_path="", rir_path=""):
        super(SpeechCommandDataset, self).__init__()
        self.classes = class_list
        self.sampling_rate = 16000
        self.sample_length = 16000
        self.dataset_path = dataset_path
        self.root = os.path.dirname(dataset_path)
        self.json_filename = json_filename
        self.is_training = is_training
        self.class_encoding = class_encoding
        self.noise_aug = noise_aug
        self.noise_path = os.path.join(self.root, "_background_noise_")
        self.musan_noise_path = musan_path
        self.rir_path = rir_path
        self.noise_dataset = self.load_noise_dataset()
        self.musan_noise_dataset = self.load_musan_noise_dataset()
        self.speech_dataset = self.load_speech_dataset()
        self.noise_levels = [0, -5, -10]

        self.augment = Compose([
            AddMusanNoise(self.musan_noise_path, p=0.5),
        ])

    def load_noise_dataset(self):
        noise_dataset = []
        for root, _, filenames in sorted(os.walk(self.noise_path, followlinks=True)):
            for fn in sorted(filenames):
                name, ext = os.path.splitext(fn)
                if ext == ".wav":
                    noise_dataset.append(os.path.join(root, fn))
        return noise_dataset
    
    def load_musan_noise_dataset(self):
        musan_dataset = []
        for root, _, filenames in sorted(os.walk(self.musan_noise_path, followlinks=True)):
            for fn in sorted(filenames):
                name, ext = os.path.splitext(fn)
                if ext == ".wav":
                    musan_dataset.append(os.path.join(root, fn))
        return musan_dataset

    def load_speech_dataset(self):
        with open(self.json_filename, 'r') as f:
            json_data = [json.loads(line) for line in f.readlines()]

        dataset_list = []
        for item in json_data:
            category = item["command"]
            if category in self.classes:
                category_label = category
            else:
                category_label = "unknown"
            dataset_list.append([item["audio_filepath"], category_label])
        return dataset_list

    def _spec_augmentation(self, x, num_time_mask=1, num_freq_mask=1, max_time=25, max_freq=25):
        """perform spec augmentation 
        Args:
            x: input feature, T * F 2D
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
        Returns:
            augmented feature
        """
        max_freq_channel, max_frames = x.size()

        # time mask
        for i in range(num_time_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_time)
            end = min(max_frames, start + length)
            x[:, start:end] = 0

        # freq mask
        for i in range(num_freq_mask):
            start = random.randint(0, max_freq_channel - 1)
            length = random.randint(1, max_freq)
            end = min(max_freq_channel, start + length)
            x[start:end, :] = 0

        return x

    def load_audio(self, speech_path):
        waveform, _ = torchaudio.load(speech_path)

        if waveform.shape[1] < self.sample_length:
            # padding if the audio length is smaller than sampling length.
            waveform = F.pad(waveform, [0, self.sample_length - waveform.shape[1]])
       
        if self.is_training:
            pad_length = int(waveform.shape[1] * 0.1)
            waveform = F.pad(waveform, [pad_length, pad_length])
            offset = torch.randint(0, waveform.shape[1] - self.sample_length + 1, size=(1,)).item()
            waveform = waveform.narrow(1, offset, self.sample_length)

            if self.noise_aug and random.random() < 0.8: 
                noise_index = torch.randint(0, len(self.noise_dataset), size=(1,)).item()
                noise, _ = torchaudio.load(self.noise_dataset[noise_index])
                offset = torch.randint(0, noise.shape[1] - self.sample_length + 1, size=(1,)).item()
                noise = noise.narrow(1, offset, self.sample_length)
                background_volume = torch.rand(size=(1,)).item() * 0.1
                waveform.add_(noise.mul_(background_volume)).clamp(-1, 1)

        return waveform

    def add_noise(self, data, noise_level):
        noise_index = torch.randint(0, len(self.musan_noise_dataset), size=(1,)).item()
        noise, _ = torchaudio.load(self.musan_noise_dataset[noise_index])
        if noise.shape[1] < data.shape[1]:
            noise = F.pad(noise, [0, data.shape[1] - noise.shape[1]])
        else:
            offset = torch.randint(0, noise.shape[1] - data.shape[1] + 1, size=(1,)).item()
            noise = noise.narrow(1, offset, data.shape[1])
        noise = noise * 10 ** (noise_level / 20.0)
        return data + noise


    def one_hot(self, speech_category):
        encoding = self.class_encoding[speech_category]
        return encoding

    def __len__(self):
        return len(self.speech_dataset)
    
    def __getitem__(self, index):
        speech_path, speech_category = self.speech_dataset[index]
        speech_path = os.path.join(self.root, speech_path)
        label = self.one_hot(speech_category)

        if speech_category == "silence":
            waveform = torch.zeros(1, self.sampling_rate)
        else:
            waveform = self.load_audio(speech_path)

        return waveform, label
