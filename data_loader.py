
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import librosa
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class KAISD_data( Dataset):
  def __init__(self, base_path, csv_path):
    self.df = pd.read_csv(csv_path)
    self.data = []
    self.labels = []
    self.c2i={}
    self.categories = sorted(self.df['label'].unique())
    for i, category in enumerate(self.categories):
      self.c2i[category]=i

    for j in tqdm(range(len(self.df))):
      row = self.df.iloc[j]
      file_path = os.path.join(base_path,row['filename'])
      file_data = load_audio(file_path, 8)
      self.data += file_data
      self.labels += [self.c2i[row['label'] ] ] * len (file_data)
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]

def get_melspectrogram_db( file_path, sr = None, max_sec = 8, n_fft=2048,
                           hop_length=512, n_mels=128, fmin=20, fmax=8300,
                           top_db=80 ):
  wav_segments = []
  spec_dbs = []
  wav,sr = librosa.load(file_path,sr=22050)
  while wav.shape[0] >= max_sec*sr:
    wav_segments.append (wav[:max_sec*sr])
    wav = wav[max_sec*sr:]
  #last segment which will be shorter that max_sec
  wav =np.pad(wav,int(np.ceil((max_sec*sr-wav.shape[0])/2)),mode='reflect')
  wav_segments.append( wav)
  for wav in wav_segments:
    spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
                hop_length = hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
    spec_db=librosa.power_to_db(spec,top_db=top_db)
    spec_dbs.append ( spec_db)
  return spec_dbs

def spec_to_image(spec, eps=1e-6):
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps)
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  spec_scaled = spec_scaled.astype(np.uint8)
  return spec_scaled

def load_audio( file_path, max_sec):
  data = []
  spec_dbs = get_melspectrogram_db(file_path, max_sec)
  for db in spec_dbs:
    data.append (spec_to_image(db)[np.newaxis,...])
  return data
