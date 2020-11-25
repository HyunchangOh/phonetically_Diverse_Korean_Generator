import argparse
import torch
import json
import os
import csv
import torchaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from networks.cnn import CNN
from data_loader import KAISD_data
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def parse():
    parser = argparse.ArgumentParser(description="kaisd classifier")
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()
    return args

### main function ###
if __name__ == '__main__':
    args = parse().__dict__
    data_path = args['data_path']
    csv_path = args['csv_path']

    train_data = KAISD_data(data_path, csv_path)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    else:
        device=torch.device('cpu')

    ### load the CNN ###
    net = CNN(input_shape = (1,128,345))

    # ### Define loss function and optimizer ###
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # ### Training the Network ###
    losses = []
    pixel_accs = []

    for epoch in tqdm ( range (20) ):
        net.train()
        for batch_idx, (audio, label) in enumerate(train_loader):
            audio = audio.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.long)
            output = net( audio )
            optimizer.zero_grad()
            loss = criterion ( output, label)
            loss.backward()
            optimizer.step()

            ### print statistics ###
            losses.append(loss.item())
            print(f'epoch: {epoch + 1}')
            print(f'loss: {np.mean(losses[-1])}')
