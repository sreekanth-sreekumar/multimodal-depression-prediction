import torch
from torch.utils.data.dataset import Dataset
import joblib
import numpy as np
import csv

if torch.cuda.is_available():
    torch.set_default_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class MultDataset(Dataset):
    def __init__(self, split, device):
        self.split = split
        
        with open('audio_features_' + self.split + '.sav') as f:
            audio = joblib.load(f)

        with open('video_features_' + self.split + '.sav') as f:
            video = joblib.load(f)

        with open('text_features_' + self.split + '.sav') as f:
            text = joblib.load(f)

        self.data = dict()

        with open('../data/train_split.csv', 'r') as csv_file:
            with open('./data/' + self.split + '_split.csv', 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)
                for row in csv_reader:
                    id = row[0]

                    audio_features = audio[id]
                    video_features = video[id]
                    video_concat_array = np.concatenate((
                        video_features['au'], 
                        video_features['gaze'],
                        video_features['pose']
                    ), axis = 0)
                    text_features = text[id]

                    self.data[len(self.data)] = {
                        'audio': torch.from_numpy(audio_features).to(device),
                        'video': torch.from_numpy(video_concat_array).to(device),
                        'text': torch.from_numpy(text_features).to(device),
                        'binary': row[1],
                        'severity': row[2]
                    }
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

