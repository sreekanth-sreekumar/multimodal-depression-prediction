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
    def __init__(self, split):
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
                    
                    # Getting audio features and length
                    audio_features = audio[id]
                    audio_length = len(audio_features)

                    # Getting video features and length
                    video_features = video[id]
                    video_concat_array = np.concatenate((
                        video_features['au'], 
                        video_features['gaze'],
                        video_features['pose']
                    ), axis = 1)
                    video_length = len(video_features)

                    text_features = text[id]
                    text_length = len(text_features)

                    self.data[len(self.data)] = {
                        'audio': torch.from_numpy(audio_features),
                        'audio_length': audio_length,
                        'video': torch.from_numpy(video_concat_array),
                        'video_length': video_length,
                        'text': torch.from_numpy(text_features),
                        'text_length': text_length,
                        'binary': row[1],
                        'severity': row[2]
                    }
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def get_collate_fn(device):

        def collate_fn(data):
            max_text_len = max(d['text_length'] for d in data)
            max_audio_len = max(d['audio_length'] for d in data)
            max_video_len = max(d['video_length'] for d in data)

            batch = dict()

            for sample in data:
                for key in data[0].keys():
                    
                    # Adding padding to each features
                    if key == 'audio':
                        pad_rep = torch.zeros(max_audio_len - sample['audio_length'], sample[key].shape[1])
                        padded = torch.cat((sample[key]))
                    elif key == 'video':
                        pad_rep = torch.zeros(max_audio_len - sample['video_length'], sample[key].shape[1])
                        padded = torch.cat((sample[key]))
                    
                    elif key == 'text':
                        pad_rep = torch.zeros(max_audio_len - sample['text_length'], sample[key].shape[1])
                        padded = torch.cat((sample[key]))
                    else:
                        padded = sample[key]
                    batch[key].append(padded)

            for key in batch.keys():
                    batch[key] = batch[key].to(device)

            return batch

        return collate_fn

                
