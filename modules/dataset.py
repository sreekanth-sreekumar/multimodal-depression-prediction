import torch
from torch.utils.data.dataset import Dataset
import joblib
import numpy as np
import csv

#Set Default Type Based on CUDA Availability
if torch.cuda.is_available():
    torch.set_default_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class MultDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.data = dict()

        #Open Audio Features
        with open('audio_features_' + self.split + '.sav') as f:
            audio = joblib.load(f)

        #Open Video Features
        with open('video_features_' + self.split + '.sav') as f:
            video = joblib.load(f)

        #Open Text Features
        with open('text_features_' + self.split + '.sav') as f:
            text = joblib.load(f)

        #Open File According to Split
        with open('./data/' + self.split + '_split.csv', 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)

            #For Each Row in the Split File
            for row in csv_reader:
                #Get Row ID
                id = row[0]

                if id is not None:
                    #Getting Audio Features and Length
                    audio_features = audio[id]
                    audio_length = len(audio_features)

                    #Getting Video Features and Length
                    video_features = video[id]
                    video_concat_array = np.concatenate((
                        video_features['au'],
                        video_features['gaze'],
                        video_features['pose']
                    ), axis=1)
                    video_length = len(video_features)

                    #Getting Text Features and Length
                    text_features = text[id]
                    text_length = len(text_features)

                    #Add Audio, Video, and Text Data to Dictionary
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
        #Return Length of Dictionary
        return len(self.data)

    def __getitem__(self, index):
        #Get Specific Item From Dictionary
        return self.data[index]

    def get_collate_fn(device):
        def collate_fn(data):
            #Get Max Length Per Modality
            max_text_len = max(d['text_length'] for d in data)
            max_audio_len = max(d['audio_length'] for d in data)
            max_video_len = max(d['video_length'] for d in data)

            #Create Batch Dictionary
            batch = dict()

            #For Each Row of Data
            for sample in data:
                #For Each Piece of Information in the Row
                for key in data[0].keys():
                    
                    # Adding Zero Padding to Each Feature Based on Max Length of that Modality
                    if key == 'audio':
                        pad_rep = torch.zeros(max_audio_len - sample['audio_length'], sample[key].shape[1])
                        padded = torch.cat((sample[key]))
                    elif key == 'video':
                        pad_rep = torch.zeros(max_video_len - sample['video_length'], sample[key].shape[1])
                        padded = torch.cat((sample[key]))
                    elif key == 'text':
                        pad_rep = torch.zeros(max_text_len - sample['text_length'], sample[key].shape[1])
                        padded = torch.cat((sample[key]))
                    else:
                        padded = sample[key]
                    batch[key].append(padded)

            #Put Each Row of Data From Batch on Device
            for key in batch.keys():
                    batch[key] = batch[key].to(device)
            return batch

        return collate_fn
