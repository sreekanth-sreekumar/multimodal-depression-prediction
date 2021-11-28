import torch
from torch.utils.data.dataset import Dataset
import joblib
import numpy as np
import csv
from collections import defaultdict

class MultDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.data = dict()

        #Open Audio Features
        with open('./saved_data/audio_features_' + self.split + '.sav', 'rb') as f:
            audio = joblib.load(f)

        #Open Video Features
        with open('./saved_data/video_features_' + self.split + '.sav', 'rb') as f:
            video = joblib.load(f)

        #Open Text Features
        with open('./saved_data/text_features_' + self.split + '.sav', 'rb') as f:
            text = joblib.load(f)

        #Open File According to Split
        with open('./data/' + self.split + '_split.csv', 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)

            #For Each Row in the Split File
            for row in csv_reader:
                #Get Row ID
                id = row[0]

                if id:
                    #Getting Audio Features and Length
                    audio_features = audio[id]
                    audio_length = len(audio_features)

                    #Getting Video Features and Length
                    video_features = video[id]
                    
                    # TODO: Pose has an indeterminate value in some samples.
                    # These sample timestamps needs to be handled for gaze and au as well.
                    video_concat_array = video_features['au']
                    video_length = len(video_concat_array)

                    #Getting Text Features and Length
                    text_features = text[id]
                    text_length = len(text_features)

                    #Add Audio, Video, and Text Data to Dictionary
                    self.data[len(self.data)] = {
                        'audio': torch.from_numpy(audio_features.astype(np.float32)),
                        'audio_length': audio_length,
                        'video': torch.from_numpy(video_concat_array.astype(np.float32)),
                        'video_length': video_length,
                        'text': torch.from_numpy(text_features.astype(np.float32)),
                        'text_length': text_length,
                        'binary': int(row[1]),
                        'severity': int(row[2]),
                        'audio_dim': audio_features.shape[1],
                        'video_dim': video_concat_array.shape[1],
                        'text_dim':  text_features.shape[1] 
                    }
    
    def __len__(self):
        #Return Length of Dictionary
        return len(self.data)

    def get_dim(self):
        return self.data[0]['audio_dim'], self.data[0]['video_dim'], self.data[0]['text_dim']

    def __getitem__(self, index):
        #Get Specific Item From Dictionary
        return self.data[index]

    def get_collate_fn(device):
        def collate_fn(data):
            #Get Max Length Per Modality
            max_text_len = max(d['text_length'] for d in data)
            max_audio_len = max(d['audio_length'] for d in data)
            max_video_len = max(d['video_length'] for d in data)

            batch = defaultdict(list)

            #For Each Row of Data
            for sample in data:
                #For Each Piece of Information in the Row
                for key in data[0].keys():
                    
                    # Adding Zero Padding to Each Feature Based on Max Length of that Modality
                    if key == 'audio':
                        pad_rep = torch.zeros(max_audio_len - sample['audio_length'], sample[key].shape[1])
                        padded = torch.cat((sample[key], pad_rep), dim=0)

                    elif key == 'video':
                        pad_rep = torch.zeros(max_video_len - sample['video_length'], sample[key].shape[1])
                        padded = torch.cat((sample[key], pad_rep), dim=0)
                    
                    elif key == 'text':
                        pad_rep = torch.zeros(max_text_len - sample['text_length'], sample[key].shape[1])
                        padded = torch.cat((sample[key], pad_rep), dim=0)

                    else:
                        padded = sample[key]
                        
                    batch[key].append(padded)

            for key in ['audio', 'video', 'text']:
                    batch[key] = torch.stack(batch[key]).to(device)

            return batch

        return collate_fn