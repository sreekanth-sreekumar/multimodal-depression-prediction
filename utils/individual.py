from sentence_transformers import SentenceTransformer
import csv
import numpy as np
import re
import joblib

model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cuda')

def get_setence_features(id):
    # Getting sentence embedding
    with open('./data/' + id + '_TRANSCRIPT.csv') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        utterances = []
        for row in csv_reader:
          if row:
            row = row[0].split('\t')
            #Only Extract Utterances for Participant (Not Automated Agent)
            if row[2].lower() == 'participant':
                utterances.append(row[3])
        embeddings = model.encode(utterances, convert_to_numpy=True)
        return embeddings
    

def read_text_file(file_name):
    features = []
    with open(file_name) as file:
        lines = file.readlines()[1:] #Skip First Row When Reading Line
        row_count = 0
        for line in lines:
            feature = line.split(',')[2:] #Skip First Two Columns In Line
            # Include features only if they aren't indeterminate and has sucess=1
            # Also subsample by 0.3s
            if ' -1.#IND' not in feature and int(feature[1]):
                features.append(np.array([float(f) for f in feature]))

    features = np.array(features)
    # Subsampling by 4
    feat_range = np.arange(features.shape[0])
    features = features[feat_range%9 == 0]
    return features

#Dataset for Representing Text
class TextDataset():
    def __init__(self, split):
        self.split = split
        
        #Open Split File
        total_utterances = {}
        print('Loading up text features ...')
        with open('./data/' + self.split + '_split.csv', 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                if row[0]:
                    id = row[0]
                    utterances = get_setence_features(id) #Get Utterances for Given ID
                    total_utterances[id] = utterances
        self.save_linguistic_features(total_utterances)     

    def save_linguistic_features(self, feats):
        print('Saving text features ...')
        with open('./saved_data/text_features_' + self.split + '.sav', 'wb') as f:
            joblib.dump(feats, f)

#Dataset for Representing Audio
class AudioDataset():
    def __init__(self, split):
        self.split = split
        total_audio = {}

        # Open Split File
        print('Loading up audio features ...')
        with open('./data/' + self.split + '_split.csv', 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                if row[0]:
                    id = row[0]
                    # Reading Audio Features
                    covarep = []
                    with open('./data/' + id + '_COVAREP.csv') as file:
                        cv_reader = csv.reader(file)
                        for row in cv_reader:
                            # Skipping lines with VAV = 0
                            if int(row[1]):
                                audio_features = np.array([float(f) for f in row[:1] + row[2:]])
                                covarep.append(audio_features)
                            
                    covarep = np.array(covarep)
                    # Subsampling by 4
                    cov_range = np.arange(covarep.shape[0])
                    covarep = covarep[cov_range%12 == 0]
                    total_audio[id] = covarep
        self.save_audio_features(total_audio)

    def save_audio_features(self, feats):
        print('Saving audio features ...')
        with open('./saved_data/audio_features_' + self.split + '.sav', 'wb') as f:
            joblib.dump(feats, f)

#Dataset for Representing Video
class VideoDataset():
    def __init__(self, split):
        self.split = split

        total_video = {}

        #Open Split File
        print('Loading up video features ...')
        with open('./data/' + self.split + '_split.csv', 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                if row[0]:
                    id = row[0]
                    # Reading Action units, pose and gaze features
                    au = read_text_file('./data/' + id + '_CLNF_AUs.txt')
                    gaze = read_text_file('./data/' + id + '_CLNF_gaze.txt')
                    pose = read_text_file('./data/' + id + '_CLNF_pose.txt')

                    total_video[id] = {'au': au, 'gaze': gaze, 'pose': pose}
        self.save_video_features(total_video)
      
    def save_video_features(self, feats):
        print('Saving video features ...')
        with open('./saved_data/video_features_' + self.split + '.sav', 'wb') as f:
            joblib.dump(feats, f)
