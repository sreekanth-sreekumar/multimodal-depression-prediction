import nltk
nltk.download('stopwords')
import csv
import numpy as np
import re
from nltk.corpus import stopwords
import joblib

stop = set(stopwords.words('english'))
# Getting pretrained glove vectors
glove = {}
with open('glove.6B.50d.txt', 'r', encoding='utf-8') as f:
    for l in f:
        line = l.split(' ')
        word = line[0]
        vector = np.asarray(line[1:], dtype='float32')
        glove[word] = vector

def get_glove_features(id):
    # Reading utternaces from transcript files
    utterances = np.array([])
    with open('./data/' + id + '_TRANSCRIPT.csv') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
          if row:
            row = row[0].split('\t')
            if row[2].lower() == 'participant':
                line = row[3].lower()

                #Stripping punctuations
                line_by_words = re.findall(r'(?:\w+)', line, flags = re.UNICODE)
                new_line=[]
                #getting glove vectors of the words, if not random vector
                for word in line_by_words:
                    if word not in stop:
                        try:
                            vector = glove[word]
                        except KeyError:
                            # Embedding dim = 50
                            vector = np.random.normal(scale=0.6, size = (50, ))
                        new_line.append(vector)
                new_line.append(glove['.'])
                utterances = np.append(utterances, new_line)
    return utterances

class TextDataset():
    def __init__(self, split):
        self.split = split
        
        #Open split file
        total_utterances = {}
        with open('./data/' + self.split + '_split.csv', 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                id = row[0]

                utterances = get_glove_features(id)

                total_utterances[id] = utterances
        self.save_linguistic_features(total_utterances)     

    def save_linguistic_features(self, feats):
        with open('./saved_data/text_features_' + self.split + '.sav', 'wb') as f:
            joblib.dump(feats, f)
        
class AudioDataset():
    def __init__(self, split):
        self.split = split
        total_audio = {}
        with open('./data/' + self.split + '_split.csv', 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                id = row[0]
                # Reading audio features
                covarep = np.array([])
                with open('./data/' + id + '_COVAREP.csv') as file:
                    csv_reader = csv.reader(file)
                    for row in csv_reader:
                        if not int(row[1]):
                           covarep = np.append(covarep, [float(f) for f in row[2:]])

                total_audio[id] = covarep
        self.save_audio_features(total_audio)

    def save_audio_features(self, feats):
        with open('./saved_data/audio_features_' + self.split + '.sav', 'wb') as f:
            joblib.dump(feats, f)

def read_text_file(file_name):
    features = np.array([])
    with open(file_name) as file:
        lines = file.readlines()[1:]
        for line in lines:
            feature = line.split(',')[2:]
            if not '-1.#IND' in feature:
                features = np.append(features, [float(f) for f in feature])
    return features

class VideoDataset():
    def __init__(self, split):
        self.split = split
        
        #Open split file
        total_video = {}
        with open('./data/' + self.split + '_split.csv', 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                id = row[0]
                # Reading Action units, pose and gaze features
                au = read_text_file('./data/' + id + '_CLNF_AUs.txt')
                gaze = read_text_file('./data/' + id + '_CLNF_gaze.txt')
                pose = read_text_file('./data/' + id + '_CLNF_pose.txt')

                total_video[id] = {'au': au, 'gaze': gaze, 'pose': pose}
        self.save_video_features(total_video)
      
    def save_video_features(self, feats):
        with open('./saved_data/video_features_' + self.split + '.sav', 'wb') as f:
            joblib.dump(feats, f)  
