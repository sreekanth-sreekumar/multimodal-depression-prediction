import nltk
import csv
import numpy as np
import re
from nltk.corpus import stopwords
import joblib

nltk.download('stopwords')
stop = set(stopwords.words('english'))
glove = {}

def load_glove_embeddings():
    # Getting Pretrained GLOVE Vectors
    with open('glove.6B.50d.txt', 'r', encoding="utf8") as f:
        for l in f:
            line = l.split(' ')
            word = line[0]
            vector = np.asarray(line[1:], dtype='float32')
            glove[word] = vector

def get_glove_features(id):
    # Reading Utterance From Transcript File for Given ID
    utterances = np.array([])

    with open('./data/' + id + '_TRANSCRIPT.csv') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
          if row:
            row = row[0].split('\t')
            #Only Extract Utterances for Participant (Not Automated Agent)
            if row[2].lower() == 'participant':
                line = row[3].lower()

                #Stripping Punctuations
                line_by_words = re.findall(r'(?:\w+)', line, flags=re.UNICODE)
                new_line = []

                #Getting Glove Vectors of the Words, Using Random Vector If Not Found
                for word in line_by_words:
                    if word not in stop:
                        try:
                            vector = glove[word]
                        except KeyError: #Can't Find Word in Pretrained Vector Dictionary
                            vector = np.random.normal(scale=0.6, size=(50, )) #Create Random Vector of Size 50
                        new_line.append(vector)

                #Append a Period to End of New Line
                new_line.append(glove['.'])

                #Append to List of Utterances
                utterances = np.append(utterances, new_line)
    return utterances

def read_text_file(file_name):
    features = np.array([])
    with open(file_name) as file:
        lines = file.readlines()[1:] #Skip First Row When Reading Line
        for line in lines:
            feature = line.split(',')[2:] #Skip First Two Columns In Line
            if '-1.#IND' not in feature: #Include Feature Only If Not Indeterminate
                features = np.append(features, [float(f) for f in feature])
    return features

#Dataset for Representing Text
class TextDataset():
    def __init__(self, split):
        self.split = split
        
        #Open Split File
        total_utterances = {}
        with open('./data/' + self.split + '_split.csv', 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                id = row[0]
                utterances = get_glove_features(id) #Get Utterances for Given ID
                total_utterances[id] = utterances
        self.save_linguistic_features(total_utterances)     

    def save_linguistic_features(self, feats):
        with open('./saved_data/text_features_' + self.split + '.sav', 'wb') as f:
            joblib.dump(feats, f)

#Dataset for Representing Audio
class AudioDataset():
    def __init__(self, split):
        self.split = split
        total_audio = {}

        # Open Split File
        with open('./data/' + self.split + '_split.csv', 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                id = row[0]

                # Reading Audio Features
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

#Dataset for Representing Video
class VideoDataset():
    def __init__(self, split):
        self.split = split

        total_video = {}

        #Open Split File
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

if __name__ == '__main__':
    load_glove_embeddings()
