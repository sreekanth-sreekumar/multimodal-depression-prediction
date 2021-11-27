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
    print('Loading glove embeddings into a dictionary ...')
    with open('glove.6B.50d.txt', 'r', encoding="utf8") as f:
        for l in f:
            line = l.split(' ')
            word = line[0]
            vector = np.asarray(line[1:], dtype='float32')
            glove[word] = vector

def get_glove_features(id):
    # Reading Utterance From Transcript File for Given ID
    utterances = []

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
                utterances.extend(new_line)
    return np.array(utterances)

def read_text_file(file_name):
    features = []
    with open(file_name) as file:
        lines = file.readlines()[1:] #Skip First Row When Reading Line
        row_count = 0
        for line in lines:
            feature = line.split(',')[2:] #Skip First Two Columns In Line
            # Include features only if they aren't indeterminate and has sucess=1
            # Also subsample by 0.3s
            if ' -1.#IND' not in feature and int(feature[1]) and row_count%3 == 0:
                features.append(np.array([float(f) for f in feature]))
            row_count += 1
    return np.array(features)

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
                    print(id)
                    utterances = get_glove_features(id) #Get Utterances for Given ID
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
                    print(id)
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
                    covarep = covarep[cov_range%4 == 0]
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
                    print(id)
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
