import os
import zipfile
import io
import requests
from requests.auth import HTTPBasicAuth

from individual import * 

if (not os.path.exists('./data')):
    os.mkdir('./data')
      
if (not os.path.exists('./saved_data')):
    os.mkdir('./saved_data')

if __name__ == '__main__':
    #Load Glove Embeddings
    load_glove_embeddings()

    #URL and Token for Authentication
    DATABASE_URL = 'https://dcapswoz.ict.usc.edu/wwwdaicwoz/'
    AUTH_TOKEN = HTTPBasicAuth('daicwozuser', 'Tqiy7T7CD9OBTa1VZ5CLjgni')

    #Path to Split Files
    TRAIN_PATH = 'train_split_Depression_AVEC2017.csv'
    TEST_PATH = 'full_test_split.csv'
    DEV_PATH = 'dev_split_Depression_AVEC2017.csv'

    #Get Index Page and All Participant IDs Using RegExp
    req = requests.get(DATABASE_URL, auth=AUTH_TOKEN, stream=True)
    ind_ids = re.findall('href="(.*?)_P.zip"', str(req.text))

    #For each ID, Extract Its Contents
    print('Extracting subject contents ...')
    for id in ind_ids:
        req = requests.get(DATABASE_URL + id + '_P.zip', auth=AUTH_TOKEN, stream=True)
        z = zipfile.ZipFile(io.BytesIO(req.content))
        z.extractall(path='./data/')
        print('Id: {} extracted'.format(id))

    #File for Each Split
    print('Getting each split files ...')
    train_file = requests.get(DATABASE_URL + TRAIN_PATH, auth=AUTH_TOKEN, stream=True)
    test_file = requests.get(DATABASE_URL + TEST_PATH, auth=AUTH_TOKEN, stream=True)
    dev_file = requests.get(DATABASE_URL + DEV_PATH, auth=AUTH_TOKEN, stream=True)

    #Save Training Split
    print('Writing train file ...')
    with open('./data/train_split.csv', 'w+') as csv_file:
        writer = csv.writer(csv_file)
        for line in train_file.iter_lines():
            writer.writerow(line.decode('utf-8').split(','))

    #Save Dev Split
    print('Writing dev file ...')
    with open('./data/dev_split.csv', 'w+') as csv_file:
        writer = csv.writer(csv_file)
        for line in dev_file.iter_lines():
            writer.writerow(line.decode('utf-8').split(','))

    #Save Testing Split
    print('Writing test file ...')
    with open('./data/test_split.csv', 'w+') as csv_file:
        writer = csv.writer(csv_file)
        for line in test_file.iter_lines():
            writer.writerow(line.decode('utf-8').split(','))

    # Create Train Dataset for Audio, Video, and Text
    AudioDataset('train')
    VideoDataset('train')
    TextDataset('train')

    AudioDataset('test')
    VideoDataset('test')
    TextDataset('test')

    AudioDataset('dev')
    VideoDataset('dev')
    TextDataset('dev')