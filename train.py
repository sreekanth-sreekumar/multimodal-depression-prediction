import torch
import argparse
from modules.dataset import MultDataset
from torch.utils.data import DataLoader
# from modules.model import MULTModel
import datetime
import torch.nn as nn
# from torch.optim import Adam
# from torch.nn import L1Loss
# from torch.optim.lr_scheduler import ReduceLROnPlateau

def mask_attn(actual_num_tokens, max_length_token, device):
    masks = []
    for m in range(len(actual_num_tokens)):
        mask = [0] * actual_num_tokens[m] + [1] * (max_length_token - actual_num_tokens[m])
        masks.append(mask)
    masks = torch.tensor(masks).unsqueeze(2).to(device)
    return masks


parser = argparse.ArgumentParser(description="daic-woz depression detection")

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=5,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

number_of_epochs = 1
patience = 20

args = parser.parse_args()

# Getting each dataset
train_dataset = MultDataset('train')
test_dataset = MultDataset('test')
dev_dataset = MultDataset('dev')

hyp_params = args
hyp_params.layers = args.nlevels
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_dataset), len(dev_dataset), len(test_dataset)
hyp_params.output_dim = 1
hyp_params.criterion = ''

hyp_params.orig_d_a, hyp_params.orig_d_v, hyp_params.orig_d_l = train_dataset.get_dim()

# model = MULTModel(hyp_params)
# optimizer = Adam(model.parameters(), lr=0.001)
# criterion = L1Loss()
# scheduler = ReduceLROnPlateau(optimizer, mode='min', patience = args.when, factor=0.1, verbose=True)

use_cuda = True if torch.cuda.is_available() else False

# Initialising device
print('cuda') if torch.cuda.is_available() else print('cpu')
device = torch.device('cuda' if use_cuda else 'cpu')
# model.to(device)

load_params = {
    'batch_size': 16,
    'collate_fn': MultDataset.get_collate_fn(device)
}
patience_counter = 0

if __name__ == '__main__':

    t = datetime.datetime.now()
    timestamp = str(t.date()) + ' ' + str(t.hour) + ' hours ' + str(t.minute) + ' minutes ' + str(t.second) + ' seconds'
    print('Training starts: ', timestamp)

    for epoch in range(number_of_epochs):
        print('Epoch ', epoch)
        train_loader = DataLoader(train_dataset, shuffle=True, **load_params)
        val_loader = DataLoader(dev_dataset, shuffle=True, **load_params)

        # losses = []
        # model.train()
        # torch.enable_grad()
        print(epoch)
        for i, data in enumerate(train_loader):
            # optimizer.zero_grad()

            audio = data['audio']
            audio_length = data['audio_length']
            video = data['video']
            video_length = data['video_length']
            text = data['text']
            text_length = data['text_length']

            print('Text', text.size())
            print('Video', video.size())
            print('Audio', audio.size())

            # text_mask = mask_attn(text_length, text.shape[1], device)
            # audio_mask = mask_attn(audio_length, audio.shape[1], device)
            # video_mask = mask_attn(video_length, video.shape[1], device)

            # model(text, audio, video, text_mask, audio_mask, video_mask)

    