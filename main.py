import torch
from data_models.dataset import MultDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':

    # Initialising device
    device = torch.device('cuda' if torch.cuda_is_available() else 'cpu')
    
    # Getting each dataset
    train_dataset = MultDataset('train')
    test_dataset = MultDataset('test')
    dev_dataset = MultDataset('dev')

    # Loading data loaders
    train_loader = Dataloader(train_dataset, shuffle=True, batch_size=32)
    test_loader = Dataloader(test_dataset, shuffle=True, batch_size=32)
    dev_loader = Dataloader(dev_dataset, shuffle=True, batch_size=32)