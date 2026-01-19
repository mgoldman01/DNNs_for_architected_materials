import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def collate_augmented(data_list: list):
#     # data_list is a list of tuples
#     return [[data, label] for data, label in zip(data_list, labels_list)]
    
def train(net, dataloader, criterion, optimizer):
    net.train()  # Set the model to training mode
    running_loss = 0.0
    pbar = tqdm(dataloader)  # Use tqdm for progress bars
    for inputs, targets in pbar:
        inputs = inputs.to(device)  # Move inputs to GPU
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_description(f'Train Loss: {running_loss / (pbar.n + 1):.4f}')
        # print("Data index is: f{idx}")
    return running_loss / len(dataloader)


def validate(net, dataloader, criterion):
    net.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    pbar = tqdm(dataloader)  # Use tqdm for progress bars
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs = inputs.to(device)  # Move inputs to GPU
            targets = targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            pbar.set_description(f'Val Loss: {running_loss / (pbar.n + 1):.4f}')
    return running_loss / len(dataloader)
