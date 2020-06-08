from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import wandb
import numpy as np

from datapreprocessing import DataProprocessing
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

def train_step(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
        
    train_loss = 0
    n_iter = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        n_iter += 1
    
    average_loss = train_loss / n_iter
    print('\tTrain Loss: {:.4f}'.format(average_loss))
    wandb.log({
        "Train Loss": average_loss})
    return average_loss
    

def test_step(model, device, test_loader, criterion):
    model.eval()
    
    valid_loss = 0
    valid_n_iter = 0
    
    total = 0
    correct = 0
    
    for images, labels in test_loader:         
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        loss = criterion(outputs, labels)
    
        total += labels.size(0)
        correct += torch.sum(predicted == labels.data)
        valid_loss += loss.item()
        valid_n_iter += 1
    
    epoch_acc = 100 * correct / total
    average_loss = valid_loss / valid_n_iter
    print('\tValid Loss: {:.4f}'.format(average_loss))
    print('\tValid Acc: {:.4f}'.format(epoch_acc))
    
    wandb.log({
        "Test Accuracy": epoch_acc,
        "Test Loss": average_loss})
    
    
    return average_loss

          
def main():
    wandb.init(project="chart-type")
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Resnet50 for Chart Classification')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=443, metavar='S',
                        help='random seed (default: 443)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    #not used
    use_cuda = not args.no_cuda and torch.cuda.is_available()    
    
    wandb.config.update(args)

    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if use_gpu else "cpu")

    kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}
    
    # hardcoded values
    TASK1_FILEPATH = '../labels_task1.csv'
    TRAIN_IMAGES_DIR = '/chart/images/'
    NUMBER_CHART_TYPES = 10
    
    dp = DataProprocessing(TRAIN_IMAGES_DIR, None, TASK1_FILEPATH, seed=args.seed)
    train_dataset = dp.get_train_dataset()
    validation_dataset = dp.get_val_dataset()
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)        

    model = models.resnet50(pretrained=False)
    # Reset the last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUMBER_CHART_TYPES)
    model = model.to(device)
    print("\n\n# Parameters: ", sum([param.nelement() for param in model.parameters()]))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    wandb.watch(model)

    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        print('epoch ' + str(epoch))
        train_step(model, device, train_loader, optimizer, criterion, epoch)
        test_step(model, device, valid_loader, criterion)
        
    np.save("weights", model.state_dict())
    wandb.save("weights.npy")

    
if __name__ == '__main__':
    main()