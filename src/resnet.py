from __future__ import print_function
import argparse
import os
import torch
import torch.optim as optim
import wandb
import numpy as np
from tqdm import tqdm

from datapreprocessing import DataProprocessing
from EarlyStopping import EarlyStopping
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn


def train_step(model, device, train_loader, optimizer, criterion, epoch, t):
    model.train()
        
    train_loss = 0
    n_iter = 0
    
    total = 0
    correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total += labels.size(0)
        correct += torch.sum(predicted == labels.data)
        train_loss += loss.item()
        n_iter += 1
        
        t.postfix[0]["train_it"] = n_iter
        t.update()
    
    epoch_acc = 100 * correct / total
    average_loss = train_loss / n_iter
    print('\tTrain Loss: {:.4f}'.format(average_loss))
    print('\tTrain Acc: {:.4f}'.format(epoch_acc))
    wandb.log({
        "Train Loss": average_loss,
        "Train Acc": epoch_acc
    })
    return average_loss


def test_step(model, device, test_loader, criterion, early_stopping, t):
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
        
        t.postfix[1]["val_it"] = valid_n_iter
        t.update()
    
    epoch_acc = 100 * correct / total
    average_loss = valid_loss / valid_n_iter
    print('\tValid Loss: {:.4f}'.format(average_loss))
    print('\tValid Acc: {:.4f}'.format(epoch_acc))
    
    stop_execution = False
    early_stopping(average_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        stop_execution = True
    
    wandb.log({
        "Test Accuracy": epoch_acc,
        "Test Loss": average_loss})
    
    return stop_execution


def main():
    parser = argparse.ArgumentParser(description='PyTorch Resnet50 for Chart Classification')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')    
    parser.add_argument('--seed', type=int, default=443, metavar='S',
                        help='random seed (default: 443)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--patience', type=int, default=5, metavar='N',
                        help='how many epochs to wait before stopping a non-improving execution')
    parser.add_argument('--num-workers', type=int, default=16, metavar='N',
                        help='processors for data loading tasks')
    parser.add_argument('--num-output-classes', type=int, default=10, metavar='N',
                        help='number of classes in the dataset')
    parser.add_argument('--dataset-path', type=str, default='./data', 
                        help='location of input images')
    parser.add_argument('--labels-path', type=str, default='../labels_task1.csv', 
                        help='location of labels for task')
    parser.add_argument('--out-dir', type=str, default='./outputs', 
                        help='location for output data')
    
    args = parser.parse_args()
    
    run = wandb.init(project="chart-type")
    wandb.config.update(args)
    
    wandb.run.save()
    run_name = wandb.run.name
    output_dir_name = os.path.join(args.out_dir, run_name)
    os.makedirs(output_dir_name, exist_ok=True)

    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:1" if use_gpu else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_gpu else {}
    
    TASK1_FILEPATH = args.labels_path
    TRAIN_IMAGES_DIR = args.dataset_path
    
    dp = DataProprocessing(TRAIN_IMAGES_DIR, None, TASK1_FILEPATH, seed=args.seed)
    train_dataset = dp.get_train_dataset()
    validation_dataset = dp.get_val_dataset()
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)        

    model = models.resnet50(pretrained=False)
    # Reset the last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.num_output_classes)
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # use gpu 1 and 2
        model = nn.DataParallel(model, [1,2])
    
    model = model.to(device)
    print("\n\n# Parameters: ", sum([param.nelement() for param in model.parameters()]))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    wandb.watch(model)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=os.path.join(args.out_dir, 'checkpoint.pt'))
    
    criterion = nn.CrossEntropyLoss()
    total_iterations = args.epochs * (len(train_loader) + len(valid_loader))
    with tqdm(total=args.epochs, postfix=[dict(train_it=0), dict(val_it=0)]) as t:
        for epoch in range(1, args.epochs + 1):
            train_step(model, device, train_loader, optimizer, criterion, epoch, t)
            stop_execution = test_step(model, device, valid_loader, criterion, early_stopping, t)
            if stop_execution:
                break
        
    torch.save(model.module.state_dict(), os.path.join(output_dir_name, "weights"))
#     wandb.save("weights.npy")


if __name__ == '__main__':
    main()
