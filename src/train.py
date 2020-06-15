"""
Train an image classifier on the CHART-Synthetic dataset
run: CUDA_VISIBLE_DEVICES=X python train.py --dataset-path /path1 --labels-path /path2 --out=dir /path3
     where X=0 or X=1 or ... X=0,..,n depending on the GPUs available
     check args below for more parameters
last run:
CUDA_VISIBLE_DEVICES=1,2 python src/train.py --batch-size 64 --test-batch-size 64 --epochs 2 --lr 1e-3 --num-output-classes 10 --dataset-path /mnt/chart/raw/images --labels-path /mnt/chart-classification/labels_task1.csv --out-dir /mnt/chart/output --infra compass --architecture resnet50
"""

from ExperimentalRun import ExperimentalRun
from datapreprocessing import DataProprocessing
from torchvision import models
import argparse
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser(description='Chart Image Classification')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')    
    parser.add_argument('--seed', type=int, default=443, metavar='S',
                        help='random seed (default: 443)')
    #parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
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
    parser.add_argument('--project_name', type=str, default='chart', 
                        help='Project name on Wandb')
    parser.add_argument('--architecture', type=str, default='', 
                    help='Highlevel description of the deep learning model used')
    parser.add_argument('--infra', type=str, default='', 
                    help='Description of the infrastructure used for training')
    parser.add_argument('--notes', type=str, default='', 
                    help='Any particular note about the run')
    
    args = parser.parse_args()
    
    classes = ['Donut', 'Grouped horizontal bar', 'Grouped vertical bar',
               'Horizontal box', 'Line', 'Pie', 'Scatter',
               'Stacked horizontal bar', 'Stacked vertical bar', 'Vertical box']
    provider = DataProprocessing(args.dataset_path, None, args.labels_path, seed=args.seed)
    
    model = models.resnet50(pretrained=False)
    # Reset the last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.num_output_classes)
    
    run = ExperimentalRun(model, provider, args, notes=args.notes, tags=['chart', 'image classification'])
    run.train()
    
if __name__ == '__main__':
    main()