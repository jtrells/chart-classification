import os
import torch
import skimage
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import json
from datapreprocessing import DataProprocessing

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, transform=None):        
        self.root_dir = images_dir
        self.transform = transform        
        self.images = [x for x in os.listdir(self.root_dir) if x[-3:] == 'png']
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()       
        image = self.read_image(idx)
        if self.transform:
            image = self.transform(image)
        return (image, self.images[idx][:-4])
    
    def read_image(self, idx):
        image_id = self.images[idx]
        img_name = os.path.join(self.root_dir, image_id)
        image = skimage.io.imread(img_name)[:,:,:3]
        return image



def save_json_output(output, out_dir):
    name = output[0]
    label = output[1]
    pred = {
        "task1": {
            "output": {
                "chart_type": output[1]
            }
        }
    }
    with open(os.path.join(out_dir, name + ".json"), 'w') as f:   
        json.dump(pred, f)
        
    
def main():
    test_path = '/mnt/chart/raw/test_release/task1/png'
    output_path = '/mnt/chart/test_results'
    model_weights_path = '/mnt/chart/output/splendid-valley-40/weights'
    
    classes = ['Donut', 'Grouped horizontal bar', 'Grouped vertical bar',
               'Horizontal box', 'Line', 'Pie', 'Scatter',
               'Stacked horizontal bar', 'Stacked vertical bar', 'Vertical box']
    
    provider = DataProprocessing('/mnt/chart/raw/images', None, '/mnt/chart-classification/labels_task1.csv', seed=443)
    train_dataset = provider.get_train_dataset()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = TestDataset(test_path, transform=transform)
    
    loader = DataLoader(test_dataset, batch_size=128, num_workers=16)
    
    # load model
    device = torch.device("cuda:0")
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    model.load_state_dict(torch.load(model_weights_path))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)
    model.eval() 
    
    for images, names in loader:
        images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        labs = train_dataset.codec.inverse_transform(predicted.cpu())
        for x in zip(names, labs):
            save_json_output(x, output_path)

if __name__ == '__main__':
    main()