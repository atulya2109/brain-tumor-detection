import torch
from torchvision.io import read_image
import os
from torch.utils.data import Dataset

class BrainDataset(Dataset):
    
    def get_image_data(self):
        img_data = [{'file': image, 'label': 1} for image in os.listdir(self.positive_dir)]
        img_data += [{'file': image, 'label': 0} for image in os.listdir(self.negative_dir)]
        return img_data
    
    def __init__(self,positive_dir, negative_dir,transform=None):
        super().__init__()
        self.positive_dir = positive_dir
        self.negative_dir = negative_dir
        self.image_data = self.get_image_data()
        self.transform = transform
        
    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image_name =  self.image_data[idx]['file']
        img_label = torch.tensor([self.image_data[idx]['label']], dtype = torch.int64)
        img_dir = self.positive_dir if img_label == 1 else self.negative_dir
        img_path = os.path.join(img_dir, image_name)
        image = read_image(img_path)
        
        if image.size(0) == 1:
            image = image.expand(3,-1,-1)
            
        if self.transform:
            image = self.transform(image)
    
        return image, img_label, image_name