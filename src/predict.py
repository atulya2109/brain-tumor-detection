import torch
from convnet import ConvNet
from torchvision.io import read_image
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse

def get_prediction(model_path,image_path):
    '''
    Returns the prediction for the given image
    '''
    cnn = ConvNet()
    device = torch.device("cuda")
    
    x = torch.load(model_path)
    cnn.load_state_dict(x)

    img = read_image(image_path)/255.
    plt.imshow(img.permute(1,2,0).numpy())
    img = transforms.Resize((256,256))(img)
    img.to(device)
    
    res = nn.functional.softmax(cnn.eval()(img[None]), dim=-1).detach().cpu().numpy().flatten()
    
    return "Yes" if res.argmax() == 1 else "No"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("image_path")
    args = parser.parse_args()
    print(get_prediction(args.model_path,args.image_path))