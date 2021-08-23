import torch
import numpy as np
from dataset import BrainDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from convnet import ConvNet
from sklearn.metrics import accuracy_score, precision_score, f1_score
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse

def metrics(predicted, true, summary, step):
    
    scores_dir = args.scoresDir or "../scores/" 
    summary.add_scalar(os.path.join(scores_dir, 'accuracy_score'), accuracy_score(predicted, true), global_step=step)
    summary.add_scalar(os.path.join(scores_dir, 'precision'), precision_score(predicted, true), global_step=step)
    summary.add_scalar(os.path.join(scores_dir, 'f1_score'), f1_score(predicted, true), global_step=step)

def train():
    
    data_dir = args.dataDir or "../data/"
    runs_dir = args.runsDir or "../runs/"
    models_dir = args.modelsDir or "../models"

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda")
    data = BrainDataset(os.path.join(data_dir, 'yes'), 
                        os.path.join(data_dir, 'no'), transform = transforms.Compose([
                        transforms.Resize((256,256)),
                        lambda x: x/255.]))

    size = len(data)
    splits = (np.array([0.8, 0.19, 0.01])*size).astype('i')
    splits[0] += size - splits.sum()

    train_dataset, test_dataset, val_dataset = torch.utils.data.dataset.random_split(data, splits)

    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=0, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=0, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=0, shuffle=False)

    cnn = ConvNet()
    cnn.to(device)

    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    writer = SummaryWriter(os.path.join(runs_dir, f'cnn/{datetime.now().strftime("%Y%m%d_%H%M%S")}'))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=5e-4)

    epochs = 5

    itr = 1
    for epoch in range(epochs):
        cnn.train()
        print(f'\nEpoch: {epoch+1}')
    
        for images, labels, _ in train_dataloader:
            
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = cnn(images)
            loss = criterion(outputs, labels.squeeze())
            
            loss.backward()
            optimizer.step()
            
            del(images)
            del(labels)
            
            if itr % 100 == 0:
                preds, trues = [], []
                for i, j, _ in val_dataloader:
                    pred = cnn.eval()(i.to(device))
                    preds.append(F.softmax(pred, dim=1).detach().cpu().numpy().argmax(1))
                    trues.append(j.detach().cpu().numpy().astype('i'))
                metrics(np.array(preds).flatten(), np.array(trues).flatten(), writer, itr)
            itr += 1
            print(f'Iter: {itr:10}', end='\r')
        torch.save({
                'epoch': epoch,
                'model_state_dict': cnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(models_dir,f'cnn_ckpt_{epoch}.pth')
        )
            
    torch.save(cnn.state_dict(), os.path.join(models_dir,'cnn_model.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoresDir", type=str,help="Path to the scores directory")
    parser.add_argument("--dataDir", type=str,help="Path to the data directory")
    parser.add_argument("--runsDir", type=str,help="Path to the runs directory")
    parser.add_argument("--modelsDir", type=str,help="Path to the models directory")
    args = parser.parse_args()

   
    train()