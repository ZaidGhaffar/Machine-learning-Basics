import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score
from tqdm import tqdm

PATH_LOD_DIR = r""
PATH_CSV = r""

class GettingPreprocessData(Dataset):
    def __init__(self,path_csv):
        super().__init__()
        self.df = path_csv
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor(row[:-1],dtype=torch.float32)
        Labels  = torch.tensor(row[-1],dtype=torch.float32)
        return (features,Labels)
    
    
def DataLoader(path_csv,batch_size=64):
    gettingdata = GettingPreprocessData(path_csv)
    train_split,test_split = train_test_split(gettingdata,test_split=0.2)
    train_loader = DataLoader(
        train_split,
        work_num = 2,
        shuffle=True,
        batch_size = batch_size
    )
    
    Test_loader = DataLoader(
        test_split,
        work_num = 2,
        shuffle=True,
        batch_size = batch_size
    )
    return train_loader,Test_loader



class ModelBuilding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear((3,5)),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self,x):
        return self.model(x)
    
    
class Trainer:
    def __init__(self,path_csv,batch_size=10):
        self.device = torch.cuda("cuda" if torch.cuda.is_available() else "cpu")
        self.Train_loader,self.Val_loader = DataLoader(path_csv,batch_size)
        self.model = ModelBuilding()
        self.loss_fun = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.mdoel.parameters(),lr=0.001)
        self.writer = SummaryWriter(PATH_LOD_DIR)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=2,
            factor=0.1
            mode= "min"
        )
        
        
    def Train_epochs(self):
        self.model.train()
        all_epochs = []
        all_predic = []
        all_labels = []
        for x,y in self.Train_loader:
            x,y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            yhat = self.model(y).squeeze()
            loss = self.loss_fun(yhat,y)
            loss.backward()
            self.optimizer.step()
            
            all_epochs +=loss.item()
            preds = (yhat > 0.5).float()
            all_predic.extend(preds.detach.cpu())
            all_labels.extend(y.detach.cpu())
            
            accuracy = accuracy_score(all_predic,all_labels)
            precision = precision_score(all_predic,all_labels)
            avg_loss = all_epochs/len(self.Train_loader)
            
            
            
            return {
                "precision":   precision ,
                "accuracy":   accuracy,
                "Loss":        avg_loss,
            }
            
    def Val_epochs(self):
        self.model.eval()
        all_epochs = []
        all_predic = []
        all_labels = []
        for x,y in self.Train_loader:
            x,y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            yhat = self.model(y).squeeze()
            loss = self.loss_fun(yhat,y)
            
            
            all_epochs +=loss.item()
            preds = (yhat > 0.5).float()
            all_predic.extend(preds.detach.cpu())
            all_labels.extend(y.detach.cpu())
            
            accuracy = accuracy_score(all_predic,all_labels)
            precision = precision_score(all_predic,all_labels)
            avg_loss = all_epochs/len(self.Train_loader)
            
            
            return {
                "precision":   precision ,
                "accuracy":   accuracy,
                "Loss":        avg_loss,
            }
            
    def train(self,EPOCHS=10):
        best_model = float('inf')
        for epoch in tqdm(range(EPOCHS)):
                traing_metrics = self.Train_epochs()
                val_metrics =  self.Val_epochs()
                print(f"Epoch {epoch+1}/{EPOCHS}")
                print(f"training metrics ->  \n  precision:  Loss: {traing_metrics["Loss"]}\n  {traing_metrics["precision"]} \n accuracy: {traing_metrics["accuracy"]}")
                print("\n")
                print("\n")
                print(f"Val metrics ->  \n  precision:  Loss: {val_metrics["Loss"]}\n  {val_metrics["precision"]} \n accuracy: {val_metrics["accuracy"]}")
                
                
                self.writer.add_scalar("train/loss",traing_metrics["Loss"],epoch)
                self.writer.add_scalar("Loss/Validation", val_metrics["Loss"], epoch)
                self.writer.add_scalar("Accuracy/Train", traing_metrics["acuracy"], epoch)
                self.writer.add_scalar("Accuracy/Validation", val_metrics["accuracy"], epoch)
                self.writer.add_scalar("Precision/Train", traing_metrics["precision"], epoch)
                self.writer.add_scalar("Precision/Validation", val_metrics["precision"], epoch)
                
                self.scheduler.step(val_metrics["Loss"])
                if val_metrics["Loss"] <best_model:
                    best_model = val_metrics["Loss"]
                    torch.save(self.model.state_dict(),"model.pt")
                    print(f"New best model saved with validation loss: {best_model:.4f}")
                    
                    
                    
if __name__ == "__main__":
    trainer = Trainer(PATH_CSV)
    trainer.train()
    
                
                
        
