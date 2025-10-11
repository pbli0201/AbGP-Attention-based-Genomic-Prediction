#####################  AbGP model and ablation study

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler


class GtoP_dataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class AbGP(nn.Module):
    def __init__(self, input_size, output_size):
        super(AbGP, self).__init__()
        self.GtoP_model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Flatten(),
            nn.Linear(16 * (input_size // 4), 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.6),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.Linear(32, output_size),
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        return self.GtoP_model(x)


class conv_xr(nn.Module):
    def __init__(self, input_size, output_size):
        super(conv_xr, self).__init__()
        self.conv_xr_model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Flatten(),
            nn.Linear(32 * (input_size // 2), 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.6),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.Linear(32, output_size),
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        return self.conv_xr_model(x)



class maxpooling_xr(nn.Module):
    def __init__(self, input_size, output_size):
        super(maxpooling_xr, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_size)  
            dummy_output = self.conv_layers(dummy_input)  
            flattened_size = dummy_output.numel()  
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_size, 256)  
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)

        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = x.unsqueeze(1)  
        x = self.conv_layers(x) 
        x = self.flatten(x) 
        x = self.fc1(x)  
        x = self.dropout(x)
        x = self.fc2(x)  
        x = self.fc3(x)  
        x = self.fc4(x)  
        return x



class batch_xr(nn.Module):
    def __init__(self, input_size, output_size):
        super(batch_xr, self).__init__()
        self.batch_xr_model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Flatten(),
            nn.Linear(16 * (input_size // 4), 256),
            nn.ReLU(),
            nn.Dropout(0.6),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.Linear(32, output_size),
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        return self.batch_xr_model(x)


class dr_xr(nn.Module):
    def __init__(self, input_size, output_size):
        super(dr_xr, self).__init__()
        self.dr_xr_model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Flatten(),
            nn.Linear(16 * (input_size // 4), 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.Linear(32, output_size),
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        return self.dr_xr_model(x)


class linear_xr(nn.Module):
    def __init__(self, input_size, output_size):
        super(linear_xr, self).__init__()
        self.linear_xr_model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Flatten(),
            nn.Linear(16 * (input_size // 4), 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        return self.linear_xr_model(x)


class activation_xr(nn.Module):
    def __init__(self, input_size, output_size):
        super(activation_xr, self).__init__()
        self.activation_xr_model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Flatten(),
            nn.Linear(16 * (input_size // 4), 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.6),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.Linear(32, output_size),
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        return self.activation_xr_model(x)


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.unsqueeze(1).to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def test_loop(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def predict_and_evaluate(dataloader, model):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            pred = model(X).cpu().numpy()
            predictions.extend(pred.flatten())
            actuals.extend(y.numpy())
    correlation, _ = pearsonr(predictions, actuals)
    return correlation, predictions, actuals


pl.seed_everything(9527)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



phe_1496 = pd.read_csv('HX-B_phe_20.csv', index_col=0)
phe_1496.replace(' ', np.nan, inplace=True)
phe_1496 = phe_1496.apply(pd.to_numeric, errors='coerce')
Genotypes_1496 = pd.read_pickle('HX-B_(1.25%)_20.csv')


Phenotypes = pd.DataFrame(phe_1496.iloc[:, 7])
missing_phe_index = Phenotypes[Phenotypes.isna().any(axis=1)].index

Genotypes = Genotypes_1496.drop(index=missing_phe_index)
phe = Phenotypes.drop(index=missing_phe_index)


GtoP_X_train, GtoP_X_test, GtoP_y_train, GtoP_y_test = train_test_split(
    Genotypes, phe, test_size=0.2, )


scaler = StandardScaler()
GtoP_y_train = scaler.fit_transform(GtoP_y_train.values.reshape(-1,1)).ravel()
GtoP_y_test  = scaler.transform(GtoP_y_test.values.reshape(-1,1)).ravel()

GtoP_y_train = pd.Series(GtoP_y_train)
GtoP_y_test  = pd.Series(GtoP_y_test)

batch_size = 90 
train_dataset = GtoP_dataset(GtoP_X_train, GtoP_y_train)
test_dataset  = GtoP_dataset(GtoP_X_test,  GtoP_y_test)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)


model_list = [
    ("AbGP",    AbGP),
    ("conv_xr",         conv_xr),
    ("maxpooling_xr",   maxpooling_xr),
    ("batch_xr",        batch_xr),
    ("dr_xr",           dr_xr),
    ("linear_xr",       linear_xr),
    ("activation_xr",   activation_xr),
]

epochs = 5
learning_rate = 1e-4
loss_fn = nn.MSELoss()

all_results = {}
num_repeat = 5  

for name, model_class in model_list:
    print("=========================================")
    print(f"start training: {name}")
    
    corr_list = []
    train_loss_history_all = []
    test_loss_history_all  = []
    
    for run_idx in range(num_repeat):
        print(f"  [Run {run_idx+1}/{num_repeat}]")
        
        model = model_class(input_size=GtoP_X_train.shape[1], output_size=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            train_loss = train_loop(train_loader, model, loss_fn, optimizer)
            test_loss  = test_loop(test_loader,  model, loss_fn)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            print(f"    Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")

        correlation, predictions, actuals = predict_and_evaluate(test_loader, model)
        corr_list.append(correlation)
        
        train_loss_history_all.append(train_losses)
        test_loss_history_all.append(test_losses)

        print(f"model training Pearson r in {run_idx+1}  = {correlation:.4f}")
    
    mean_corr = np.mean(corr_list)
    print(f"model {name} mean Pearson r  = {mean_corr:.4f}\n")
    
    all_results[name] = {
        "correlations": corr_list,                   
        "train_losses": train_loss_history_all,      
        "test_losses":  test_loss_history_all,       
        "mean_corr":    mean_corr
    }


LW = pd.DataFrame(all_results)
LW.to_csv('TB_xr.csv')









