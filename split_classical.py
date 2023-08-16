# Classical library imports
import torch
from torch import nn
import random
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import KFold
from statistics import mean

# Assigning the seed value for the random function
SEED = 61096
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = '0'

# Checking if GPU is available or not
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print("GPU runtime selected, GPU device name:", torch.cuda.get_device_name())
else:
  print("No GPU runtime, running on CPU mode")

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Assigning learning rate for model and epochs
num_data = 1000
lr = 0.001
epochs = 100

def extract_data(fold):
    train_path = os.path.join(f"folder{fold}", "train.csv")
    test_path = f"folder{fold}/test.csv"
    features = ['CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'REBenign', 'MinREBotnets', 'Entropy',
                'InformationRadius']

    train_set = pd.read_csv(train_path, header=None, encoding='utf-8')
    test_set = pd.read_csv(test_path, header=None, encoding='utf-8')

    x_train = train_set.iloc[:,:-1].to_numpy()
    y_train = train_set.iloc[:,-1].to_numpy()

    x_test = test_set.iloc[:, :-1].to_numpy()
    y_test = test_set.iloc[:,-1].to_numpy()

    return x_train, y_train, x_test, y_test


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
        )

    def forward(self, x):
        features = self.encoder(x)
        return features
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(4,4),
            nn.ReLU(),
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        output = self.decoder(features)
        return output

batch_size = 32
num_folds = 5

#%%
# Define cost function
criterion = nn.BCELoss()

client_models = []
server_models = []
train_loss_store = [[] for i in range(num_folds)]
train_acc_store = [[] for i in range(num_folds)]

test_loss_store = [[] for i in range(num_folds)]
test_acc_store = [[] for i in range(num_folds)]
overall_confusion_matrix_store = [[] for i in range(num_folds)]
for fold in range(1, num_folds + 1):
    client_model = Encoder().to(device)
    optimizer_encoder = torch.optim.Adam(client_model.parameters(), lr=lr)
    server_model = Decoder().to(device)
    optimizer_decoder = torch.optim.Adam(server_model.parameters(), lr=lr)
    print(f"Fold {fold}/{num_folds}")

    # Split data into training and validation sets for this fold
    x_train, y_train, x_test, y_test = extract_data(fold)
    for epoch in range(epochs):
        # Step 1: Client-side Computation
        client_model.train()

        server_model.train()
        # optimizer_decoder.zero_grad()
        for i in range(0, len(x_train), batch_size):
            inputs = torch.tensor(x_train[i:i + batch_size]).to(torch.float32).to(device)
            targets = torch.tensor(y_train[i:i + batch_size]).to(torch.float32).to(device)
            client_features = client_model(inputs).to(device)
            client_fx = (client_features.clone().detach().requires_grad_(True))
            optimizer_encoder.zero_grad()
            # Step 2: Server-side computation (decoder)
            server_output = server_model(client_fx)
            server_output = server_output.flatten().to(device)
            optimizer_decoder.zero_grad()

            loss = criterion(server_output, targets)
            loss.backward()
            dfx_client = client_fx.grad.clone().detach()
            optimizer_decoder.step()
            # Backward Propagation to Client
            client_features.backward(dfx_client)
            optimizer_encoder.step()
        # Evaluate model on validation set
        client_model.eval()
        server_model.eval()
        val_correct = 0
        val_loss = 0.0
        pred_labels = []
        true_labels = []
        test_loss_collect = []
        with torch.no_grad():

            for i in range(0, len(x_test), batch_size):
                inputs = torch.tensor(x_test[i:i + batch_size]).to(torch.float32).to(device)
                targets = torch.tensor(y_test[i:i + batch_size]).to(torch.float32).to(device)

                client_features = client_model(inputs).to(device)

                server_output = server_model(client_features)
                server_output = server_output.flatten().to(device)
                # Compute loss
                loss = criterion(server_output, targets)
                test_loss_collect.append(loss.item() * inputs.size(0))

                # Accuracy
                predicted_labels = torch.round(server_output)
                pred_labels.extend(predicted_labels.cpu().numpy())
                true_labels.extend(targets.cpu().numpy())

            test_acc = accuracy_score(true_labels, pred_labels)
            test_acc_store[fold - 1].append(test_acc)
            test_loss_store[fold - 1].append(mean(test_loss_collect))
            print(f"                      : Test Accuracy = {test_acc}, Test loss = {mean(test_loss_collect)}")
        fold_conf_matrix = confusion_matrix(true_labels, pred_labels)
    overall_confusion_matrix_store[fold - 1].append(fold_conf_matrix)
    client_models.append(client_model)
    server_models.append(server_model)
print("Overall Confusion Matrix:")
print(overall_confusion_matrix_store)
