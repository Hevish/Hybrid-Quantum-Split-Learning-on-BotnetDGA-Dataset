# Classical library imports
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import pennylane as qml

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

# Number of Clients
N=4

n_qubits=4
n_layers = 1

# Create device
def create_qc():
    dev = qml.device('default.qubit', wires=n_qubits)
    weight_shapes = {"weights": (n_layers, n_qubits*3)}

    # Create Circuit with parameter shifting backpropagation method
    @qml.qnode(dev, diff_method='parameter-shift', interface = 'torch')
    def qnode(inputs, weights):
        for i in range(n_qubits):
            qml.RX(inputs[i], wires=i)

        for i in range(n_qubits):
            qml.RZ(weights[0][3*i], wires=i)
            qml.RY(weights[0][3*i+1], wires=i)
            qml.RZ(weights[0][3*i+2], wires=i)

        for i in range(n_qubits):
            qml.CZ(wires=[i, (i+1)%n_qubits])
        return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]

    # Create quantum layer using TorchLayer
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes).cuda(device)

    # Display Circuit
    weights = np.random.uniform(low=0, high=2*np.pi, size=weight_shapes["weights"])
    print(qml.draw_mpl(qnode)(inputs = torch.tensor([0.2,0.1,0.3,0.1,0.2,0.2]), weights = weights))
    print("Number of Parameters to be optimized within quantum layer: ",len(qnode.qtape.get_parameters()))
    return qlayer
qlayer = create_qc()

def extract_data(fold):
    train_path = os.path.join(f"folder{fold}", "train.csv")
    test_path = os.path.join(f"folder{fold}", "test.csv")
    features = ['CharLength', 'TreeNewFeature', 'nGramReputation_Alexa', 'REBenign', 'MinREBotnets', 'Entropy',
                'InformationRadius']

    train_set = pd.read_csv(train_path, header=None, encoding='utf-8')
    test_set = pd.read_csv(test_path, header=None, encoding='utf-8')

    x_train = train_set.iloc[:, :-1].to_numpy()
    y_train = train_set.iloc[:, -1].to_numpy()

    x_test = test_set.iloc[:, :-1].to_numpy()
    y_test = test_set.iloc[:, -1].to_numpy()
    print(x_train)
    x_train_clients = [[] for i in range(N)]
    y_train_clients = [[] for i in range(N)]
    num_data_per_client = int(len(x_train) / N)
    idx = 0
    for start in range(0, len(x_train), num_data_per_client):
        end = start + num_data_per_client
        x_train_clients[idx].append(x_train[start:end])
        y_train_clients[idx].append(y_train[start:end])
        idx += 1
    return x_train_clients, y_train_clients, x_test, y_test


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
            qlayer,
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

# Define the encoders and decoder models for each client
client_encoder = Encoder().to(device)
server_decoder = Decoder().to(device)
#%%
# Define cost function
criterion = nn.BCELoss()
# Define Optimizer

optimizer_decoder = torch.optim.Adam(server_decoder.parameters(), lr=lr)
optimizer_encoder = torch.optim.Adam(client_encoder.parameters(), lr=lr)

# Create a nested dictionary to store the results of each fold for each client
fold = 1
columns = ['Epoch','TN', 'FP', 'FN', 'TP']
df = pd.DataFrame(columns=columns)
x_train_clients, y_train_clients, x_test, y_test = extract_data(fold)
for epoch in range(epochs):
    for client in range(N):
        client_train_set = x_train_clients[client]
        client_train_label = y_train_clients[client]
        # To make this random selection later
        client_encoder.train()
        server_decoder.train()
        # optimizer_decoder.zero_grad()
        for i in range(0, len(client_train_set), batch_size):
            inputs = torch.tensor(client_train_set[i:i + batch_size]).to(torch.float32).to(device)
            targets = torch.tensor(client_train_label[i:i + batch_size]).to(torch.float32).to(device)
            client_features = client_encoder(inputs).to(device)
            client_fx = (client_features.clone().detach().requires_grad_(True))
            optimizer_encoder.zero_grad()
            # Step 2: Server-side computation (decoder)
            server_output = server_decoder(client_fx)
            server_output = server_output.flatten().to(device)
            optimizer_decoder.zero_grad()
            loss = criterion(server_output, targets.flatten())
            loss.backward()
            dfx_client = client_fx.grad.clone().detach()
            optimizer_decoder.step()
            # Backward Propagation to Client
            client_features.backward(dfx_client)
            optimizer_encoder.step()

            # End of client's local Epoch
            # Switch to next client
        print(f"End of local epoch for Client{client + 1}")
    print(f"End of global Epoch {epoch + 1}")
    client_encoder.eval()
    server_decoder.eval()
    val_correct = 0
    val_loss = 0.0
    pred_labels = []
    true_labels = []
    test_loss_collect = []
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            inputs = torch.tensor(x_test[i:i + batch_size]).to(torch.float32).to(device)
            targets = torch.tensor(y_test[i:i + batch_size]).to(torch.float32).to(device)

            client_features = client_encoder(inputs).to(device)

            server_output = server_decoder(client_features)
            server_output = server_output.flatten().to(device)
            # Compute loss
            loss = criterion(server_output, targets.flatten())
            test_loss_collect.append(loss.item() * inputs.size(0))

            # Accuracy
            predicted_labels = torch.round(server_output)
            pred_labels.extend(predicted_labels.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
    test_acc = accuracy_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels)
    TN, FP, FN, TP = cm.ravel()
    epoch_data = pd.DataFrame.from_records([{'Epoch': epoch + 1, 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP}])
    df = pd.concat([df, epoch_data])
    # Test accuracy for each epoch on test dataset
    print(f"Epoch {epoch + 1}/{epochs}:Testing accuracy", test_acc)

# Save pd df to xlsx
excel_file = f'results/hybrid_testing_{N}_clients_fold_{fold}.xlsx'
df.to_excel(excel_file, index=False, engine='openpyxl')



