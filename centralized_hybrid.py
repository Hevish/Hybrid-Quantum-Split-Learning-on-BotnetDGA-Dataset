# Classical library imports
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import KFold
from statistics import stdev, mean
import sys
import pennylane as qml

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assigning learning rate for model and epochs
num_data = 1000
lr = 0.001
epochs = 100

# Circuit hyperparameters
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


class Centralized_Hybrid_model(nn.Module):
    def __init__(self):
        super(Centralized_Hybrid_model, self).__init__()
        self.classifier= nn.Sequential(
            nn.Linear(7,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,n_qubits),
            nn.ReLU(),
            qlayer,
            nn.Linear(n_qubits,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32,4),
            nn.ReLU(),
            nn.Linear(4,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        y_hat = self.classifier(x)
        return y_hat


batch_size = 32
num_folds = 5

# Define cost function
criterion = nn.BCELoss()

K = 5
models = []
train_loss_store = [[] for i in range(K)]
train_acc_store = [[] for i in range(K)]

test_loss_store = [[] for i in range(K)]
test_acc_store = [[] for i in range(K)]
overall_confusion_matrix_store = [[] for i in range(K)]
for fold in range(1, K + 1):
    print("For Fold:", fold)
    x_train, y_train, x_test, y_test = extract_data(fold)
    model = Centralized_Hybrid_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    fold_conf_matrix = []
    for epoch in range(epochs):
        model.train()
        train_loss_collect = []
        true_labels = []
        pred_labels = []
        for i in range(0, len(x_train), batch_size):
            inputs = torch.tensor(x_train[i:i + batch_size]).to(torch.float32).to(device)
            targets = torch.tensor(y_train[i:i + batch_size]).to(torch.float32).to(device)
            outputs = model(inputs).flatten()
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            train_loss_collect.append(loss.item() * inputs.size(0))
            loss.backward()
            optimizer.step()
            predictions = torch.round(outputs)
            pred_labels.extend(predictions.detach().cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
        accuracy = accuracy_score(true_labels, pred_labels)
        train_loss_store[fold - 1].append(mean(train_loss_collect))
        train_acc_store[fold - 1].append(accuracy)
        print(f"For epoch {epoch + 1}/{epochs}: Train Accuracy = {accuracy}, Train loss = {mean(train_loss_collect)}")

        # Testing Loop
        model.eval()
        test_loss_collect = []
        test_accuracy_collect = []
        test_pred_labels = []
        test_true_labels = []
        with torch.no_grad():

            for i in range(0, len(x_test), batch_size):
                inputs = torch.tensor(x_test[i:i + batch_size]).to(torch.float32).to(device)
                targets = torch.tensor(y_test[i:i + batch_size]).to(torch.float32).to(device)

                outputs = model(inputs).flatten()
                # Compute loss
                loss = criterion(outputs, targets)
                test_loss_collect.append(loss.item() * inputs.size(0))

                # Accuracy
                predicted_labels = torch.round(outputs)
                test_pred_labels.extend(predicted_labels.cpu().numpy())
                test_true_labels.extend(targets.cpu().numpy())
            test_acc = accuracy_score(test_true_labels, test_pred_labels)
            test_acc_store[fold - 1].append(test_acc)
            test_loss_store[fold - 1].append(mean(test_loss_collect))
            print(f"                      : Test Accuracy = {test_acc}, Test loss = {mean(test_loss_collect)}")
        fold_conf_matrix = confusion_matrix(test_true_labels, test_pred_labels)
    overall_confusion_matrix_store[fold - 1].append(fold_conf_matrix)
    models.append(model)
print(overall_confusion_matrix_store)