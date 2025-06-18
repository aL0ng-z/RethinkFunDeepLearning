from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import pandas as pd


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.hidden_layer1 = nn.Linear(input_dim, 8)
        self.hidden_layer2 = nn.Linear(8, 4)
        self.output_layer = nn.Linear(4, 1)

    def forward(self, inputs):
        a1 = torch.relu(self.hidden_layer1(inputs))
        a2 = torch.relu(self.hidden_layer2(a1))
        outputs = torch.sigmoid(self.output_layer(a2))
        return outputs


class TitanicDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.mean = {
            "Pclass": 2.236695,
            "Age": 29.699118,
            "SibSp": 0.512605,
            "Parch": 0.431373,
            "Fare": 34.694514,
            "Sex_female": 0.365546,
            "Sex_male": 0.634454,
            "Embarked_C": 0.182073,
            "Embarked_Q": 0.039216,
            "Embarked_S": 0.775910,
        }

        self.std = {
            "Pclass": 0.838250,
            "Age": 14.526497,
            "SibSp": 0.929783,
            "Parch": 0.853289,
            "Fare": 52.918930,
            "Sex_female": 0.481921,
            "Sex_male": 0.481921,
            "Embarked_C": 0.386175,
            "Embarked_Q": 0.194244,
            "Embarked_S": 0.417274,
        }

        self.data = self._load_data()
        self.feature_size = len(self.data.columns) - 1

    def _load_data(self):
        df = pd.read_csv(self.file_path)
        df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
        df = df.dropna(subset=["Age"])
        df = pd.get_dummies(df, columns=["Sex", "Embarked"], dtype=int)
        base_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_female"]

        for i in range(len(base_features)):
            df[base_features[i]] = (df[base_features[i]] - self.mean[base_features[i]]) / self.std[base_features[i]]
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.drop(columns=["Survived"]).iloc[idx].values
        label = self.data["Survived"].iloc[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


train_dataset = TitanicDataset(r"E:\电子书\RethinkFun深度学习\data\titanic\train.csv")
validation_dataset = TitanicDataset(r"E:\电子书\RethinkFun深度学习\data\titanic\validation.csv")

model = NeuralNetwork(train_dataset.feature_size)
model.to("cuda")
model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

epochs = 200

for epoch in range(epochs):
    correct = 0
    step = 0
    total_loss = 0
    for features, labels in DataLoader(train_dataset, batch_size=256, shuffle=True):
        step += 1
        features = features.to("cuda")
        labels = labels.to("cuda")
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        correct += torch.sum(((outputs >= 0.5) == labels))
        loss = torch.nn.functional.binary_cross_entropy(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / step:.4f}')
    print(f'Training Accuracy: {correct / len(train_dataset)}')

model.eval()
with torch.no_grad():
    correct = 0
    for features, labels in DataLoader(validation_dataset, batch_size=256):
        features = features.to("cuda")
        labels = labels.to("cuda")
        outputs = model(features).squeeze()
        correct += torch.sum(((outputs >= 0.5) == labels))
    print(f'Validation Accuracy: {correct / len(validation_dataset)}')
