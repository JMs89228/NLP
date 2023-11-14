import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Read in the data
df = pd.read_csv(r"winequality-red.csv", sep=";")

# Remove the nan values
df = df.dropna()

# Remove the duplicates
df = df.drop_duplicates()

# Convert the data type from object to float
df = df.astype(float)

# Pearson Correlation Analysis
corr = df.corr(method='pearson')


# Data transformation with (x - mean)/(standard deviation) for all features except the quality
for col in df.columns:
    if col != 'quality':
       df[col] = (df[col] - df[col].mean()) / df[col].std()
       df[col] = df[col].astype(float)

# Map the quality to 6 classes
df['quality'] = df['quality'].map({3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5})

# Transform and sort features that are arranged based on the result of Pearson Correlation Analysis
df = df[['alcohol', 'sulphates', 'citric acid', 'fixed acidity', 'pH', 'residual sugar', 'free sulfur dioxide',
            'total sulfur dioxide', 'chlorides', 'density', 'volatile acidity', 'quality']]

# split the data into training set and testing set with proportion of 80% and 20%
train_data = df.sample(frac=0.8, random_state=0)
test_data = df.drop(train_data.index)


# Extract the input features and target variable (quality) from the training set and testing set
X = train_data.iloc[:, :-1].values # Input features
y = train_data.iloc[:, -1].values # Target variable
X_test = test_data.iloc[:, :-1].values # Input features
y_test = test_data.iloc[:, -1].values # Target variable

X = X.reshape(-1, 1, 11)
X_test = X_test.reshape(-1, 1, 11)

# Convert the np.ndarray to tensor
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# Create 1 dimensional convolutional layer and ouptut the flatten array, then pass through the 4 layers of deep neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(16 * 9, 64)  # Adjust input size for flattened data
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 11)
    
    def forward(self, x):
        x = self.conv1d(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

# Create an instance of the model
model1 = Net()

# Calculate the accuracy of the model
def accu_fn(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=1).flatten()
    return torch.sum(y_pred == y_true).item() / len(y_true)

# Define the loss function with the shape [batch_size, 6]
def loss_function(y_pred, y_true):
    return F.cross_entropy(y_pred, y_true)

# Train the neural network
def train(model, X_train, y_train, X_test, y_test, epochs=300, lr=0.0001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    train_accu = []
    test_accu = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = loss_function(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_accu.append(accu_fn(outputs, y_train))
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = loss_function(test_outputs, y_test)
            test_losses.append(test_loss.item())
            test_accu.append(accu_fn(test_outputs, y_test))
        print(f"Epoch {epoch} \t Train Loss: {loss.item():.4f} \t Test Loss: {test_loss.item():.4f}" )
    return train_losses, test_losses, train_accu, test_accu

train_losses, test_losses, train_accu, test_accu = train(model1, X_train, y_train, X_test, y_test, epochs=300, lr=0.001)
print(f"Final Training Accuracy: {train_accu[-1]:.4f} \t Final Testing Accuracy: {test_accu[-1]:.4f}")

# Plot the training loss and testing loss, then save the figure
def fig_training_loss(train_losses, test_losses):
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, label="Training loss")
    plt.plot(test_losses, label="Testing loss")
    plt.legend()
    plt.title("Training Loss vs Testing Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("model1_training_loss.png")

# Plot the training accuracy and testing accuracy, then save the figure
def fig_training_accu(train_accu, test_accu):
    plt.figure(figsize=(10, 7))
    plt.plot(train_accu, label="Training accuracy")
    plt.plot(test_accu, label="Testing accuracy")
    plt.legend()
    plt.title("Training Accuracy vs Testing Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("model1_training_accuracy.png")

fig_training_loss(train_losses, test_losses)
fig_training_accu(train_accu, test_accu)

