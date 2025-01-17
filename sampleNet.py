import torch
from torch.nn import Sequential
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def CreateData(w=5.0, b=2.0):
    X = torch.arange(100, dtype=torch.float32).unsqueeze(1)  # Add dimension for compatibility
    y = X * w + b

    X_train, X_test = X[:80], X[80:100]
    y_train, y_test = y[:80], y[80:100]

    return X_train, X_test, y_train, y_test

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = CreateData()
    model = Model()
    epochs = 100

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Use DataLoader for batching
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    model.parameters()