import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
# from torchsummary import summary
from torchvision import datasets, transforms

# Define transformation for MNIST dataset
transform = transforms.ToTensor()

UseLeNet = False # If false, use Multi Layer CNN instead

train_data = datasets.MNIST(
    root="data",
    train=True, 
    download=True, 
    transform=transform
    )

test_data = datasets.MNIST(
    root="data",
    train=False, 
    download=True, 
    transform=transform
    )

# Load MNIST dataset using torchvision
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=128, 
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=128, 
    shuffle=True
)

for X, y in test_loader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    idx = torch.randint(len(test_data), size=(1,)).item()
    img, label = test_data[idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

if not UseLeNet:
    class MultiLayerCNN(nn.Module):
        def __init__(self):
            super(MultiLayerCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(10, 20, 5)
            self.fc = nn.Linear(320, 10)
    
        def forward(self, x):
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = x.view(-1, 320)
            x = nn.functional.log_softmax(self.fc(x), dim=1)
            return x
    
    model = MultiLayerCNN()

# Display model summary using torchsummary
# summary(net, input_size=(1, 28, 28))

# Define transformation for CIFAR10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if UseLeNet:
    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
    
        def forward(self, x):
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = LeNet()
    
def train(model, train_loader, test_loader, epochs, optimizer, loss_fn):
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += loss_fn(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {100 * correct / total:.2f}%')

    return train_losses, test_losses

# Display model summary using torchsummary
# summary(net, input_size=(3, 32, 32))

opt = torch.optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
hist = train(model, train_loader, test_loader, epochs=3, optimizer=opt, loss_fn=nn.CrossEntropyLoss())