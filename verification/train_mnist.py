import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import onnx

if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Custom transformation: reshape MNIST images to (4,14,14)
    class ReshapeTransform:
        def __call__(self, img):
            img = torch.tensor(img, dtype=torch.float32) / 255.0  # Normalize
            img = img.reshape(2, 2, 14, 14).permute(2, 0, 3, 1).reshape(4, 14, 14)
            return img

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), ReshapeTransform()])
    train_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = MNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Define a simple neural network with only linear layers
    class MNISTLinear(nn.Module):
        def __init__(self):
            super(MNISTLinear, self).__init__()
            self.fc1 = nn.Linear(4 * 14 * 14, 256)  # Flatten input
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)  # Output layer (10 classes)

        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten (4,14,14) -> (784)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)  # No activation (logits)
            return x

    # Initialize model, loss, and optimizer
    model = MNISTLinear().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training function
    def train(model, loader, epochs=5):
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

    # Evaluation function
    def accuracy(model, loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        print(f"Test Accuracy: {100 * correct / total:.2f}%")

    # Train and evaluate
    train(model, train_loader, epochs=5)
    accuracy(model, test_loader)

    # Export model to ONNX
    onnx_filename = "mnist_linear.onnx"
    dummy_input = torch.randn(1, 4, 14, 14, device=device)  # Match input shape
    torch.onnx.export(model, dummy_input, onnx_filename,
                      input_names=["input"], output_names=["output"])

    print(f"Model saved as {onnx_filename}")
