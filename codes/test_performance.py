import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim
import torch.nn as nn

def main():
    # Define transformations and load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True).to(device)
    num_epochs = 5
    batch_size = 64

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Try different num_workers values and measure performance
    for num_workers in [2, 4, 6]:  # Test with 0, 2, 4, 6 workers
        print(f"\nTesting with num_workers={num_workers}...")

        # Initialize DataLoader with different num_workers
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        # Training loop
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_time = time.time() - start_time
        print(f"Epoch time: {epoch_time:.2f}s")
        print(f"Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    main()
