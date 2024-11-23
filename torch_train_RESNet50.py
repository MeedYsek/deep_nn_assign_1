import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import UnidentifiedImageError
from tqdm import tqdm
import timm  # Required for PNASNet-5
import json  # For saving metrics
from torch.cuda.amp import autocast, GradScaler  # Mixed precision training
from torch.optim.lr_scheduler import StepLR  # Added learning rate scheduler

# Directories
train_dir = "data/train"
val_dir = "data/val"
test_dir = "data/test"

# Image parameters
IMG_SIZE = 224  # Optimized size for ResNet50 (224x224)
BATCH_SIZE = 16  # Batch size set to 16 for your GPU
NUM_CLASSES = 50  # Replace with the actual number of classes in your dataset
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # Keep number of workers as 4

# Data Transformations
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Custom Dataset Class to Handle Corrupted and Missing Images
class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(SafeImageFolder, self).__init__(root, transform)
        self.skipped_files = 0

    def __getitem__(self, index):
        try:
            return super(SafeImageFolder, self).__getitem__(index)
        except (UnidentifiedImageError, FileNotFoundError) as e:
            print(f"Error loading file {self.imgs[index][0]}: {e}")
            self.skipped_files += 1
            return None

# Custom collate function to handle missing values in batch
def custom_collate_fn(batch):
    # Filter out None values (corrupted or missing files)
    batch = [item for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == '__main__':
    # Datasets and Data Loaders
    train_dataset = SafeImageFolder(train_dir, transform=train_transforms)
    val_dataset = SafeImageFolder(val_dir, transform=val_transforms)

    # Filter out None values from the dataset
    train_dataset.samples = [s for s in train_dataset.samples if s is not None]
    val_dataset.samples = [s for s in val_dataset.samples if s is not None]

    # Data Loaders with multi-threaded data loading
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                               collate_fn=custom_collate_fn,
                                               num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                             collate_fn=custom_collate_fn,
                                             num_workers=NUM_WORKERS, pin_memory=True)

    # Load Pretrained ResNet50 Model
    model = timm.create_model('resnet50', pretrained=True)  # Load ResNet50

    # Modify the final layer for the number of classes
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Learning rate set to 1e-4

    # Learning rate scheduler (StepLR)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Decrease LR by a factor of 10 every 5 epochs

    # Initialize variables to store best model and metrics
    best_val_acc = 0.0
    training_metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Enable cuDNN auto-tuner to find the best algorithm
    torch.backends.cudnn.benchmark = True

    # Initialize scaler for mixed precision training
    scaler = GradScaler()

    # Training and Validation Loop with Progress Bars
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS} - Training") as pbar:
            for batch_idx, batch in enumerate(train_loader):
                if batch is None or any(item is None for item in batch):
                    continue
                inputs, targets = batch
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                optimizer.zero_grad()
                
                # Mixed Precision: Use autocast to reduce memory usage and speed up training
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                # Backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Accuracy": f"{100. * correct / total:.2f}%"})
                pbar.update(1)

        train_acc = 100. * correct / total
        training_metrics["train_loss"].append(train_loss / len(train_loader))
        training_metrics["train_acc"].append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{EPOCHS} - Validation") as pbar:
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch is None or any(item is None for item in batch):
                        continue
                    inputs, targets = batch
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    # Update progress bar
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Accuracy": f"{100. * correct / total:.2f}%"})
                    pbar.update(1)

        val_acc = 100. * correct / total
        training_metrics["val_loss"].append(val_loss / len(val_loader))
        training_metrics["val_acc"].append(val_acc)

        # Save the model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_resnet50.pth")
            print(f"Best model saved with accuracy: {best_val_acc:.2f}%")

        print(f"Epoch [{epoch+1}/{EPOCHS}], "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

        # Step scheduler at the end of each epoch
        scheduler.step()

    # Save training metrics to a file
    with open("training_metrics.json", "w") as f:
        json.dump(training_metrics, f)

    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%. Metrics saved to 'training_metrics.json'.")
    print(f"Number of corrupted or missing files skipped: {train_dataset.skipped_files + val_dataset.skipped_files}")
