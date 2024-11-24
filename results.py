from tqdm import tqdm
from colorama import Fore, Style
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import timm
import os

# Configurations
test_dir = "bigger_dataset/test"
IMG_SIZE = 300
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 50  # Adjust to match your dataset
MODEL_PATHS = ["best_model_efficientnet_b3_bigger_dataset_augmented_data.pth", "fossil_classifier.pth"]

# Data Transformations
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Dataset and DataLoader
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Function to evaluate a model
def evaluate_model(model, dataloader, classes):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with tqdm(total=len(dataloader), desc=Fore.CYAN + "Evaluating" + Style.RESET_ALL, colour="blue") as pbar:
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            pbar.update(1)
    
    accuracy = 100. * correct / total
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, report, cm

# Evaluate each model and store results
results = []
with tqdm(MODEL_PATHS, desc=Fore.GREEN + "Processing Models" + Style.RESET_ALL, colour="green") as model_progress:
    for model_path in model_progress:
        try:
            if not os.path.isfile(model_path):
                print(Fore.YELLOW + f"Model file {model_path} does not exist. Skipping." + Style.RESET_ALL)
                continue

            # Reconstruct the model architecture (adjust for your use case)
            model = timm.create_model('efficientnet_b3', pretrained=False)
            model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)

            # Load the state_dict safely
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            with open(model_path, 'rb') as f:
                state_dict = torch.load(f, map_location=map_location)

            model.load_state_dict(state_dict)  # Load weights
            model = model.to(DEVICE)

            print(Fore.BLUE + f"Evaluating model: {model_path}" + Style.RESET_ALL)
            accuracy, report, cm = evaluate_model(model, test_loader, test_dataset.classes)

            results.append({
                "Model": os.path.basename(model_path),
                "Accuracy": accuracy,
                "Precision": report['weighted avg']['precision'],
                "Recall": report['weighted avg']['recall'],
                "F1-score": report['weighted avg']['f1-score']
            })

            # Save confusion matrix as a heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
            plt.title(f"Confusion Matrix for {os.path.basename(model_path)}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.savefig(f"confusion_matrix_{os.path.basename(model_path)}.png")
            plt.close()

        except Exception as e:
            print(Fore.RED + f"Error loading or evaluating model {model_path}: {e}" + Style.RESET_ALL)

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("model_comparisons.csv", index=False)
print(Fore.GREEN + "Results saved to 'model_comparisons.csv'" + Style.RESET_ALL)
