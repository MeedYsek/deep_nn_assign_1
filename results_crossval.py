import torch
import timm
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from colorama import Fore, Style
import gc
import os
import numpy as np

# Konfigurácie
TEST_DIR = "bigger_dataset/test"
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 50
K_FOLDS = 5  # Number of folds for cross-validation

# Zoznam modelov na evaluáciu
MODELS_TO_EVALUATE = [
    {"path": "fossil_classifier.pth", "type": "resnet18", "img_size": 224},
    {"path": "best_model_efficientnet_b3_bigger_dataset_augmented_data.pth", "type": "efficientnet_b3", "img_size": 300},
    {"path": "best_model_resnet50_bigger_dataset_augmented_data.pth", "type": "resnet50", "img_size": 224}
]

# Funkcia na rekonštrukciu a načítanie modelov
def load_model_from_state_dict(model_path, model_type, num_classes, device):
    """
    Načíta model zo state_dict a rekonštruuje jeho architektúru.

    Args:
        model_path (str): Cesta k uloženému state_dict.
        model_type (str): Typ modelu ('resnet18' alebo 'efficientnet_b3').
        num_classes (int): Počet výstupných tried.
        device (torch.device): Používané zariadenie (CPU/GPU).

    Returns:
        torch.nn.Module: Načítaný model pripravený na evaluáciu.
    """
    if model_type == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "efficientnet_b3":
        model = timm.create_model('efficientnet_b3', pretrained=False)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    elif model_type == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Načítanie váh modelu
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

# Funkcia na evaluáciu modelu
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

# Krížová validácia
results = []
for model_info in MODELS_TO_EVALUATE:
    # Transformácie pre testovacie dáta
    test_transforms = transforms.Compose([
        transforms.Resize((model_info['img_size'], model_info['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Dataset a DataLoader
    dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        print(f"Training fold {fold+1}/{K_FOLDS} for model {model_info['type']}")

        # Subset pre tréning a validáciu
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        # Načítanie modelu
        model = load_model_from_state_dict(
            model_path=model_info["path"],
            model_type=model_info["type"],
            num_classes=NUM_CLASSES,
            device=DEVICE
        )

        # Evaluácia modelu na validačnej množine
        accuracy, report, cm = evaluate_model(model, val_loader, dataset.classes)

        fold_results.append({
            "fold": fold+1,
            "accuracy": accuracy,
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall'],
            "f1_score": report['weighted avg']['f1-score']
        })

        # Uloženie konfúznej matice pre každý fold
        output_folder = "confusion_matrices"
        os.makedirs(output_folder, exist_ok=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.classes, yticklabels=dataset.classes)
        plt.title(f"Confusion Matrix for {model_info['type']} - Fold {fold+1}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(output_folder, f"confusion_matrix_{model_info['type']}_fold_{fold+1}.png"))
        plt.close()

    # Priemerné výsledky pre tento model
    avg_results = {
        "model": model_info["type"],
        "average_accuracy": np.mean([r['accuracy'] for r in fold_results]),
        "average_precision": np.mean([r['precision'] for r in fold_results]),
        "average_recall": np.mean([r['recall'] for r in fold_results]),
        "average_f1_score": np.mean([r['f1_score'] for r in fold_results])
    }

    results.append(avg_results)

# Export výsledkov do CSV
results_df = pd.DataFrame(results)
results_df.to_csv("model_comparisons_cross_validation.csv", index=False)
print(Fore.GREEN + "Results saved to 'model_comparisons_cross_validation.csv'" + Style.RESET_ALL)
