import torch
import timm
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from colorama import Fore, Style
import gc
import os

# Konfigurácie
TEST_DIR = "bigger_dataset/test"
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 50

# Zoznam modelov na evaluáciu
MODELS_TO_EVALUATE = [
    {"path": "fossil_classifier.pth", "type": "resnet18", "img_size": 224},
    {"path": "best_model_efficientnet_b3_bigger_dataset_augmented_data.pth", "type": "efficientnet_b3", "img_size": 300},
    {"path": "best_model_resnet50_bigger_dataset_augmented_data.pth", "type": "resnet50", "img_size": 224}
]


# Funkcia na rekonštrukciu a načítanie modelov
def load_model_from_state_dict(model_path, model_type, num_classes, device):
    """
    Load model from state_dict and reconstruct its architecture.

    Args:
        model_path (str): Path to the saved state_dict.
        model_type (str): Model type ('resnet18', 'resnet50', 'efficientnet_b3').
        num_classes (int): Number of output classes.
        device (torch.device): Device (CPU/GPU).

    Returns:
        torch.nn.Module: Loaded model ready for evaluation.
    """
    if model_type == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "efficientnet_b3":
        model = timm.create_model('efficientnet_b3', pretrained=False)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load the state_dict
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

# Evaluácia a zápis do CSV
results = []
with tqdm(MODELS_TO_EVALUATE, desc=Fore.GREEN + "Processing Models" + Style.RESET_ALL, colour="green") as model_progress:
    for model_info in model_progress:
        try:
            print(Fore.BLUE + f"Loading model: {model_info['path']} ({model_info['type']})" + Style.RESET_ALL)
            model = load_model_from_state_dict(
                model_path=model_info["path"],
                model_type=model_info["type"],
                num_classes=NUM_CLASSES,
                device=DEVICE
            )
            print(Fore.BLUE + f"Model {model_info['type']} successfully loaded!" + Style.RESET_ALL)

            # Transformácie pre testovacie dáta
            test_transforms = transforms.Compose([
                transforms.Resize((model_info['img_size'], model_info['img_size'])),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            # Dataset a DataLoader
            test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            accuracy, report, cm = evaluate_model(model, test_loader, test_dataset.classes)

            results.append({
                "Model": model_info["type"],
                "Accuracy": accuracy,
                "Precision": report['weighted avg']['precision'],
                "Recall": report['weighted avg']['recall'],
                "F1-score": report['weighted avg']['f1-score']
            })

            # Uloženie konfúznej matice
            output_folder = "confusion_matrices"
            os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
            plt.title(f"Confusion Matrix for {model_info['type']}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.savefig(os.path.join(output_folder, f"confusion_matrix_{model_info['type']}.png"))
            plt.close()

        except Exception as e:
            print(Fore.RED + f"Failed to load or evaluate model {model_info['path']}: {e}" + Style.RESET_ALL)

        finally:
            # Uvoľnenie pamäte
            del model
            torch.cuda.empty_cache()
            gc.collect()

# Export výsledkov do CSV
results_df = pd.DataFrame(results)
results_df.to_csv("model_comparisons.csv", index=False)
print(Fore.GREEN + "Results saved to 'model_comparisons.csv'" + Style.RESET_ALL)
