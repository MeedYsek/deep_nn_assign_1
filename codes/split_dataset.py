import os
import shutil
from sklearn.model_selection import train_test_split
import random

# Paths
dataset_dir = r"X:\Downloads\reduced-FID"
output_dir = "bigger_dataset"
reduction_ratio = 1  # keep 80% of the dataset

print('Started creating split folders...')
# Create directories for splits
splits = ['train', 'val', 'test']
for split in splits:
    for class_name in os.listdir(dataset_dir):
        os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

print('Folders created!')

print('Started splitting images into folders...')
# Randomly reduce dataset and split data for each class
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    images = os.listdir(class_path)

    # Random reduction: Keep only a subset of images
    reduced_images = random.sample(images, int(len(images) * reduction_ratio))

    # Print reduction details
    print(f"{class_name}: {len(images)} -> {len(reduced_images)}")

    # Split the reduced dataset into train, validation, and test
    train_images, temp_images = train_test_split(reduced_images, test_size=0.20, random_state=42)  # 20% for val + test
    val_images, test_images = train_test_split(temp_images, test_size=0.25, random_state=42)  # 25% of 20% = 5%

    # Copy images to the respective folders
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(output_dir, 'train', class_name))
    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(output_dir, 'val', class_name))
    for img in test_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(output_dir, 'test', class_name))

print('Done!')
