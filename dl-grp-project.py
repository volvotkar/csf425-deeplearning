import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
import requests
import tarfile
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
from datasets import load_dataset

os.makedirs('/reports', exist_ok=True)


class TransformedSubset(Dataset):
    """Dataset wrapper that applies a transform to a subset of another dataset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

class ImageNetDataHandler:
    """
    Class to handle ImageWoof dataset (a subset of ImageNet with 10 dog breeds).
    """
    def __init__(self, root='./data', input_size=224, batch_size=32, num_workers=4,
                 subset_name='imagewoof2', val_split=0.3, download=True):
        self.root = root
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subset_name = subset_name
        self.val_split = val_split

        # Create directory if it doesn't exist
        os.makedirs(root, exist_ok=True)

        # Define normalization values for ImageNet
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        # Prepare transforms
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()

        # Handle dataset - download ImageWoof if needed
        if download:
            self._download_subset()

        # Create datasets and loaders
        self.train_dataset, self.val_dataset = self._create_datasets()
        self.train_loader = self._create_train_loader()
        self.val_loader = self._create_val_loader()

    def _download_subset(self):
        """Download ImageWoof dataset"""
        # ImageWoof is a 10-class subset of ImageNet with dog breeds
        url = f"https://s3.amazonaws.com/fast-ai-imageclas/{self.subset_name}.tgz"
        target_path = os.path.join(self.root, f"{self.subset_name}.tgz")
        extract_path = os.path.join(self.root, self.subset_name)

        # Check if already downloaded and extracted
        if os.path.exists(extract_path):
            print(f"ImageWoof dataset ({self.subset_name}) already downloaded and extracted.")
            return

        # Download the file if needed
        if not os.path.exists(target_path):
            print(f"Downloading ImageWoof dataset ({self.subset_name})...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            with open(target_path, 'wb') as file, tqdm(
                desc=target_path,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)

        # Extract the downloaded file
        print(f"Extracting ImageWoof dataset ({self.subset_name})...")
        with tarfile.open(target_path) as tar:
            tar.extractall(path=self.root)

        print(f"Successfully downloaded and extracted ImageWoof dataset ({self.subset_name}).")

    def _get_train_transform(self):
        """Define training transformations including augmentation."""
        return transforms.Compose([
            transforms.RandomResizedCrop(self.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            self.normalize,
        ])

    def _get_val_transform(self):
        """Define validation transformations."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            self.normalize,
        ])

    def _create_datasets(self):
        """Create training and validation datasets."""
        # Use ImageWoof dataset
        subset_path = os.path.join(self.root, self.subset_name)

        # ImageWoof has train and val folders
        train_path = os.path.join(subset_path, 'train')
        val_path = os.path.join(subset_path, 'val')

        if os.path.exists(train_path) and os.path.exists(val_path):
            # Use existing train/val split
            train_dataset = torchvision.datasets.ImageFolder(
                train_path,
                transform=self.train_transform
            )

            val_dataset = torchvision.datasets.ImageFolder(
                val_path,
                transform=self.val_transform
            )
        else:
            # If there's no explicit split, create one
            full_dataset = torchvision.datasets.ImageFolder(
                subset_path,
                transform=None  # Will apply transforms later
            )

            # Calculate split sizes
            val_size = int(len(full_dataset) * self.val_split)
            train_size = len(full_dataset) - val_size

            # Split dataset
            train_dataset, val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            # Create datasets with appropriate transforms
            train_dataset = TransformedSubset(train_dataset, self.train_transform)
            val_dataset = TransformedSubset(val_dataset, self.val_transform)

        return train_dataset, val_dataset

    def _create_train_loader(self):
        """Create training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def _create_val_loader(self):
        """Create validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_train_loader(self):
        """Return the training data loader."""
        return self.train_loader

    def get_val_loader(self):
        """Return the validation data loader."""
        return self.val_loader

    def get_num_classes(self):
        """Return the number of classes in the dataset."""
        return len(self.train_dataset.classes if hasattr(self.train_dataset, 'classes')
                 else self.train_dataset.dataset.classes)

    def get_class_names(self):
        """Return the class names in the dataset."""
        class_names = self.train_dataset.classes if hasattr(self.train_dataset, 'classes') else self.train_dataset.dataset.classes
        # ImageWoof class names
        imagewoof_classes = [
            'Australian terrier', 'Border terrier', 'Samoyed', 'Beagle',
            'Shih-Tzu', 'English foxhound', 'Rhodesian ridgeback',
            'Dingo', 'Golden retriever', 'Old English sheepdog'
        ]
        return imagewoof_classes if len(class_names) == 10 else class_names


class MiniImageNetDataHandler:
    """
    Class to handle the timm/mini-imagenet dataset from HuggingFace.
    """
    def __init__(self, input_size=224, batch_size=32, num_workers=4):
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define normalization values for ImageNet
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        
        # Prepare transforms
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()
        
        # Load the dataset
        self._load_dataset()
        
        # Create data loaders
        self.train_loader = self._create_train_loader()
        self.val_loader = self._create_val_loader()
    
    def _get_train_transform(self):
        """Define training transformations including augmentation."""
        return transforms.Compose([
            transforms.RandomResizedCrop(self.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            self.normalize,
        ])
    
    def _get_val_transform(self):
        """Define validation transformations."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            self.normalize,
        ])
    
    def _load_dataset(self):
        """Load the mini-imagenet dataset from HuggingFace."""
        # Load the dataset
        print("Loading timm/mini-imagenet dataset from HuggingFace...")
        self.hf_dataset = load_dataset("timm/mini-imagenet")
        
        # Create PyTorch datasets for train and validation
        self.train_dataset = HFDatasetWrapper(
            self.hf_dataset["train"],
            transform=self.train_transform
        )
        
        self.val_dataset = HFDatasetWrapper(
            self.hf_dataset["validation"],
            transform=self.val_transform
        )
        
        # Also create a test dataset for final evaluation
        self.test_dataset = HFDatasetWrapper(
            self.hf_dataset["test"],
            transform=self.val_transform
        )
    
    def _create_train_loader(self):
        """Create training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def _create_val_loader(self):
        """Create validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_train_loader(self):
        """Return the training data loader."""
        return self.train_loader
    
    def get_val_loader(self):
        """Return the validation data loader."""
        return self.val_loader
    
    def get_num_classes(self):
        """Return the number of classes in the dataset."""
        return 100  # mini-imagenet has 100 classes
    
    def get_class_names(self):
        """Return the class names in the dataset."""
        return [f"Class {i}" for i in range(100)]  # Placeholder class names


# class HFDatasetWrapper(Dataset):
#     """Wrapper for HuggingFace dataset to apply transformations."""
#     def __init__(self, hf_dataset, transform=None):
#         self.hf_dataset = hf_dataset
#         self.transform = transform
    
#     def __len__(self):
#         return len(self.hf_dataset)
    
#     def __getitem__(self, idx):
#         item = self.hf_dataset[idx]
#         image = item['image']
#         label = item['label']
        
#         if self.transform:
#             image = self.transform(image)
        
#         return image, label


class HFDatasetWrapper(Dataset):
    """Wrapper for HuggingFace dataset to apply transformations."""
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']
        label = item['label']
        
        # Check if image is grayscale and convert to RGB
        if isinstance(image, Image.Image) and image.mode == 'L':
            image = image.convert('RGB')
        # For numpy arrays
        elif isinstance(image, np.ndarray) and len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class SBPool(nn.Module):
    """
    Spatially-Balanced Pooling wrapper for downsampling layers.
    Prevents CNNs from overfitting the input size during training.
    """
    def __init__(self, layer, layer_id=None, is_training=True):
        super(SBPool, self).__init__()
        self.layer = layer
        self.layer_id = layer_id
        self.is_training = is_training
        self.original_padding = None

        # Store original padding values
        if hasattr(layer, 'padding'):
            if isinstance(layer.padding, tuple) or isinstance(layer.padding, list):
                if len(layer.padding) == 2:
                    self.original_padding = (layer.padding[0], layer.padding[0],
                                            layer.padding[1], layer.padding[1])
                else:
                    self.original_padding = layer.padding
            elif isinstance(layer.padding, int):
                self.original_padding = (layer.padding, layer.padding,
                                        layer.padding, layer.padding)

        # If no padding attribute, assume zero padding
        if self.original_padding is None:
            self.original_padding = (0, 0, 0, 0)

    def forward(self, x):
        # Original layer parameters
        if isinstance(self.layer, nn.MaxPool2d) or isinstance(self.layer, nn.AvgPool2d):
            k_h = k_w = self.layer.kernel_size if isinstance(self.layer.kernel_size, int) else self.layer.kernel_size
            s_h = s_w = self.layer.stride if isinstance(self.layer.stride, int) else self.layer.stride
            d_h = d_w = self.layer.dilation if isinstance(self.layer.dilation, int) else self.layer.dilation
        elif isinstance(self.layer, nn.Conv2d):
            k_h, k_w = self.layer.kernel_size if isinstance(self.layer.kernel_size, int) else self.layer.kernel_size
            s_h, s_w = self.layer.stride if isinstance(self.layer.stride, int) else self.layer.stride
            d_h, d_w = self.layer.dilation if isinstance(self.layer.dilation, int) else self.layer.dilation
        else:
            raise ValueError(f"Unsupported layer type: {type(self.layer)}")

        # Input dimensions
        h, w = x.shape[2], x.shape[3]

        # Original padding values
        p_top, p_bottom, p_left, p_right = self.original_padding

        # Calculate unconsumed padding
        u_h = (h + p_top + p_bottom - d_h * (k_h - 1) - 1) % s_h
        u_w = (w + p_left + p_right - d_w * (k_w - 1) - 1) % s_w

        # During training, randomize the position of unconsumed padding
        if self.is_training and self.training:
            rand_h = random.randint(0, u_h) if u_h > 0 else 0
            rand_w = random.randint(0, u_w) if u_w > 0 else 0

            p_top_new = p_top - rand_h
            p_left_new = p_left - rand_w
        else:
            # During inference, use fixed grid based on layer_id to avoid receptive field distortion
            if self.layer_id is not None:
                # Assign different fixed patterns to different layers
                pattern = self.layer_id % 4
                if pattern == 0:
                    p_top_new, p_left_new = p_top, p_left
                elif pattern == 1:
                    p_top_new, p_left_new = p_top - u_h, p_left
                elif pattern == 2:
                    p_top_new, p_left_new = p_top, p_left - u_w
                else:
                    p_top_new, p_left_new = p_top - u_h, p_left - u_w
            else:
                p_top_new, p_left_new = p_top, p_left

        # Handle information erosion by cropping input if necessary
        if p_top_new < 0:
            x = x[:, :, -p_top_new:, :]
            p_top_new = 0

        if p_left_new < 0:
            x = x[:, :, :, -p_left_new:]
            p_left_new = 0

        # Set new padding for the layer if it supports padding
        if hasattr(self.layer, 'padding'):
            if isinstance(self.layer, nn.Conv2d):
                # For Conv2d, padding is a 2-tuple or int
                if isinstance(self.layer.padding, int):
                    self.layer.padding = (p_top_new, p_left_new)
                else:
                    self.layer.padding = (p_top_new, p_left_new)
            else:
                # For pooling layers, we'll need to handle padding manually
                pass

        # For layers without padding attribute, apply padding manually
        if not hasattr(self.layer, 'padding') or isinstance(self.layer, nn.MaxPool2d) or isinstance(self.layer, nn.AvgPool2d):
            x = F.pad(x, (p_left_new, p_right, p_top_new, p_bottom))

        # Forward pass through the layer
        output = self.layer(x)

        # Reset padding to original values
        if hasattr(self.layer, 'padding'):
            self.layer.padding = self.original_padding[:2] if len(self.original_padding) > 2 else self.original_padding

        return output


# class SBMobileNetV2(nn.Module):
#     """
#     MobileNetV2 with Spatially-Balanced Pooling.
#     """
#     def __init__(self, num_classes=1000, is_training=True):
#         super(SBMobileNetV2, self).__init__()
#         self.model = models.mobilenet_v2(pretrained=True)

#         # Replace the first strided convolutional layer with SBPool
#         self.model.features[0][0] = SBPool(self.model.features[0][0], layer_id=0, is_training=is_training)

#         # Replace strided convolutions in inverted residual blocks with SBPool
#         layer_id = 1
#         for idx, block in enumerate(self.model.features[1:]):
#             if isinstance(block, nn.Sequential):
#                 for i, layer in enumerate(block):
#                     if hasattr(layer, 'conv'):
#                         for j, conv in enumerate(layer.conv):
#                             if isinstance(conv, nn.Conv2d) and (conv.stride[0] > 1 or conv.stride[1] > 1):
#                                 layer.conv[j] = SBPool(conv, layer_id=layer_id, is_training=is_training)
#                                 layer_id += 1

#         # Replace classifier layer for the specified number of classes
#         if num_classes != 1000:
#             self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

#     def forward(self, x):
#         return self.model(x)


class SBMobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, is_training=True):
        super(SBMobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.is_training = is_training
        
        # Replace all strided convolutions with SBPool
        layer_id = 0
        
        # First layer - already showing up correctly
        self.model.features[0][0] = SBPool(self.model.features[0][0], 
                                          layer_id=layer_id, 
                                          is_training=is_training)
        layer_id += 1
        
        # Debug: Print when SBPool is applied
        print(f"Applied SBPool to features[0][0], layer_id={layer_id-1}")
        
        # Process all inverted residual blocks
        for block_idx, block in enumerate(self.model.features[1:]):
            # Each block usually contains a sequence of layers
            if isinstance(block, nn.Sequential):
                for layer_idx, layer in enumerate(block):
                    # Check for InvertedResidual blocks
                    if hasattr(layer, 'conv'):
                        # Specifically target the depthwise conv (usually index 1)
                        if len(layer.conv) > 1 and isinstance(layer.conv[1], nn.Conv2d):
                            if layer.conv[1].stride[0] > 1 or layer.conv[1].stride[1] > 1:
                                # Replace with SBPool
                                old_conv = layer.conv[1]
                                layer.conv[1] = SBPool(old_conv, 
                                                      layer_id=layer_id, 
                                                      is_training=is_training)
                                layer_id += 1
                                print(f"Applied SBPool to block {block_idx+1}, layer {layer_idx}, layer_id={layer_id-1}")
        
        # Replace classifier for the target number of classes
        if num_classes != 1000:
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


class Trainer:
    """
    Trainer class for model training and evaluation.
    """
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler=None, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.model.to(self.device)

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(self.train_loader, desc="Training"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Calculate statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)

        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Calculate statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        self.val_losses.append(epoch_loss)
        self.val_accs.append(epoch_acc)

        return epoch_loss, epoch_acc

    def train(self, epochs):
        """Train the model for the specified number of epochs."""
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            if self.scheduler:
                self.scheduler.step()

            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    def evaluate_size_sensitivity(self, min_size=192, max_size=299, step=1):
        """
        Evaluate the model's sensitivity to different input sizes.
        Returns a dictionary with size as key and accuracy as value.
        """
        self.model.eval()
        size_accuracy = {}
        dataset_root = self.val_loader.dataset.root if hasattr(self.val_loader.dataset, 'root') else None

        # Loop through various sizes
        for size in tqdm(range(min_size, max_size + 1, step), desc="Evaluating sizes"):
            # Create a validation transform with the current size
            val_transform = transforms.Compose([
                transforms.Resize(max(size, 256)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # Create a new dataset with the current size
            if isinstance(self.val_loader.dataset, torchvision.datasets.ImageFolder):
                temp_dataset = torchvision.datasets.ImageFolder(
                    self.val_loader.dataset.root,
                    transform=val_transform
                )
            elif hasattr(self.val_loader.dataset, 'dataset') and hasattr(self.val_loader.dataset.dataset, 'root'):
                # For subsets
                temp_dataset = torchvision.datasets.ImageFolder(
                    self.val_loader.dataset.dataset.root,
                    transform=val_transform
                )
            else:
                # Use a TransformedSubset if dataset doesn't match expected structure
                temp_dataset = TransformedSubset(self.val_loader.dataset, val_transform)

            # Create a new loader
            temp_loader = DataLoader(
                temp_dataset,
                batch_size=self.val_loader.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            # Compute accuracy
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in temp_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            acc = 100.0 * correct / total
            size_accuracy[size] = acc
            print(f'Size {size}x{size}: Accuracy = {acc:.2f}%')

        return size_accuracy

    # def evaluate_size_sensitivity(self, min_size=192, max_size=299, step=1):
    #     """
    #     Evaluate the model's sensitivity to different input sizes for HuggingFace datasets.
    #     Returns a dictionary with size as key and accuracy as value.
    #     """
    #     self.model.eval()
    #     size_accuracy = {}
        
    #     # Get the validation dataset from the loader
    #     validation_dataset = self.val_loader.dataset
        
    #     # Loop through various sizes
    #     for size in tqdm(range(min_size, max_size + 1, step), desc="Evaluating sizes"):
    #         # Create a validation transform with the current size
    #         val_transform = transforms.Compose([
    #             transforms.Resize(max(size, 256)),
    #             transforms.CenterCrop(size),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         ])
            
    #         # Create a new dataset with the current size
    #         if isinstance(validation_dataset, HFDatasetWrapper):
    #             temp_dataset = HFDatasetWrapper(
    #                 validation_dataset.hf_dataset,
    #                 transform=val_transform
    #             )
    #         else:
    #             # Fallback for other dataset types
    #             temp_dataset = validation_dataset  # Use original dataset with warning
    #             print("Warning: Using original dataset, sizes may not be correct")
                    
    #         # Create a new loader
    #         temp_loader = DataLoader(
    #             temp_dataset,
    #             batch_size=self.val_loader.batch_size,
    #             shuffle=False,
    #             num_workers=4,
    #             pin_memory=True
    #         )
            
    #         # Compute accuracy
    #         correct = 0
    #         total = 0
            
    #         with torch.no_grad():
    #             for inputs, labels in temp_loader:
    #                 inputs, labels = inputs.to(self.device), labels.to(self.device)
    #                 outputs = self.model(inputs)
    #                 _, predicted = outputs.max(1)
    #                 total += labels.size(0)
    #                 correct += predicted.eq(labels).sum().item()
            
    #         acc = 100.0 * correct / total
    #         size_accuracy[size] = acc
    #         print(f'Size {size}x{size}: Accuracy = {acc:.2f}%')
        
    #     return size_accuracy


    def plot_size_sensitivity(self, size_accuracy):
        """Plot accuracy as a function of input size."""
        sizes = list(size_accuracy.keys())
        accuracies = list(size_accuracy.values())

        plt.figure(figsize=(12, 6))
        plt.plot(sizes, accuracies, marker='o')
        plt.xlabel('Input Size')
        plt.ylabel('Accuracy (%)')
        plt.title('Model Accuracy vs Input Size')
        plt.grid(True)
        plt.savefig('/reports/size_sensitivity.png')
        plt.show()

    def save_model(self, path):
        """Save the model to the specified path."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }, path)

    def load_model(self, path):
        """Load the model from the specified path."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_accs = checkpoint['val_accs']



def model_summary(model, input_size=(3, 224, 224), batch_size=1):
    """
    Print a detailed summary of the model architecture and parameters.
    """
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["output_shape"] = list(output.size())
            
            params = 0
            for name, param in module.named_parameters():
                params += param.numel()
            
            summary[m_key]["trainable"] = any(p.requires_grad for p in module.parameters())
            summary[m_key]["nb_params"] = params
        
        if not isinstance(module, nn.Sequential) and \
           not isinstance(module, nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))
    
    # Use OrderedDict for consistent display
    from collections import OrderedDict
    summary = OrderedDict()
    hooks = []
    
    # Register hooks
    model.apply(register_hook)
    
    # Make a forward pass
    x = torch.zeros((batch_size, *input_size)).to(next(model.parameters()).device)
    model(x)
    
    # Remove these hooks
    for h in hooks:
        h.remove()
    
    # Print summary
    print(f"{'Layer (type)':<40}{'Output Shape':<25}{'Param #':<15}{'Trainable':<10}")
    print("="*90)
    
    total_params = 0
    total_trainable_params = 0
    
    for layer in summary:
        params = summary[layer]["nb_params"]
        trainable = summary[layer]["trainable"]
        
        total_params += params
        if trainable:
            total_trainable_params += params
        
        output_shape = str(summary[layer]["output_shape"])
        
        print(f"{layer:<40}{output_shape:<25}{params:<15}{str(trainable):<10}")
    
    print("="*90)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {total_trainable_params:,}")
    print(f"Non-trainable params: {total_params - total_trainable_params:,}")
    
    return summary



def visualize_kernels(model, layer_names=None, max_filters=16, figsize=(15, 15)):
    """
    Visualize convolutional kernels in the specified layers.
    This function extracts and visualizes the learned filters.
    """
    if layer_names is None:
        # Find all convolutional layers
        layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_names.append(name)
    
    for layer_name in layer_names:
        # Navigate through model to get the layer
        layer = model
        for name in layer_name.split('.'):
            if name.isdigit():
                layer = layer[int(name)]
            else:
                layer = getattr(layer, name)
        
        # Get the SBPool's internal layer if needed
        if isinstance(layer, SBPool):
            layer = layer.layer
        
        # Skip if not a convolutional layer
        if not isinstance(layer, nn.Conv2d):
            print(f"Layer {layer_name} is not a convolutional layer. Skipping.")
            continue
        
        # Get weights
        weights = layer.weight.data.clone().cpu()
        
        # Calculate mean filter across all output channels
        mean_filter = weights.mean(dim=0)
        
        # Set up the plot
        filters_per_row = int(np.sqrt(max_filters)) + 1
        fig, axes = plt.subplots(filters_per_row, filters_per_row, figsize=figsize)
        fig.suptitle(f'Filters from {layer_name}', fontsize=16)
        
        # Plot the mean filter
        ax = axes[0, 0]
        if mean_filter.shape[0] == 3:  # RGB filter
            # Normalize each channel
            for c in range(3):
                mean_filter[c] = (mean_filter[c] - mean_filter[c].min()) / (mean_filter[c].max() - mean_filter[c].min() + 1e-8)
            
            # Convert to numpy for plotting
            mean_filter_np = mean_filter.permute(1, 2, 0).numpy()
            ax.imshow(mean_filter_np)
        else:
            # For non-RGB, just take the mean across input channels
            mean_filter_single = mean_filter.mean(dim=0)
            ax.imshow(mean_filter_single, cmap='viridis')
        
        ax.set_title('Mean Filter')
        ax.axis('off')
        
        # Plot individual filters (up to max_filters)
        num_filters = min(weights.shape[0], max_filters-1)
        for i in range(num_filters):
            row, col = (i+1) // filters_per_row, (i+1) % filters_per_row
            ax = axes[row, col]
            
            # Get the first filter for this output channel
            filter_weights = weights[i, 0].mean(dim=0) if weights.shape[1] > 1 else weights[i, 0]
            
            # Normalize for better visualization
            filter_weights = (filter_weights - filter_weights.min()) / (filter_weights.max() - filter_weights.min() + 1e-8)
            
            ax.imshow(filter_weights, cmap='viridis')
            ax.set_title(f'Filter {i+1}')
            ax.axis('off')
        
        # Turn off any unused subplots
        for i in range(num_filters+1, filters_per_row * filters_per_row):
            row, col = i // filters_per_row, i % filters_per_row
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'/reports/kernels_{layer_name.replace(".", "_")}.png')
        plt.show()


def visualize_receptive_field(model, input_size=(224, 224), device='cuda'):
    """
    Visualize the effective receptive field of the model using gradient backpropagation.
    This helps understand how different input sizes affect the model's receptive field.
    """
    model = model.to(device)
    model.eval()
    
    # Create input tensor
    x = torch.zeros((1, 3, *input_size), requires_grad=True, device=device)
    
    # Forward pass
    y = model(x)
    
    # If the output is a classification logit, convert to features
    if len(y.shape) == 2:  # [batch_size, num_classes]
        # Use the last feature layer instead
        # Need to register a hook to get features
        features = None
        
        def hook_fn(module, input, output):
            nonlocal features
            features = output
        
        # Find the last convolutional or feature layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d) or 'features' in name:
                hook = module.register_forward_hook(hook_fn)
                model(x)  # Forward pass to trigger hook
                hook.remove()
                break
        
        if features is None:
            print("Could not find a suitable feature layer for receptive field visualization")
            return None
        
        # Use the feature map for gradient visualization
        y = features
    
    # Create target at the center of the output feature map
    center_h, center_w = y.shape[2] // 2, y.shape[3] // 2
    target = torch.zeros_like(y)
    target[0, :, center_h, center_w] = 1.0
    
    # Backward pass to compute gradients
    y.backward(target)
    
    # Extract gradient at input
    grad = x.grad.abs().sum(dim=1).squeeze().cpu().numpy()
    
    # Normalize for visualization
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    
    # Plot the receptive field
    plt.figure(figsize=(10, 10))
    plt.imshow(grad, cmap='jet')
    plt.colorbar(label='Gradient Magnitude')
    plt.title(f'Receptive Field Visualization (Input Size: {input_size[0]}x{input_size[1]})')
    plt.axis('off')
    plt.savefig(f'/reports/receptive_field_{input_size[0]}x{input_size[1]}.png')
    plt.show()
    
    return grad




def analyze_receptive_field_sizes(model, input_sizes, device='cuda'):
    """
    Analyze how the effective receptive field changes with different input sizes.
    Creates a plot showing the relationship between input size and receptive field size.
    """
    model = model.to(device)
    model.eval()
    
    rf_sizes = []
    
    for size in input_sizes:
        print(f"Analyzing receptive field for input size {size}x{size}...")
        rf = visualize_receptive_field(model, input_size=(size, size), device=device)
        
        # Measure the effective receptive field size (area with significant gradient)
        threshold = 0.2  # Threshold for considering part of receptive field
        binary_rf = (rf > threshold).astype(np.float32)
        
        # Find the bounds of the receptive field
        rows, cols = np.where(binary_rf > 0)
        if len(rows) > 0 and len(cols) > 0:
            rf_height = rows.max() - rows.min()
            rf_width = cols.max() - cols.min()
            rf_size = (rf_height + rf_width) / 2  # Average of height and width
        else:
            rf_size = 0
        
        rf_sizes.append(rf_size)
    
    # Plot receptive field size vs input size
    plt.figure(figsize=(10, 6))
    plt.plot(input_sizes, rf_sizes, marker='o', linestyle='-')
    plt.xlabel('Input Size')
    plt.ylabel('Effective Receptive Field Size')
    plt.title('Receptive Field Size vs Input Size')
    plt.grid(True)
    plt.savefig('/reports/receptive_field_sizes.png')
    plt.show()
    
    return rf_sizes



def visualize_feature_map_sizes(model, input_sizes, device='cuda'):
    """
    Visualize how feature map sizes change across the network for different input sizes.
    This helps understand the downsampling pattern through the network.
    """
    model = model.to(device)
    model.eval()
    
    # Store feature map sizes
    all_feature_maps = {}
    
    for input_size in input_sizes:
        feature_maps = []
        hooks = []
        
        # Register hooks for all convolutional and pooling layers
        def hook_fn(module, input, output):
            feature_maps.append(output.shape[2:])  # Store H,W dimensions
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        with torch.no_grad():
            x = torch.zeros((1, 3, input_size, input_size), device=device)
            model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        all_feature_maps[input_size] = feature_maps
    
    # Plot feature map sizes for different input sizes
    plt.figure(figsize=(12, 8))
    
    for idx, input_size in enumerate(input_sizes):
        heights = [fm[0] for fm in all_feature_maps[input_size]]
        plt.plot(heights, label=f'Input {input_size}x{input_size}', marker='o')
    
    plt.xlabel('Layer Index')
    plt.ylabel('Feature Map Height')
    plt.title('Feature Map Sizes Throughout Network')
    plt.legend()
    plt.grid(True)
    plt.savefig('/reports/feature_map_sizes.png')
    plt.show()
    
    return all_feature_maps



# # Set random seeds for reproducibility
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

# # Configuration
# input_size = 224
# batch_size = 32
# learning_rate = 0.001
# weight_decay = 1e-4
# epochs = 5
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Create dataset handler - will load mini-imagenet from HuggingFace
# print("Preparing mini-imagenet dataset...")
# dataset_handler = MiniImageNetDataHandler(
#     input_size=input_size,
#     batch_size=batch_size
# )

# num_classes = dataset_handler.get_num_classes()
# print(f"Number of classes: {num_classes}")

# # Create both models - standard and SBPool
# print("Creating standard MobileNetV2...")
# standard_model = models.mobilenet_v2(pretrained=True)
# # Adjust the classifier for the number of classes (100 for mini-imagenet)
# standard_model.classifier[1] = nn.Linear(standard_model.classifier[1].in_features, num_classes)

# print("Creating SBPool MobileNetV2...")
# sbpool_model = SBMobileNetV2(num_classes=num_classes, is_training=True)

# # # Print model summaries
# # print("\n================ Standard MobileNetV2 Summary ================")
# # model_summary(standard_model, input_size=(3, input_size, input_size))

# # print("\n================ SBPool MobileNetV2 Summary ================")
# # model_summary(sbpool_model, input_size=(3, input_size, input_size))

# # # Visualize kernels (untrained)
# # print("\nVisualizing initial kernels for Standard MobileNetV2...")
# # visualize_kernels(standard_model, ['features.0.0', 'features.1.conv.0', 'features.18.0'])

# # print("\nVisualizing initial kernels for SBPool MobileNetV2...")
# # visualize_kernels(sbpool_model, ['model.features.0.0.layer', 'model.features.1.conv.0', 'model.features.18.0'])

# # # Visualize feature map sizes for different input sizes
# # input_size_range = [192, 208, 224, 240, 256]

# # print("\nVisualizing feature map sizes for Standard MobileNetV2...")
# # std_feature_maps = visualize_feature_map_sizes(standard_model, input_size_range, device)

# # print("\nVisualizing feature map sizes for SBPool MobileNetV2...")
# # sbpool_feature_maps = visualize_feature_map_sizes(sbpool_model, input_size_range, device)

# # # Visualize receptive fields (untrained)
# # print("\nVisualizing initial receptive fields...")
# # for size in [192, 224, 256]:
# #     print(f"\nStandard MobileNetV2 - Input Size {size}x{size}")
# #     visualize_receptive_field(standard_model, input_size=(size, size), device=device)
    
# #     print(f"\nSBPool MobileNetV2 - Input Size {size}x{size}")
# #     visualize_receptive_field(sbpool_model, input_size=(size, size), device=device)

# # Create loss function and optimizers
# criterion = nn.CrossEntropyLoss()

# standard_optimizer = torch.optim.Adam(
#     standard_model.parameters(),
#     lr=learning_rate,
#     weight_decay=weight_decay
# )

# sbpool_optimizer = torch.optim.Adam(
#     sbpool_model.parameters(),
#     lr=learning_rate,
#     weight_decay=weight_decay
# )

# # Create schedulers
# standard_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     standard_optimizer, 
#     T_max=epochs
# )

# sbpool_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     sbpool_optimizer, 
#     T_max=epochs
# )

# # Create trainers
# standard_trainer = Trainer(
#     model=standard_model,
#     train_loader=dataset_handler.get_train_loader(),
#     val_loader=dataset_handler.get_val_loader(),
#     criterion=criterion,
#     optimizer=standard_optimizer,
#     scheduler=standard_scheduler,
#     device=device
# )

# sbpool_trainer = Trainer(
#     model=sbpool_model,
#     train_loader=dataset_handler.get_train_loader(),
#     val_loader=dataset_handler.get_val_loader(),
#     criterion=criterion,
#     optimizer=sbpool_optimizer,
#     scheduler=sbpool_scheduler,
#     device=device
# )

# # Train the models
# print("Training standard MobileNetV2 on mini-imagenet...")
# standard_trainer.train(epochs)
# standard_trainer.save_model('standard_mobilenet_v2_mini_imagenet.pth')

# print("Training SBPool MobileNetV2 on mini-imagenet...")
# sbpool_trainer.train(epochs)
# sbpool_trainer.save_model('sbpool_mobilenet_v2_mini_imagenet.pth')

# # # Visualize kernels after training
# # print("\nVisualizing trained kernels for Standard MobileNetV2...")
# # visualize_kernels(standard_model, ['features.0.0', 'features.1.conv.0', 'features.18.0'])

# # print("\nVisualizing trained kernels for SBPool MobileNetV2...")
# # visualize_kernels(sbpool_model, ['model.features.0.0.layer', 'model.features.1.conv.0', 'model.features.18.0'])

# # Analyze receptive field sizes after training
# print("\nAnalyzing receptive field sizes for trained models...")
# input_size_analysis = range(188, 257, 9)

# print("\nStandard MobileNetV2 receptive field analysis:")
# std_rf_sizes = analyze_receptive_field_sizes(standard_model, input_size_analysis, device)

# print("\nSBPool MobileNetV2 receptive field analysis:")
# sbpool_rf_sizes = analyze_receptive_field_sizes(sbpool_model, input_size_analysis, device)

# # Compare receptive field sizes
# plt.figure(figsize=(10, 6))
# plt.plot(input_size_analysis, std_rf_sizes, marker='o', label='Standard CNN')
# plt.plot(input_size_analysis, sbpool_rf_sizes, marker='x', label='SBPool CNN')
# plt.xlabel('Input Size')
# plt.ylabel('Effective Receptive Field Size')
# plt.title('Receptive Field Size Comparison')
# plt.legend()
# plt.grid(True)
# plt.savefig('/reports/receptive_field_size_comparison.png')
# plt.show()

# # Evaluate size sensitivity
# print("Evaluating standard model size sensitivity...")
# standard_size_accuracy = standard_trainer.evaluate_size_sensitivity(
#     min_size=192, max_size=224, step=1
# )
# standard_trainer.plot_size_sensitivity(standard_size_accuracy)

# print("Evaluating SBPool model size sensitivity...")
# sbpool_size_accuracy = sbpool_trainer.evaluate_size_sensitivity(
#     min_size=192, max_size=224, step=1
# )
# sbpool_trainer.plot_size_sensitivity(sbpool_size_accuracy)

# # Plot comparison of accuracy vs input size
# sizes = list(standard_size_accuracy.keys())
# std_accuracies = list(standard_size_accuracy.values())
# sb_accuracies = list(sbpool_size_accuracy.values())

# plt.figure(figsize=(12, 6))
# plt.plot(sizes, std_accuracies, marker='o', label='Standard CNN')
# plt.plot(sizes, sb_accuracies, marker='x', label='SBPool CNN')
# plt.xlabel('Input Size')
# plt.ylabel('Accuracy (%)')
# plt.title('Model Accuracy vs Input Size Comparison (Mini-ImageNet)')
# plt.legend()
# plt.grid(True)
# plt.savefig('/reports/size_sensitivity_comparison_mini_imagenet.png')
# plt.show()

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Configuration
input_size = 224
batch_size = 32
learning_rate = 0.001
weight_decay = 1e-4
epochs = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create dataset handler - will download ImageWoof
print("Preparing ImageWoof dataset...")
dataset_handler = ImageNetDataHandler(
    root='./data',
    input_size=input_size,
    batch_size=batch_size,
    subset_name='imagewoof2',
    download=True
)

num_classes = dataset_handler.get_num_classes()
print(f"Number of classes: {num_classes}")
print(f"Class names: {dataset_handler.get_class_names()}")

# Create both models - standard and SBPool
print("Creating standard MobileNetV2...")
standard_model = models.mobilenet_v2(pretrained=True)
# Adjust the classifier for the number of classes
standard_model.classifier[1] = nn.Linear(standard_model.classifier[1].in_features, num_classes)

print("Creating SBPool MobileNetV2...")
sbpool_model = SBMobileNetV2(num_classes=num_classes, is_training=True)

# Print model summaries
print("\n================ Standard MobileNetV2 Summary ================")
model_summary(standard_model, input_size=(3, input_size, input_size))

print("\n================ SBPool MobileNetV2 Summary ================")
model_summary(sbpool_model, input_size=(3, input_size, input_size))

# # Visualize kernels (untrained)
# print("\nVisualizing initial kernels for Standard MobileNetV2...")
# visualize_kernels(standard_model, ['features.0.0', 'features.1.conv.0', 'features.18.0'])

# print("\nVisualizing initial kernels for SBPool MobileNetV2...")
# visualize_kernels(sbpool_model, ['model.features.0.0.layer', 'model.features.1.conv.0', 'model.features.18.0'])

# # Visualize feature map sizes for different input sizes
# input_size_range = [192, 208, 224, 240, 256]

# print("\nVisualizing feature map sizes for Standard MobileNetV2...")
# std_feature_maps = visualize_feature_map_sizes(standard_model, input_size_range, device)

# print("\nVisualizing feature map sizes for SBPool MobileNetV2...")
# sbpool_feature_maps = visualize_feature_map_sizes(sbpool_model, input_size_range, device)

# # Visualize receptive fields (untrained)
# print("\nVisualizing initial receptive fields...")
# for size in [192, 224, 256]:
#     print(f"\nStandard MobileNetV2 - Input Size {size}x{size}")
#     visualize_receptive_field(standard_model, input_size=(size, size), device=device)
    
#     print(f"\nSBPool MobileNetV2 - Input Size {size}x{size}")
#     visualize_receptive_field(sbpool_model, input_size=(size, size), device=device)

# Create loss function and optimizers
criterion = nn.CrossEntropyLoss()

standard_optimizer = torch.optim.Adam(
    standard_model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

sbpool_optimizer = torch.optim.Adam(
    sbpool_model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

# Create schedulers
standard_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    standard_optimizer, 
    T_max=epochs
)

sbpool_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    sbpool_optimizer, 
    T_max=epochs
)

# Create trainers
standard_trainer = Trainer(
    model=standard_model,
    train_loader=dataset_handler.get_train_loader(),
    val_loader=dataset_handler.get_val_loader(),
    criterion=criterion,
    optimizer=standard_optimizer,
    scheduler=standard_scheduler,
    device=device
)

sbpool_trainer = Trainer(
    model=sbpool_model,
    train_loader=dataset_handler.get_train_loader(),
    val_loader=dataset_handler.get_val_loader(),
    criterion=criterion,
    optimizer=sbpool_optimizer,
    scheduler=sbpool_scheduler,
    device=device
)

# Train the models
print("Training standard MobileNetV2 on ImageWoof...")
standard_trainer.train(epochs)
standard_trainer.save_model('standard_mobilenet_v2_imagewoof.pth')

print("Training SBPool MobileNetV2 on ImageWoof...")
sbpool_trainer.train(epochs)
sbpool_trainer.save_model('sbpool_mobilenet_v2_imagewoof.pth')

# # Visualize kernels after training
# print("\nVisualizing trained kernels for Standard MobileNetV2...")
# visualize_kernels(standard_model, ['features.0.0', 'features.1.conv.0', 'features.18.0'])

# print("\nVisualizing trained kernels for SBPool MobileNetV2...")
# visualize_kernels(sbpool_model, ['model.features.0.0.layer', 'model.features.1.conv.0', 'model.features.18.0'])

# Analyze receptive field sizes after training
print("\nAnalyzing receptive field sizes for trained models...")
input_size_analysis = range(188, 257, 9)  # [192, 200, 208, ..., 256]

print("\nStandard MobileNetV2 receptive field analysis:")
std_rf_sizes = analyze_receptive_field_sizes(standard_model, input_size_analysis, device)

print("\nSBPool MobileNetV2 receptive field analysis:")
sbpool_rf_sizes = analyze_receptive_field_sizes(sbpool_model, input_size_analysis, device)

# Compare receptive field sizes
plt.figure(figsize=(10, 6))
plt.plot(input_size_analysis, std_rf_sizes, marker='o', label='Standard CNN')
plt.plot(input_size_analysis, sbpool_rf_sizes, marker='x', label='SBPool CNN')
plt.xlabel('Input Size')
plt.ylabel('Effective Receptive Field Size')
plt.title('Receptive Field Size Comparison')
plt.legend()
plt.grid(True)
plt.savefig('/reports/receptive_field_size_comparison.png')
plt.show()

# Evaluate size sensitivity
print("Evaluating standard model size sensitivity...")
standard_size_accuracy = standard_trainer.evaluate_size_sensitivity(
    min_size=192, max_size=299, step=2
)
standard_trainer.plot_size_sensitivity(standard_size_accuracy)

print("Evaluating SBPool model size sensitivity...")
sbpool_size_accuracy = sbpool_trainer.evaluate_size_sensitivity(
    min_size=192, max_size=299, step=2
)
sbpool_trainer.plot_size_sensitivity(sbpool_size_accuracy)

# Plot comparison of accuracy vs input size
sizes = list(standard_size_accuracy.keys())
std_accuracies = list(standard_size_accuracy.values())
sb_accuracies = list(sbpool_size_accuracy.values())

plt.figure(figsize=(12, 6))
plt.plot(sizes, std_accuracies, marker='o', label='Standard CNN')
plt.plot(sizes, sb_accuracies, marker='x', label='SBPool CNN')
plt.xlabel('Input Size')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy vs Input Size Comparison (ImageWoof)')
plt.legend()
plt.grid(True)
plt.savefig('/reports/size_sensitivity_comparison_imagewoof.png')
plt.show()
