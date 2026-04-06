print("\n------------------- IMPORTING LIBRARIES -------------------")

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from torchvision import transforms, datasets

from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.auto import tqdm
from timeit import default_timer as timer

from helper_functions import *

# ------------------- DEVICE -------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

BATCH_SIZE = 16

# ------------------- DATA LOADING -------------------
data_dir = "../Data/1/PCB_DATASET/images"

base_transform = transforms.Resize((224, 224))

dataset = datasets.ImageFolder(
    root=data_dir,
    transform=base_transform
)

print(f"Detected classes: {dataset.classes}")
labels = dataset.targets

# ------------------- TRAIN-TEST SPLIT -------------------
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(range(len(dataset)), labels))

train_subset = Subset(dataset, train_idx)
test_subset = Subset(dataset, test_idx)

# ------------------- CUSTOM DATASET -------------------
class CustomSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# ------------------- TRANSFORMS -------------------
print("\n------------------- PREPROCESSING -------------------")

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.GaussianBlur(3),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = CustomSubset(train_subset, train_transforms)
test_dataset = CustomSubset(test_subset, test_transforms)

# ------------------- DATALOADERS -------------------
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

class_names = dataset.classes

# ------------------- MODEL (Extended TinyVGG) -------------------
class ExtendedTinyVGGCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x

# ------------------- MODEL INIT -------------------
model = ExtendedTinyVGGCNN(
    input_channels=3,
    num_classes=len(class_names)
).to(device)

# ------------------- LOSS & OPTIMIZER -------------------
loss_fn = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.001)

# ------------------- TRAINING -------------------
print("\n------------------- TRAINING STARTED -------------------")

epochs = 20
start_time = timer()

for epoch in tqdm(range(epochs)):
    print(f"\nEpoch {epoch+1}/{epochs}")

    train_acc, train_loss = train_step(
        dataloader=train_dataloader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn
    )

    test_acc, test_loss = test_step(
        dataloader=test_dataloader,
        model=model,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn
    )

    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}")
    print(f"Test  Loss: {test_loss:.3f} | Test  Acc: {test_acc:.3f}")

end_time = timer()

print_train_time(start=start_time, end=end_time, device=device)