import deeplake
from torchvision import transforms

# Load datasets
train_ds = deeplake.load('hub://activeloop/nabirds-dataset-train')
val_ds = deeplake.load('hub://activeloop/nabirds-dataset-val')

# Define transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply transformations to datasets
train_ds.transform = transform
val_ds.transform = transform

# Create DataLoaders using the pytorch method
train_loader = train_ds.pytorch(num_workers=4, batch_size=64, shuffle=True)
val_loader = val_ds.pytorch(num_workers=4, batch_size=64, shuffle=False)

print(len(train_loader), len(val_loader))