import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json

# Custom Dataset for multimodal data
class BloodCancerDataset(Dataset):
    def __init__(self, df, image_root_dir, transform=None):
        """
        df: DataFrame containing at least columns ['image_path', 'ehr_json', 'label']
        image_root_dir: base path where images are stored
        transform: torchvision transforms for images
        """
        self.df = df
        self.image_root = image_root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        image = Image.open(f"{self.image_root}/{img_path}").convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load EHR JSON
        ehr = json.loads(row["ehr_json"])
        # Convert to vector — select some keys (customize as per your data)
        ehr_features = [
            ehr["ehr_age"],
            1 if ehr["ehr_gender"] == "Male" else 0,
            ehr["lab_wbc"],
            ehr["lab_hemoglobin"],
            ehr["lab_platelets"],
            ehr["lab_blasts_pct"],
            # add more features as required
        ]
        ehr_tensor = torch.tensor(ehr_features, dtype=torch.float32)

        label = torch.tensor(row["label"], dtype=torch.long)

        return {"image": image, "ehr": ehr_tensor, "label": label}


# Data loading
# Suppose you have a Parquet or CSV with these columns
df = pd.read_parquet("blood_cells_cancer_dataset.parquet")  # or read_csv
# Or build from your own data

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = BloodCancerDataset(df, image_root_dir="images", transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# Similarly make val_loader / test_loader

# Model architecture: simple multimodal model
class MultiModalModel(nn.Module):
    def __init__(self, ehr_input_dim, num_classes):
        super(MultiModalModel, self).__init__()
        # Pre-trained CNN for images
        self.cnn = models.resnet18(pretrained=True)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()  # remove final layer

        # MLP for EHR
        self.ehr_mlp = nn.Sequential(
            nn.Linear(ehr_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, ehr):
        img_feat = self.cnn(image)  # feature from CNN
        ehr_feat = self.ehr_mlp(ehr)  # feature from EHR
        combined = torch.cat([img_feat, ehr_feat], dim=1)
        out = self.classifier(combined)
        return out

# Instantiate model
ehr_input_dim = 6  # change based on how many EHR features you have
num_classes = len(df["label"].unique())
model = MultiModalModel(ehr_input_dim, num_classes)

# Training loop (simple)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        images = batch["image"].to(device)
        ehr = batch["ehr"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images, ehr)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), "multimodal_blood_cancer_model.pth")
print("Model saved.")
