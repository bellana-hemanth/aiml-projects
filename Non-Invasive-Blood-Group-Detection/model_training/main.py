import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Image transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
data_dir = r'C:\Users\heman\OneDrive\dataset_blood_group'  # Change if uploading to GitHub
image_dataset = datasets.ImageFolder(data_dir, transform=data_transform)

# Split dataset
train_size = int(0.8 * len(image_dataset))
val_size = len(image_dataset) - train_size
train_dataset, val_dataset = random_split(image_dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
dataloaders = {'train': train_loader, 'val': val_loader}

# Define CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 8)  # 8 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Metrics history
train_acc_history, val_acc_history = [], []
train_loss_history, val_loss_history = [], []
train_precision_history, train_recall_history, train_f1_history = [], [], []
val_precision_history, val_recall_history, val_f1_history = [], [], []

# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    
    for phase in ['train', 'val']:
        model.train() if phase == 'train' else model.eval()
        print(f"{'Training' if phase == 'train' else 'Validating'}...")

        running_loss, running_corrects = 0.0, 0
        all_preds, all_labels = [], []

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

        if phase == 'train':
            train_acc_history.append(epoch_acc.item())
            train_loss_history.append(epoch_loss)
            train_precision_history.append(precision)
            train_recall_history.append(recall)
            train_f1_history.append(f1)
        else:
            val_acc_history.append(epoch_acc.item())
            val_loss_history.append(epoch_loss)
            val_precision_history.append(precision)
            val_recall_history.append(recall)
            val_f1_history.append(f1)

        print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'{phase.capitalize()} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}')

# Save model
torch.save(model.state_dict(), 'fingerprint_blood_group_model.pkl')

# Plot metrics
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(epochs, train_acc_history, label='Train Acc', color="blue")
plt.plot(epochs, val_acc_history, label='Val Acc', color="orange")
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, train_loss_history, label='Train Loss', color='blue')
plt.plot(epochs, val_loss_history, label='Val Loss', color='orange')
plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

plt.subplot(2, 2, 3)
plt.plot(epochs, train_precision_history, label='Train Precision', color='blue')
plt.plot(epochs, val_precision_history, label='Val Precision', color='orange')
plt.title('Precision'); plt.xlabel('Epoch'); plt.ylabel('Precision'); plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs, train_recall_history, label='Train Recall', color='blue')
plt.plot(epochs, val_recall_history, label='Val Recall', color='orange')
plt.title('Recall'); plt.xlabel('Epoch'); plt.ylabel('Recall'); plt.legend()

plt.tight_layout()
plt.show()

# Predict function
def predict_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    image = data_transform(image)
    image = image.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return preds.item()

# Upload GUI
def upload_and_predict():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", ".bmp;.jpg;.jpeg;.png")])
    if not file_path:
        messagebox.showinfo("Info", "No file selected")
        return

    predicted_class = predict_image(file_path, model)
    blood_groups = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
    result = f'The predicted blood group is: {blood_groups[predicted_class]}'
    messagebox.showinfo("Prediction Result", result)

# Call GUI function
upload_and_predict()
