import os
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

# Step 1: Load Dataset
data = pd.read_csv('./data/train_classes.csv')

# Step 2: Split Tags into Individual Labels
data['tags'] = data['tags'].str.split()
all_labels = set(tag for tags in data['tags'] for tag in tags)


# Step 3: Calculate Imbalance Measures
def calculate_ir(lbl_counts):
    max_count = max(lbl_counts.values())
    ir = {label: max_count / count for label, count in lbl_counts.items()}
    return ir


label_counts = {label: sum(label in tags for tags in data['tags']) for label in all_labels}
ir_lbl = calculate_ir(label_counts)
mean_ir = np.mean(list(ir_lbl.values()))

# Step 4: Identify Minority Labels
minority_labels = [label for label, ir in ir_lbl.items() if ir > mean_ir]

# Filter Minority Instances
minority_instances = data[data['tags'].apply(lambda tags: any(label in tags for label in minority_labels))]

# # Step 5: Load Images and Extract Features
# image_folder = './data/train-jpg/'
#
# # Define transformation for image preprocessing
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize images
#     transforms.ToTensor(),  # Convert to tensor
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
# ])
#
# # Load pretrained model for feature extraction
# pretrained_model = models.resnet50(pretrained=True)
# pretrained_model.eval()  # Set to evaluation mode
# feature_extractor = torch.nn.Sequential(*list(pretrained_model.children())[:-1])  # Remove last layer
#
#
# # Function to extract features
# def extract_features(image_path):
#     image = Image.open(image_path).convert('RGB')
#     image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
#     # with torch.no_grad(): # If we want to use a model for feature extraction
#     #     features = feature_extractor(image_tensor).flatten().numpy()  # Extract and flatten features
#     features = image_tensor.flatten().numpy()
#
#     return features
#
#
# # Extract features for all images
# image_features = []
# n = 0
# for img_name in data['image_name']:
#     n += 1
#     img_path = image_folder + img_name + ".jpg"
#     if os.path.exists(img_path):
#         features = extract_features(img_path)
#         image_features.append(features)
#     else:
#         print(f"Warning: Image {img_name} not found in {image_folder}")
#     if n % 100 == 0:
#         print(str(n) + "/40478")

# Load the data
loaded_data = np.load("./data/raw_pixels.npz")
data["raw_pixels"] = [None] * len(data)
# Iterate over the loaded data and assign tensors to the DataFrame
for idx, key in enumerate(loaded_data):
    print(f"Processing {key}")  # Optional: Print the key (e.g., image_0, image_1, ...)
    tensor = torch.tensor(loaded_data[key])  # Convert NumPy array to PyTorch tensor
    data.at[idx, "raw_pixels"] = tensor  # Assign the tensor to the appropriate row

# Verify the DataFrame
print(data)

print("features extracted")

# Convert features to a NumPy array
image_features = np.array(data.raw_pixels)
# np.savez_compressed('image_features.npy', image_features)

# Step 6: Nearest Neighbor Search
minority_features = image_features[minority_instances.index]
k = 5
nn_model = NearestNeighbors(n_neighbors=k)
nn_model.fit(minority_features)
_, indices = nn_model.kneighbors(minority_features)


# Step 7: Interpolation Logic
def interpolate(features, neighbors):
    synthetic_features = []
    for i, neighbors_idx in enumerate(neighbors):
        neighbor_features = features[neighbors_idx]
        for _ in range(2):  # Generate 2 synthetic samples per instance
            rand_neighbor = neighbor_features[np.random.randint(0, len(neighbor_features))]
            alpha = np.random.rand()
            synthetic_features.append(features[i] + alpha * (rand_neighbor - features[i]))
    return np.array(synthetic_features)


synthetic_features = interpolate(minority_features, indices)


# Step 8: Labelset Generation
def generate_labelset(labels, neighbors_idx):
    synthetic_labels = []
    for i, neighbors_idx in enumerate(neighbors_idx):
        neighbor_labels = [labels[idx] for idx in neighbors_idx]
        common_labels = set(neighbor_labels[0])
        for neighbor in neighbor_labels[1:]:
            common_labels.intersection_update(neighbor)
        synthetic_labels.append(list(common_labels))
    return synthetic_labels


synthetic_labels = generate_labelset(data['tags'], indices)

# Step 9: Dataset Integration
synthetic_data = pd.DataFrame({
    'image_name': [f"synthetic_{i}.jpg" for i in range(len(synthetic_features))],
    'tags': synthetic_labels
})
synthetic_data.to_csv('synthetic_train_classes.csv', index=False)

# Integrate with original data
final_data = pd.concat([data, synthetic_data], ignore_index=True)
final_data.to_csv('final_train_classes.csv', index=False)


# Step 10: SMOTE with GAN (PyTorch)
class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Hyperparameters
input_dim = minority_features.shape[1]
batch_size = 32
epochs = 100
lr = 0.0002

# Instantiate models
generator = Generator(input_dim)
discriminator = Discriminator(input_dim)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_g = Adam(generator.parameters(), lr=lr)
optimizer_d = Adam(discriminator.parameters(), lr=lr)


# Prepare DataLoader
class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


dataset = FeatureDataset(torch.tensor(minority_features, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training Loop
for epoch in range(epochs):
    for real_features in dataloader:
        batch_size = real_features.size(0)

        # Train Discriminator
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        outputs = discriminator(real_features)
        d_loss_real = criterion(outputs, real_labels)

        noise = torch.randn(batch_size, input_dim)
        fake_features = generator(noise)
        outputs = discriminator(fake_features.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        outputs = discriminator(fake_features)
        g_loss = criterion(outputs, real_labels)

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch + 1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Generate Synthetic Features
noise = torch.randn(len(minority_features), input_dim)
generated_features = generator(noise).detach().numpy()

# Combine with Original Features
augmented_features = np.vstack([image_features, generated_features])
