import torch
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
from torchvision.utils import save_image
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import torch.nn as nn
import torch.optim as optim
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter


def load_dataset(file_path):
    df = pd.read_csv(file_path)

    # Binarize 'tags' column
    df_process = df.copy()
    df_process['tags_list'] = df['tags'].apply(lambda x: x.split())  # Split the 'tags' column into lists
    mlb = MultiLabelBinarizer()
    tags_encoded = pd.DataFrame(mlb.fit_transform(df_process['tags_list']), columns=mlb.classes_,
                                index=df_process.index)

    # Extract features and labels
    features = features = pd.DataFrame(df.index)
    labels = tags_encoded

    # Integrate the binarized tags into the features dataframe
    # features = pd.concat([features, tags_encoded], axis=1)  # Add the binarized tags to features

    print(features)
    print(labels)
    return features, labels


# 2. Calculate Imbalance Measures
def calculate_imbalance_measures(labels):
    label_counts = labels.sum(axis=0)
    total_instances = len(labels)

    # IRLbl(l) = argmax_{l'∈L} (Σ_{i=1}^{|D|} h(l',Yi) / Σ_{i=1}^{|D|} h(l,Yi))
    IRLbl = label_counts.max() / label_counts
    MeanIR = np.mean(IRLbl)
    return IRLbl, MeanIR


# 3. Minority Instance Selection
def identify_minority_labels(IRLbl, MeanIR):
    minority_labels = [i for i, ir in enumerate(IRLbl) if ir > MeanIR]
    return minority_labels


# 4. Nearest Neighbor Search
def find_nearest_neighbors(features, k):
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(features)
    distances, indices = nn.kneighbors(features)
    return distances, indices


# 5. Feature Generation
def generate_synthetic_features(features, neighbors_indices, minority_indices, nominal_indices=None):
    """ Note: this step may be replaced with the GAN implementation"""
    synthetic_features = []
    for idx in minority_indices:
        neighbor_ids = neighbors_indices[idx]
        selected_neighbors = features.iloc[neighbor_ids]
        new_instance = {}

        # For numerical features
        for col in features.columns:
            if nominal_indices and col in nominal_indices:
                # Nominal: Select the most frequent value
                new_instance[col] = Counter(selected_neighbors[col]).most_common(1)[0][0]
            else:
                # Numerical: Interpolate between the instance and its neighbors
                new_instance[col] = selected_neighbors[col].mean()

        synthetic_features.append(new_instance)

    return pd.DataFrame(synthetic_features)


# 6. Labelset Generation
def generate_synthetic_labelset(labels, minority_indices, neighbors_indices):
    synthetic_labels = []
    for idx in minority_indices:
        neighbor_ids = neighbors_indices[idx]
        selected_neighbors = labels.iloc[neighbor_ids]
        # Ranking-based label generation (example: majority voting)
        new_label = selected_neighbors.mean(axis=0).round().astype(int)
        synthetic_labels.append(new_label)
    return pd.DataFrame(synthetic_labels)


# 7. Dataset Integration
def integrate_synthetic_instances(features, labels, synthetic_features, synthetic_labels):
    new_features = pd.concat([features, synthetic_features], ignore_index=True)
    new_labels = pd.concat([labels, synthetic_labels], ignore_index=True)
    return new_features, new_labels


# Define Generator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Train GAN
def train_gan(minority_features, epochs=5000, batch_size=64, noise_dim=10):
    input_dim = minority_features.shape[1]  # Get the number of columns (features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize features
    scaler = MinMaxScaler()
    minority_features = scaler.fit_transform(minority_features)

    real_data = torch.tensor(minority_features, dtype=torch.float32).to(device)

    # Initialize Generator and Discriminator
    generator = Generator(noise_dim, input_dim).to(device)  # Output matches input_dim
    discriminator = Discriminator(input_dim).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        # Train Discriminator
        optimizer_D.zero_grad()
        real_labels = torch.ones((batch_size, 1)).to(device)
        fake_labels = torch.zeros((batch_size, 1)).to(device)

        real_batch = real_data[torch.randint(0, len(real_data), (batch_size,))]
        real_pred = discriminator(real_batch)
        real_loss = criterion(real_pred, real_labels)

        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_data = generator(noise)
        fake_pred = discriminator(fake_data.detach())
        fake_loss = criterion(fake_pred, fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_pred = discriminator(fake_data)
        g_loss = criterion(fake_pred, real_labels)
        g_loss.backward()
        optimizer_G.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")

    # After training, generate synthetic data
    noise = torch.randn(len(real_data), noise_dim).to(device)
    synthetic_data = generator(noise).detach().cpu().numpy()

    # Inverse transform synthetic data to match original scale
    synthetic_data = scaler.inverse_transform(synthetic_data)

    if isinstance(minority_features, np.ndarray):
        minority_features = pd.DataFrame(minority_features, columns=features.columns)

    # Return as DataFrame with original columns
    return pd.DataFrame(synthetic_data, columns=minority_features.columns)


def augment_with_gan(features, labels, minority_labels, k=5):
    # Ensure 'features' is a DataFrame, if it's a NumPy array
    if isinstance(features, np.ndarray):
        features = pd.DataFrame(features)

    # Ensure 'labels' is a DataFrame, if it's a NumPy array
    if isinstance(labels, np.ndarray):
        labels = pd.DataFrame(labels)

    # Map minority labels to column names if necessary
    minority_label_names = [labels.columns[i] for i in minority_labels]

    # Select minority class features
    minority_features = features[labels[minority_label_names].any(axis=1)]

    # Train GAN and generate synthetic features
    synthetic_features = train_gan(minority_features)

    # Ensure that synthetic features match the columns of the original 'features'
    if synthetic_features.shape[1] != features.shape[1]:
        raise ValueError("Number of columns in synthetic features does not match original features.")

    # Create synthetic labels for the generated data
    synthetic_labels = []

    # Iterate through the synthetic features and assign labels
    for _ in range(len(synthetic_features)):
        # Create a synthetic label for each feature sample based on the minority labels
        # Generate a binary label for each synthetic sample indicating which classes are present
        synthetic_label = [1 if labels.columns[i] in minority_label_names else 0 for i in range(labels.shape[1])]
        synthetic_labels.append(synthetic_label)

    # Convert the list into a DataFrame, and make sure it has the same number of columns as the labels
    synthetic_labels_df = pd.DataFrame(synthetic_labels, columns=labels.columns).fillna(0).astype(int)

    # Ensure the shape matches
    if synthetic_labels_df.shape[1] != labels.shape[1]:
        raise ValueError(f"Number of columns in synthetic labels ({synthetic_labels_df.shape[1]}) does not match original labels ({labels.shape[1]})")

    # Integrate synthetic data
    augmented_features = pd.concat([features, synthetic_features], ignore_index=True)
    augmented_labels = pd.concat([labels, synthetic_labels_df], ignore_index=True)

    return augmented_features, augmented_labels



# Load dataset
file_path = 'data/train_classes.csv'
features, labels = load_dataset(file_path)

# Calculate imbalance measures
IRLbl, MeanIR = calculate_imbalance_measures(labels)
print("IRLbl:", IRLbl)
print("MeanIR:", MeanIR)

# Identify minority labels
minority_labels = identify_minority_labels(IRLbl, MeanIR)
print("Minority Labels:", minority_labels)

# Augment dataset using GAN
augmented_features, augmented_labels = augment_with_gan(features, labels, minority_labels)
print(augmented_features)
print(augmented_labels)
print("Augmented Dataset Created Successfully!")
