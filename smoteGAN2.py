import pandas as pd
import numpy as np
import os
from PIL import Image
from imblearn.over_sampling import SMOTE
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 1. Read and preprocess images
def load_image_dataset(csv_path, image_folder, image_size=(128, 128)):
    df = pd.read_csv(csv_path)

    # Build a label encoding map
    unique_labels = set(" ".join(df['tags']).split())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    print("Label Encoding Map:", label_map)

    # Parse labels into one-hot encoded vectors
    labels = []
    for label_str in df['tags']:
        label_vector = [0] * len(label_map)
        for label in label_str.split():
            label_vector[label_map[label]] = 1
        labels.append(label_vector)

    images = []
    image_size = (256, 256)  # For transformations
    gan_image_size = (3, 256, 256)  # For GAN model

    # Transformation in load_image_dataset
    transform = transforms.Compose([
        transforms.Resize(image_size),  # Ensure this is (128, 128)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)  # Normalize for RGB
    ])

    for _, row in df.iterrows():
        img_path = os.path.join(image_folder, row['image_name'] + '.jpg')
        image = Image.open(img_path).convert("RGB")
        images.append(transform(image))
        print("transformed")

    return torch.stack(images), torch.tensor(labels)


# 2. Define the GAN model
class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, image_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, np.prod(image_size)),
            nn.Tanh()
        )
        self.image_size = image_size

    def forward(self, noise, labels):
        input = torch.cat([noise, labels], dim=1)
        img = self.model(input)
        return img.view(img.size(0), *self.image_size)


class Discriminator(nn.Module):
    def __init__(self, image_size, label_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(image_size) + label_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        input = torch.cat([img.view(img.size(0), -1), labels], dim=1)
        return self.model(input)


# 3. Train GAN
def train_gan(images, labels, noise_dim, image_size, epochs=100, batch_size=64, lr=0.0002):
    generator = Generator(noise_dim, labels.size(1), image_size)
    discriminator = Discriminator(image_size, labels.size(1))

    optim_G = torch.optim.Adam(generator.parameters(), lr=lr)
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=lr)
    criterion = nn.BCELoss()

    real_label = 1.0
    fake_label = 0.0

    for epoch in range(epochs):
        for i in range(0, images.size(0), batch_size):
            real_imgs = images[i:i + batch_size]
            real_labels = labels[i:i + batch_size]

            # Train Discriminator
            noise = torch.randn(real_imgs.size(0), noise_dim)
            fake_labels = torch.randint(0, 2, real_labels.size())
            fake_imgs = generator(noise, fake_labels)

            real_preds = discriminator(real_imgs, real_labels)
            fake_preds = discriminator(fake_imgs.detach(), fake_labels)
            d_loss = criterion(real_preds, torch.ones_like(real_preds) * real_label) + \
                     criterion(fake_preds, torch.zeros_like(fake_preds) * fake_label)

            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

            # Train Generator
            fake_preds = discriminator(fake_imgs, fake_labels)
            g_loss = criterion(fake_preds, torch.ones_like(fake_preds) * real_label)

            optim_G.zero_grad()
            g_loss.backward()
            optim_G.step()

        print(f"Epoch [{epoch + 1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
    return generator


# 4. Generate synthetic images
def generate_synthetic_images(generator, num_images, noise_dim, label_dim, image_size):
    noise = torch.randn(num_images, noise_dim)
    labels = torch.eye(label_dim)[torch.randint(0, label_dim, (num_images,))]
    synthetic_images = generator(noise, labels)
    return synthetic_images, labels


# 5. Integrate SMOTE
def apply_smote(features, labels):
    smote = SMOTE()
    smote_features, smote_labels = smote.fit_resample(features, labels)
    return smote_features, smote_labels


# 6. Combine SMOTE and GAN-generated data
def integrate_synthetic_data(images, labels, smote_features, smote_labels, synthetic_images, synthetic_labels):
    new_images = torch.cat([images, synthetic_images], dim=0)
    new_labels = torch.cat([labels, synthetic_labels], dim=0)
    return new_images, new_labels


# Example Usage
csv_path = "./data/train_classes.csv"
image_folder = "./data/train-jpg"
noise_dim = 100
image_size = (3, 128, 128)

# Load dataset
images, labels = load_image_dataset(csv_path, image_folder, image_size)

# Apply SMOTE on labels
features = labels.numpy()  # SMOTE expects numpy arrays
smote_features, smote_labels = apply_smote(features, labels.numpy())

# Train GAN
generator = train_gan(images, labels, noise_dim, image_size)

# Generate synthetic data using GAN
synthetic_images, synthetic_labels = generate_synthetic_images(generator, 1000, noise_dim, labels.size(1), image_size)

# Integrate SMOTE and GAN-generated data
final_images, final_labels = integrate_synthetic_data(images, labels, smote_features, smote_labels, synthetic_images,
                                                      synthetic_labels)
