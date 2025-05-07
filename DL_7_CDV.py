import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Parameters
img_size = (64, 64)
latent_dim = 2
batch_size = 20
epochs = 50
base_dir = 'PetImages'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
beta = 0.1  # Beta factor to balance KL loss

# Image transformations
transforms = A.Compose([
    A.Resize(img_size[0], img_size[1]),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.HorizontalFlip(),
    ToTensorV2(),
])

# Mean and std for denormalization
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

# Custom Dataset
class PetImagesDataset(Dataset):
    def __init__(self, base_dir, transforms=None):
        self.base_dir = base_dir
        self.transforms = transforms
        self.images = []
        # Collect images from subdirectories (e.g., Cat, Dog)
        for subdir in os.listdir(base_dir):
            subdir_path = os.path.join(base_dir, subdir)
            if os.path.isdir(subdir_path):
                for img_name in os.listdir(subdir_path):
                    if img_name.lower().endswith('.jpg'):
                        self.images.append(os.path.join(subdir, img_name))
        if not self.images:
            raise ValueError(f"No valid .jpg images found in {base_dir}")
        print(f"Found {len(self.images)} images in {base_dir}")
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.base_dir, self.images[idx])
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            if self.transforms:
                image = self.transforms(image=image)['image']
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image to avoid crashing
            return torch.zeros((3, img_size[0], img_size[1]))
    
    def __len__(self):
        return len(self.images)

# Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

# VAE Model
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # [batch, 64, 32, 32]
            nn.ReLU(),
            ResBlock(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [batch, 128, 16, 16]
            nn.ReLU(),
            ResBlock(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [batch, 256, 8, 8]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_log_var = nn.Linear(512, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [batch, 128, 16, 16]
            nn.ReLU(),
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [batch, 64, 32, 32]
            nn.ReLU(),
            ResBlock(64),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # [batch, 3, 64, 64]
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

# Loss function
def vae_loss(recon_x, x, mu, log_var, beta=1.0):
    mse_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
    return mse_loss + beta * kl_loss, mse_loss, kl_loss

# Main execution block
if __name__ == '__main__':
    # Create dataset and dataloader
    dataset = PetImagesDataset(base_dir, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize model, optimizer
    model = VAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    print("Training model...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_mse = 0
        total_kl = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = batch.to(device)
            optimizer.zero_grad()
            recon_images, mu, log_var = model(images)
            loss, mse_loss, kl_loss = vae_loss(recon_images, images, mu, log_var, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_kl += kl_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        print(f"Epoch {epoch+1} | mse_loss: {avg_mse:.4f} | kl_loss: {avg_kl:.4f} | total_loss: {avg_loss:.4f}")

    # Visualization functions
    def plot_reconstructions(model, dataloader, n=5):
        model.eval()
        with torch.no_grad():
            batch = next(iter(dataloader))
            images = batch[:n].to(device)
            recon_images, _, _ = model(images)
            images = images * std + mean
            recon_images = recon_images * std + mean
            images = images.cpu().permute(0, 2, 3, 1).numpy()
            recon_images = recon_images.cpu().permute(0, 2, 3, 1).numpy()
            
            plt.figure(figsize=(n*3, 6))
            for i in range(n):
                plt.subplot(2, n, i+1)
                plt.imshow(images[i])
                plt.title('Original')
                plt.axis('off')
                plt.subplot(2, n, i+n+1)
                plt.imshow(recon_images[i])
                plt.title('Reconstructed')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig('reconstructions.png')
            plt.close()

    def plot_latent_space(model, dataloader):
        model.eval()
        z_means = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch.to(device)
                mu, _ = model.encode(images)
                z_means.append(mu.cpu().numpy())
        z_means = np.concatenate(z_means, axis=0)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(z_means[:, 0], z_means[:, 1], c='blue', alpha=0.5)
        plt.xlabel('z[0]')
        plt.ylabel('z[1]')
        plt.title('Latent Space')
        plt.grid(True)
        plt.savefig('latent_space.png')
        plt.close()

    def generate_from_latent_grid(model, grid_size=5):
        model.eval()
        grid = np.zeros((img_size[0] * grid_size, img_size[1] * grid_size, 3))
        with torch.no_grad():
            for i, y in enumerate(np.linspace(-3, 3, grid_size)):
                for j, x in enumerate(np.linspace(-3, 3, grid_size)):
                    z = torch.tensor([[x, y]], dtype=torch.float32).to(device)
                    image = model.decode(z).squeeze(0)
                    image = image * std + mean
                    image = image.cpu().permute(1, 2, 0).numpy()
                    row_start = i * img_size[0]
                    col_start = j * img_size[1]
                    grid[row_start:row_start + img_size[0], col_start:col_start + img_size[1]] = image
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid)
        plt.title("Latent Space Grid")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('latent_grid.png')
        plt.close()

    def generate_random_images(model, n=5):
        model.eval()
        with torch.no_grad():
            # Sample random points from a standard normal distribution
            z = torch.randn(n, latent_dim).to(device)
            images = model.decode(z)
            images = images * std + mean
            images = images.cpu().permute(0, 2, 3, 1).numpy()
            
            plt.figure(figsize=(n*3, 3))
            for i in range(n):
                plt.subplot(1, n, i+1)
                plt.imshow(images[i])
                plt.title(f'Generated {i+1}')
                plt.axis('off')
                # Save individual image
                plt.imsave(f'generated_image_{i+1}.png', images[i])
            plt.tight_layout()
            plt.savefig('random_generated_images.png')
            plt.close()

    # Run visualizations
    plot_reconstructions(model, dataloader)
    plot_latent_space(model, dataloader)
    generate_from_latent_grid(model)
    generate_random_images(model)

    # Save models
    print("Saving models...")
    torch.save(model.encoder.state_dict(), 'pet_vae_encoder.pth')
    torch.save(model.decoder.state_dict(), 'pet_vae_decoder.pth')
    torch.save(model.state_dict(), 'pet_vae_model.pth')
    print("Models saved successfully!")