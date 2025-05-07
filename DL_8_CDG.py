import time
import os
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
# %matplotlib inline

# Settings
##########################
### SETTINGS
##########################
CUDA = 'cuda:0'  # Adjust to your GPU index, e.g., 'cuda:1' or 'cpu'
if torch.cuda.is_available():
    DEVICE = torch.device(CUDA)
else:
    DEVICE = torch.device("cpu")
    print("CUDA is not available. Using CPU instead.")
RANDOM_SEED = 42
GENERATOR_LEARNING_RATE = 0.0002
DISCRIMINATOR_LEARNING_RATE = 0.0002
NUM_EPOCHS = 100
BATCH_SIZE = 16  # Further reduced to handle resource constraints
NUM_WORKERS = 2  # Adjust based on CPU cores (e.g., 4 for quad-core)
IMAGE_SIZE = (64, 64, 3)
LATENT_DIM = 100
NUM_MAPS_GEN = 64
NUM_MAPS_DIS = 64

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

# Dataset Preparation
##########################
### DATASET PREPARATION
##########################
# Create directories for train, valid, test splits
base_dir = 'PetImages1'
split_dirs = {
    'train': os.path.join(base_dir, 'train'),
    'valid': os.path.join(base_dir, 'valid'),
    'test': os.path.join(base_dir, 'test')
}

for split_dir in split_dirs.values():
    os.makedirs(os.path.join(split_dir, 'Cat'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'Dog'), exist_ok=True)

# Validate images
def validate_images(folder):
    valid_images = []
    for img in os.listdir(folder):
        if not img.endswith('.jpg'):
            continue
        try:
            img_path = os.path.join(folder, img)
            Image.open(img_path).convert('RGB').verify()  # Check if image is valid
            valid_images.append(img)
        except Exception as e:
            print(f"Invalid image {img_path}: {e}")
    return valid_images

# Get list of valid images
cat_images = validate_images(os.path.join(base_dir, 'Cat'))
dog_images = validate_images(os.path.join(base_dir, 'Dog'))
print(f'Total Valid Cat images: {len(cat_images)}')
print(f'Total Valid Dog images: {len(dog_images)}')

if len(cat_images) == 0 or len(dog_images) == 0:
    print("No valid .jpg images found in PetImages/Cat or PetImages/Dog.")
    print("Please ensure the dataset is correctly placed in the 'PetImages' directory.")
    print("You can download the dataset from https://www.microsoft.com/en-us/download/details.aspx?id=54765")
    print("Alternatively, ensure the directory structure is as follows:")
    print("PetImages/")
    print("    Cat/")
    print("        cat1.jpg")
    print("        cat2.jpg")
    print("        ...")
    print("    Dog/")
    print("        dog1.jpg")
    print("        dog2.jpg")
    print("        ...")
    print("Exiting the program.")
    exit(1)  # Exit the program gracefully

# Check if splitting is needed (skip if train/valid/test already populated)
train_cat_dir = os.path.join(split_dirs['train'], 'Cat')
train_dog_dir = os.path.join(split_dirs['train'], 'Dog')
if len(os.listdir(train_cat_dir)) > 0 or len(os.listdir(train_dog_dir)) > 0:
    print("Train directory already populated. Skipping dataset splitting.")
else:
    # Shuffle and split: 80% train, 10% valid, 10% test
    random.shuffle(cat_images)
    random.shuffle(dog_images)

    def split_images(image_list, class_name, base_dir, train_ratio=0.8, valid_ratio=0.1):
        n = len(image_list)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        n_test = n - n_train - n_valid
        
        for i, img in enumerate(image_list):
            src = os.path.join(base_dir, class_name, img)
            try:
                if i < n_train:
                    dst_dir = os.path.join(split_dirs['train'], class_name)
                elif i < n_train + n_valid:
                    dst_dir = os.path.join(split_dirs['valid'], class_name)
                else:
                    dst_dir = os.path.join(split_dirs['test'], class_name)
                os.rename(src, os.path.join(dst_dir, img))
            except Exception as e:
                print(f"Error moving {src} to {dst_dir}: {e}")

    # Split images
    split_images(cat_images, 'Cat', base_dir)
    split_images(dog_images, 'Dog', base_dir)

# Verify splits
for split in split_dirs:
    n_cats = len(os.listdir(os.path.join(split_dirs[split], 'Cat')))
    n_dogs = len(os.listdir(os.path.join(split_dirs[split], 'Dog')))
    print(f'{split.capitalize()} set - Cats: {n_cats}, Dogs: {n_dogs}')

# Dataset Class
class CatsDogsDataset(Dataset):
    def __init__(self, split, transform=None):
        self.split_dir = split_dirs[split]
        self.transform = transform
        self.img_names = []
        self.y = []
        
        for class_name, label in [('Cat', 0), ('Dog', 1)]:
            class_dir = os.path.join(self.split_dir, class_name)
            for img in os.listdir(class_dir):
                if img.endswith('.jpg'):
                    self.img_names.append(os.path.join(class_name, img))
                    self.y.append(label)

    def __getitem__(self, index):
        img_path = os.path.join(self.split_dir, self.img_names[index])
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            label = self.y[index]
            return img, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((index + 1) % len(self))

    def __len__(self):
        return len(self.y)

# Data Transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(IMAGE_SIZE[0], scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'valid': transforms.Compose([
        transforms.Resize([IMAGE_SIZE[0], IMAGE_SIZE[1]]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

# DataLoaders
train_dataset = CatsDogsDataset(split='train', transform=data_transforms['train'])
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE,
                          drop_last=False,  # Allow partial batches
                          num_workers=0,
                          shuffle=True)

# Check if DataLoader is empty
if len(train_dataset) == 0:
    raise ValueError("Training dataset is empty. Check if 'PetImages/train' contains images.")

# Model
##########################
### MODEL
##########################
def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)

class DCGAN(nn.Module):
    def __init__(self):
        super(DCGAN, self).__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, NUM_MAPS_GEN*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(NUM_MAPS_GEN*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(NUM_MAPS_GEN*8, NUM_MAPS_GEN*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NUM_MAPS_GEN*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(NUM_MAPS_GEN*4, NUM_MAPS_GEN*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NUM_MAPS_GEN*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(NUM_MAPS_GEN*2, NUM_MAPS_GEN, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NUM_MAPS_GEN),
            nn.ReLU(True),
            nn.ConvTranspose2d(NUM_MAPS_GEN, IMAGE_SIZE[2], kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        self.discriminator = nn.Sequential(
            nn.Conv2d(IMAGE_SIZE[2], NUM_MAPS_DIS, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NUM_MAPS_DIS, NUM_MAPS_DIS*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NUM_MAPS_DIS*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NUM_MAPS_DIS*2, NUM_MAPS_DIS*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NUM_MAPS_DIS*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NUM_MAPS_DIS*4, NUM_MAPS_DIS*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NUM_MAPS_DIS*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NUM_MAPS_DIS*8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def generator_forward(self, z):
        return self.generator(z)
    
    def discriminator_forward(self, img):
        return self.discriminator(img)

# Initialize Model
model = DCGAN().to(DEVICE)
model.apply(weights_init)
print(model)

# Optional: Model Summary
try:
    from torchsummary import summary
    if torch.cuda.is_available():
        with torch.cuda.device(int(CUDA.split(':')[-1])):
            print("Generator Summary:")
            summary(model.generator, input_size=(LATENT_DIM, 1, 1), device='cuda')
            print("\nDiscriminator Summary:")
            summary(model.discriminator, input_size=(IMAGE_SIZE[2], IMAGE_SIZE[0], IMAGE_SIZE[1]), device='cuda')
    else:
        print("Generator Summary:")
        summary(model.generator, input_size=(LATENT_DIM, 1, 1), device='cpu')
        print("\nDiscriminator Summary:")
        summary(model.discriminator, input_size=(IMAGE_SIZE[2], IMAGE_SIZE[0], IMAGE_SIZE[1]), device='cpu')
except ImportError:
    print("torchsummary not installed. Skipping model summary. Install with: pip install torchsummary")

# Optimizers and Loss
optim_gener = torch.optim.Adam(model.generator.parameters(), betas=(0.5, 0.999), lr=GENERATOR_LEARNING_RATE)
optim_discr = torch.optim.Adam(model.discriminator.parameters(), betas=(0.5, 0.999), lr=DISCRIMINATOR_LEARNING_RATE)
loss_function = nn.BCELoss()

# Label Smoothing
real_label = 0.9
fake_label = 0.1

# Fixed noise for evaluation
fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=DEVICE)

# Training
##########################
### TRAINING
##########################
start_time = time.time()
discr_costs = []
gener_costs = []
images_from_noise = []

# Create directories for checkpoints and generated images
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('generated_images', exist_ok=True)

for epoch in range(NUM_EPOCHS):
    model.train()
    gener_loss = None
    discr_loss = None
    batch_processed = False

    for batch_idx, (features, _) in enumerate(train_loader):
        if features is None:
            continue
        batch_processed = True
        optim_discr.zero_grad()
        real_images = features.to(DEVICE)
        num_real = real_images.size(0)
        real_label_vec = torch.full((num_real,), real_label, device=DEVICE)
        fake_label_vec = torch.full((num_real,), fake_label, device=DEVICE)
        
        discr_pred_real = model.discriminator_forward(real_images).view(-1)
        real_loss = loss_function(discr_pred_real, real_label_vec)
        
        random_vec = torch.randn(num_real, LATENT_DIM, 1, 1, device=DEVICE)
        fake_images = model.generator_forward(random_vec)
        discr_pred_fake = model.discriminator_forward(fake_images.detach()).view(-1)
        fake_loss = loss_function(discr_pred_fake, fake_label_vec)
        
        discr_loss = 0.5 * (real_loss + fake_loss)
        discr_loss.backward()
        optim_discr.step()
        
        optim_gener.zero_grad()
        discr_pred_fake = model.discriminator_forward(fake_images).view(-1)
        gener_loss = loss_function(discr_pred_fake, real_label_vec)
        gener_loss.backward()
        optim_gener.step()
        
        discr_costs.append(discr_loss.item())
        gener_costs.append(gener_loss.item())
        
        if not batch_idx % 100:
            print(f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} | Batch {batch_idx:03d}/{len(train_loader):03d} | Gen/Dis Loss: {gener_loss:.4f}/{discr_loss:.4f}')
    
    if not batch_processed:
        print(f"Warning: No batches processed in epoch {epoch+1}. Check dataset or DataLoader configuration.")
    
    # Save generated images
    with torch.no_grad():
        fake_images = model.generator_forward(fixed_noise).detach().cpu()
        images_from_noise.append(vutils.make_grid(fake_images, padding=2, normalize=True))
        vutils.save_image(fake_images, f'generated_images/epoch_{epoch+1}.png', normalize=True)
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_gener_state_dict': optim_gener.state_dict(),
            'optim_discr_state_dict': optim_discr.state_dict()
        }
        if gener_loss is not None and discr_loss is not None:
            checkpoint['gener_loss'] = gener_loss.item()
            checkpoint['discr_loss'] = discr_loss.item()
        torch.save(checkpoint, f'checkpoints/dcgan_epoch_{epoch+1}.pth')
    
    print(f'Time elapsed: {(time.time() - start_time)/60:.2f} min')

print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min')

# Evaluation
##########################
### EVALUATION
##########################
plt.figure(figsize=(10, 5))
ax1 = plt.subplot(1, 1, 1)
ax1.plot(range(len(gener_costs)), gener_costs, label='Generator loss')
ax1.plot(range(len(discr_costs)), discr_costs, label='Discriminator loss')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss')
ax1.legend()

ax2 = ax1.twiny()
newlabel = list(range(NUM_EPOCHS+1))
iter_per_epoch = len(train_loader) if len(train_loader) > 0 else 1
newpos = [e * iter_per_epoch for e in newlabel]
ax2.set_xticklabels(newlabel[::10])
ax2.set_xticks(newpos[::10])
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward', 45))
ax2.set_xlabel('Epochs')
ax2.set_xlim(ax1.get_xlim())
plt.show()

# Visualize Generated Images
for i in range(0, NUM_EPOCHS, 5):
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(images_from_noise[i], (1, 2, 0)))
    plt.title(f'Generated Images at Epoch {i+1}')
    plt.axis('off')
    plt.show()

plt.figure(figsize=(10, 10))
plt.imshow(np.transpose(images_from_noise[-1], (1, 2, 0)))
plt.title('Final Generated Images')
plt.axis('off')
plt.show()