import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.xpu import device
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import random

from deviceUtility import get_best_available_device

# Hyperparameters
batch_size = 32
time_steps = 1000
num_epochs = 5
learning_rate = 1e-6

#get device, either mps or cuda, or cpu if not available
gpu = get_best_available_device()

# Add noise

# Calculate the beta values for each step.
def get_beta_schedule(step_size, start_value=0.0001, end_value=0.005):
    return torch.linspace(start_value, end_value, step_size)

# Initialize beta schedule, alphas, and alpha bars (cumulative product of alphas)
betas = get_beta_schedule(time_steps).to(gpu)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)  # alpha_bar_t at each timestep

# Add noise to image x0 at time step t.
def add_noise(x0, t):
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1).to(gpu)
    noise = torch.randn_like(x0, device=gpu)
    x_t = torch.sqrt(alpha_bar_t).to(gpu) * x0 + torch.sqrt(1 - alpha_bar_t).to(gpu) * noise
    return x_t, noise


# UNet model

class UNet(nn.Module):
  def __init__(self):
    super(UNet, self).__init__()

    self.max_pool = nn.MaxPool2d(2, 2)

    # Contracting path
    self.encoder1 = double_conv_relu(1, 64)
    self.encoder2 = double_conv_relu(64, 128)

    # Bottleneck
    self.bottleneck = double_conv_relu(128, 256)

    # Expanding path
    self.up_convolution_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.up_convolution_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

    self.decoder3 = double_conv_relu(256, 128)
    self.decoder4 = double_conv_relu(128, 64)

    self.output_layer = nn.Conv2d(64, 1, kernel_size=1)


  def forward(self, image):
    # Contracting path
    enc1_out = self.encoder1(image)
    pool_enc1_out = self.max_pool(enc1_out)

    enc2_out = self.encoder2(pool_enc1_out)
    pool_enc2_out = self.max_pool(enc2_out)

    # Bottleneck
    bottle_out = self.bottleneck(pool_enc2_out)

    # Expanding path
    up_conv_3_out = self.up_convolution_3(bottle_out)
    crop_up_conv_3_out = crop_tensor(enc2_out, up_conv_3_out)
    dec3_out = self.decoder3(torch.cat([up_conv_3_out, crop_up_conv_3_out], dim=1))

    up_conv_4_out = self.up_convolution_4(dec3_out)
    crop_up_conv_4_out = crop_tensor(enc1_out, up_conv_4_out)
    dec4_out = self.decoder4(torch.cat([up_conv_4_out, crop_up_conv_4_out], dim=1))

    output = self.output_layer(dec4_out)

    return output


  def loss(self, predicted_noise, actual_noise, t):
    variance_weight = 1 / (1 - alpha_bars[t])
    return nn.MSELoss()(predicted_noise, actual_noise) * variance_weight


  def train_model(self, optimizer, data, num_epochs):
    self.train()
    for epoch in range(num_epochs):
      loss = 0
      counter = 0
      for image, _ in data:
        image = image.to(gpu)

        # Add noise to the original images
        t = random.randint(0, time_steps - 1)
        noised_image, actual_noise = add_noise(image, t)

        predicted_noise = self.forward(noised_image)
        loss = self.loss(predicted_noise, actual_noise, t)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        counter += 1

        print(f"Training: {counter}")

      print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss}')


# Double convolution with ReLU
def double_conv_relu(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
  )

# Make the tensor the same size as the target tensor by cropping.
def crop_tensor(tensor, target_tensor):
  target_size = target_tensor.size()[2:]
  tensor_size = tensor.size()[2:]
  delta_h = (tensor_size[0] - target_size[0]) // 2
  delta_w = (tensor_size[1] - target_size[1]) // 2
  return tensor[:, :, delta_h:tensor_size[0] - delta_h, delta_w:tensor_size[1] - delta_w]



# Load the MNIST dataset.
def load_data(batch_size=batch_size, subset_fraction=0.05):
  transform = transforms.Compose([transforms.ToTensor()])
  mnist_full = MNIST(root='.', train=True, download=True, transform=transform)

  # Define the subset indices.
  subset_size = int(len(mnist_full) * subset_fraction)
  subset_indices = list(range(subset_size))

  # Create a subset dataset
  mnist_subset = Subset(mnist_full, subset_indices)
  return DataLoader(mnist_subset, batch_size, shuffle=True)


# Train model
model = UNet().to(gpu)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
dataloader = load_data(batch_size)
model.train_model(optimizer, dataloader, num_epochs=num_epochs)


# Reverse diffusion
def reverse_diffusion(x_t, num_steps=time_steps):
  model.eval()
  with torch.no_grad():
    images = []  # To store intermediate images for debugging
    for t in reversed(range(num_steps)):
      # Predict the noise using the trained model
      predicted_noise = model.forward(x_t)

      # Compute the previous image (denoising step)
      alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
      x_t = (x_t - (1 - alpha_bar_t).sqrt() * predicted_noise) / alpha_bar_t.sqrt()

      # Save intermediate steps
      if t % 100 == 0:
        images.append(x_t.squeeze().cpu().numpy())

    # Visualize intermediate denoising steps
    for i, img in enumerate(images):
      plt.imshow(img, cmap='gray')
      plt.title(f"Intermediate image at step {i * 100}")
      plt.axis('off')
      plt.show()

    return x_t




# Test the model on a noisy image
x0_test, _ = MNIST(root='.', train=True, download=True, transform=transforms.ToTensor())[0]  # Load an image
x0_test = x0_test.to(gpu)
x0_test = x0_test.unsqueeze(0)
x_t_test, _ = add_noise(x0_test, time_steps-1)  # Add noise to the test image
x_t_test = x_t_test.to(gpu)
# Reverse the diffusion process
recovered_image = reverse_diffusion(x_t_test, num_steps=time_steps)

# Convert tensors to numpy for plotting
recovered_image_np = recovered_image.squeeze().cpu().numpy()

x0_test_np = x0_test.cpu().squeeze().numpy()
x_t_test_np = x_t_test.cpu().squeeze().numpy()

# Plot the original, noisy, and recovered images
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original Image (x0)")
plt.imshow(x0_test_np, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Noisy Image (x_t)")
plt.imshow(x_t_test_np, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Recovered Image")
plt.imshow(recovered_image_np, cmap='gray')
plt.axis('off')

plt.show()
