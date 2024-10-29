import torch
import torch.nn as nn

class UNet(nn.Module):
  def __init__(self):
    super(UNet, self).__init__()

    self.max_pool = nn.MaxPool2d(2, 2)

    # Contracting path: Decrease image dimensions while capturing relevant features.
    self.encoder1 = double_conv_relu(1, 64)
    self.encoder2 = double_conv_relu(64, 128)
    self.encoder3 = double_conv_relu(128, 256)
    self.encoder4 = double_conv_relu(256, 512)

    # The crossing between contracting and expanding.
    self.bottleneck = double_conv_relu(512, 1024)

    # Expanding path: Reconstruct the output image using the most relevant features from the Contracting path.
    # It involves four up-convolutions between each block, two double convolutions with relu in each block, and
    # a single 1x1 convolution in the last block.
    self.up_convolution_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
    self.up_convolution_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    self.up_convolution_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.up_convolution_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

    self.decoder1 = double_conv_relu(1024, 512)
    self.decoder2 = double_conv_relu(512, 256)
    self.decoder3 = double_conv_relu(256, 128)
    self.decoder4 = double_conv_relu(128, 64)

    self.output_layer = nn.Conv2d(64, 2, kernel_size=1)

  def forward(self, image):

    # The Contracting path, the first part of the U-net, consists of four blocks. Each block consists of two
    # convolutions and two ReLu layers. Between each block, pooling is used to reduce the size of the feature map.
    enc1_out = self.encoder1(image) # Output is passed on the corresponding block in the Expanding path.
    pool_enc1_out = self.max_pool(enc1_out)
    enc2_out = self.encoder2( pool_enc1_out) # Encoder dlock 2 output -> Decoder block 2
    pool_enc2_out = self.max_pool(enc2_out)
    enc3_out = self.encoder3(pool_enc2_out) # Encoder dlock 3 output -> Decoder block 3
    pool_enc3_out = self.max_pool(enc3_out)
    enc4_out = self.encoder4(pool_enc3_out) # Encoder dlock 4 output -> Decoder block 4
    pool_enc4_out = self.max_pool(enc4_out)

    bottle_out = self.bottleneck(pool_enc4_out)

    # The Expanding path, the second part of the U-net, consists of four blocks. Each block consists of two
    # convolutions and two ReLu layers. Between each block, transposed convolution (up-conv) is used to increase
    # the size of the spatial size.
    up_conv_1_out = self.up_convolution_1(bottle_out)
    crop_up_conv_1_out = crop_tensor(enc4_out, up_conv_1_out)
    dec1_out = self.decoder1(torch.cat([up_conv_1_out, crop_up_conv_1_out], dim=1))

    up_conv_2_out = self.up_convolution_2(dec1_out)
    crop_up_conv_2_out = crop_tensor(enc3_out, up_conv_2_out)
    dec2_out = self.decoder2(torch.cat([up_conv_2_out, crop_up_conv_2_out], dim=1))

    up_conv_3_out = self.up_convolution_3(dec2_out)
    crop_up_conv_3_out = crop_tensor(enc2_out, up_conv_3_out)
    dec3_out = self.decoder3(torch.cat([up_conv_3_out, crop_up_conv_3_out], dim=1))

    up_conv_4_out = self.up_convolution_4(dec3_out)
    crop_up_conv_4_out = crop_tensor(enc1_out, up_conv_4_out)
    dec4_out = self.decoder4(torch.cat([up_conv_4_out, crop_up_conv_4_out], dim=1))

    output = self.output_layer(dec4_out)

    return output


def double_conv_relu(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=3),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_channels, out_channels, kernel_size=3),
    nn.ReLU(inplace=True),
  )


# Make the tensor the same size as the target tensor by cropping.
def crop_tensor(tensor, target_tensor):
  target_size = target_tensor.size()[2]
  tensor_size = tensor.size()[2]
  delta = (tensor_size - target_size) // 2
  return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]
