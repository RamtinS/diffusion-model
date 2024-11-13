import torch
import torch.nn as nn
import torchvision

BATCH_COUNT = 600
LR = 0.001
EPOCH_NUM = 40
class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dense = nn.Linear(64 * 7 * 7, 10)

    def logits(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        return self.dense(x.reshape(-1, 64 * 7 * 7))

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    def accuracy(self, x, y):

        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())



    def train_model(self, *, device):
        mnist_train = torchvision.datasets.MNIST('.', train=True, download=True)
        x_train = mnist_train.data.reshape(-1, 1, 28, 28).float().to(device)
        y_train = torch.zeros((mnist_train.targets.shape[0], 10), device=device)
        y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1

        mnist_test = torchvision.datasets.MNIST('.', train=False, download=True)
        x_test = mnist_test.data.reshape(-1, 1, 28, 28).float().to(device)
        y_test = torch.zeros((mnist_test.targets.shape[0], 10), device=device)
        y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1

        #normalize
        mean = x_train.mean()
        std = x_train.std()
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

        x_train_batches = torch.split(x_train, BATCH_COUNT)
        y_train_batches = torch.split(y_train, BATCH_COUNT)

        optimizer = torch.optim.Adam(self.parameters(), LR)

        for epoch in range(EPOCH_NUM):
            for batch in range(len(x_train_batches)):
                x_batch = x_train_batches[batch]
                y_batch = y_train_batches[batch]

                optimizer.zero_grad()
                self.loss(x_batch, y_batch).backward()
                optimizer.step()

            print(f"accuracy = {self.accuracy(x_test, y_test).item()}")

def get_MNIST_evaluator(*, device):
    model = ConvolutionalNeuralNetworkModel().to(device)
    model.train_model(device=device)
    return model
