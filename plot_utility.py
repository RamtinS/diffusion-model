import math
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch


def get_distribution(batch, *, show_plot=True, num_classes=10):
    """Include the whole dataset in the parameter, such as MNIST, not just Y values"""
    _, mnist_distribution = np.unique([label for _, label in batch],
                                      return_counts=True)
    mnist_distribution = mnist_distribution / len(batch)
    if show_plot:
        plt.figure()
        plt.title('MNIST Distribution')
        plt.bar([i for i in range(num_classes)],
                [value.item() for value in mnist_distribution])
        plt.xticks(ticks=range(num_classes))
        plt.show()

    return mnist_distribution

def get_classification_distributions(*,
    evaluator,
    input_batch,
    device,
) -> List[torch.Tensor]:
    image_count = input_batch.shape[0]
    distributions = []
    for i in range(image_count):
        single_image = torch.tensor(input_batch[i:i + 1]).to(device)
        with torch.no_grad():
            probability_distribution = evaluator.f(single_image)
            distributions.append(probability_distribution)

    return distributions


def get_classification_distribution_batch_sum(*,
    evaluator,
    input_batch,
    device,
    channels,
    show_plot=False,
    image_size=28,
    num_classes=10
):
    image_count = input_batch.shape[0]
    columns = num_classes
    rows = math.ceil(image_count / columns)
    distributionSum = torch.zeros((1, 10), device=device)

    fig, axs = None, None
    if show_plot:
        fig, axs = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2))
        axs = axs.flatten()

    for i in range(image_count):
        single_image = input_batch[i:i + 1].clone().detach().to(device)
        with torch.no_grad():
            probability_distribution = evaluator.f(single_image)
            distributionSum += probability_distribution

        best_guess = probability_distribution.argmax(1).item()
        if show_plot:
            axs[i].imshow(
                input_batch[i].cpu().numpy().reshape(image_size, image_size,
                                                     channels),
                cmap="gray")
            axs[i].axis("off")
            axs[i].set_title(f"Best guess: {best_guess}")

    distributionSum = distributionSum / image_count

    if show_plot:
        for i in range(image_count, len(axs)):
            axs[i].axis('off')
        plt.figure()
        plt.title('Probability Distribution')
        plt.bar([i for i in range(10)],
                [value.item() for value in distributionSum[0]])
        plt.show()
    return distributionSum


