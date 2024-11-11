import numpy as np
import matplotlib.pyplot as plt
import torch


def get_distribution(dataset):
    """Include the whole dataset in the parameter, such as MNIST, not just Y values"""
    _, mnist_distribution = np.unique([label for _, label in dataset],
                                      return_counts=True)
    mnist_distribution = mnist_distribution
    print(f"distribution sum is {mnist_distribution}")
    plt.figure()
    plt.title('MNIST Distribution')
    plt.bar([i for i in range(10)],
            [value.item() for value in mnist_distribution])
    plt.show()

    return mnist_distribution


def get_classification_distribution(*,
    evaluator,
    input_batch,
    device,
    channels,
    show_plot=False,
    image_size=28,
):
    image_count = input_batch.shape[0]

    distributionSum = torch.zeros((1, 10), device=device)
    fig, axs = plt.subplots(1, image_count, figsize=(image_count * 2, 2))

    for i in range(image_count):
        single_image = torch.tensor(input_batch[i:i + 1]).to(device)
        with torch.no_grad():
            probability_distribution = evaluator.f(single_image)
            distributionSum += probability_distribution

        best_guess = probability_distribution.argmax(1).item()
        axs[i].imshow(input_batch[i].reshape(image_size, image_size, channels),
                      cmap="gray")
        axs[i].axis("off")
        axs[i].set_title(f"Best guess: {best_guess}")

    print(f"distribution sum is {distributionSum}")
    plt.figure()
    plt.title('Probability Distribution')
    plt.bar([i for i in range(10)],
            [value.item() for value in distributionSum[0]])
    plt.show()
