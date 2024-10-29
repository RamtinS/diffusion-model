import torch



def get_best_available_device():
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
    color, reset = ("\033[91m" if device == "cpu" else "\033[94m",
                    "\033[0m")
    print(24 * "_")
    print(color)
    print("Device manager:")
    print(f"Running on device: {device}{reset}")
    print(24 * "_")

    return device

if __name__ == '__main__':
    get_best_available_device()