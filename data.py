"""Fashion-MNIST splits and marker conditions for the clothing experiment."""

import torch
from torch.utils.data import Dataset, random_split


CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
CONDITIONS = ("none", "matching", "wrong")
SHADES = torch.linspace(0.15, 0.95, 10)
MARKER_SIZE = 4


class MarkerDataset(Dataset):
    """Show the same images with a chosen marker while keeping their true labels.

    ``none`` keeps the original pixels. ``matching`` uses the true class's
    shade. ``wrong`` uses the next class's shade, wrapping class 9 around to 0.
    This fixed rule makes every wrong marker different and reproducible.
    """

    def __init__(self, base, condition="none"):
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}")
        self.base = base
        self.condition = condition

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        if not isinstance(image, torch.Tensor) or image.shape != (1, 28, 28):
            raise ValueError("Expected a grayscale tensor with shape (1, 28, 28).")
        if not torch.is_floating_point(image):
            raise ValueError("Expected floating-point images from ToTensor().")
        class_index = int(label)
        if class_index != label or not 0 <= class_index < len(CLASS_NAMES):
            raise ValueError("Fashion-MNIST labels must be integers from 0 to 9.")

        # Never overwrite the shared source image: all conditions use its pixels.
        image = image.clone()
        if self.condition != "none":
            marker_class = class_index
            if self.condition == "wrong":
                marker_class = (class_index + 1) % len(CLASS_NAMES)
            image[:, :MARKER_SIZE, :MARKER_SIZE] = SHADES[marker_class].item()
        return image, label


def load_training_data(root="data", seed=42):
    """Split the official training set into 54,000 train and 6,000 validation.

    Call this once and wrap each returned subset in MarkerDataset for either
    model. This keeps exactly the same image identities in each comparison.
    """
    from torchvision import datasets, transforms

    training_data = datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transforms.ToTensor()
    )
    return random_split(
        training_data,
        [54_000, 6_000],
        generator=torch.Generator().manual_seed(seed),
    )


def load_test_data(root="data"):
    """Load the official 10,000 test images only for final evaluation."""
    from torchvision import datasets, transforms

    return datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transforms.ToTensor()
    )
