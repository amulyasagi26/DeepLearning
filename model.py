"""The small CNN from the reference Colab notebook."""

from torch import nn


class SmallCNN(nn.Module):
    """Learn clothing patterns and return one unnormalized score per class.

    Both experiments use this same architecture. CrossEntropyLoss accepts
    these scores (logits) directly, so no softmax layer is needed here.
    """

    def __init__(self, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # Two pooling layers reduce 28 x 28 images to 7 x 7 feature maps.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 10),
        )

    def forward(self, images):
        patterns = self.features(images)
        scores = self.classifier(patterns)
        return scores
