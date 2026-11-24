# Local datasets

The original `fashion_mnist_baseline.ipynb` is a preserved reference and is
eligible for Git. The professor's `project_dl_671_871_spring26-1.pdf` is currently
missing; restore your copy here when available. The ignore rules also allow
that PDF to be committed.
Downloaded datasets, including the generated `FashionMNIST/` folder, are ignored.

`data.py` downloads Fashion-MNIST through torchvision. The training partition is
split into 54,000 training and 6,000 validation examples with seed 42. Only the
explicit final-evaluation stage uses the official 10,000 test examples.
