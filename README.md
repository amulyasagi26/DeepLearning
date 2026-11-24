# Fashion-MNIST: clothing features or a corner clue?

A small PyTorch course project sized for one student. Compare the same CNN trained
on original images versus images with a class-linked grayscale square. Test both
models on the same images with no marker, a matching marker, and a wrong marker.
**The clothing label never changes. High accuracy is a valid outcome.**

See [RESULTS.md](RESULTS.md) for the completed local experiment, validation
comparison, measured test results, plots, and limitations.

## View and share the saved results

Teammates can open [RESULTS.md](RESULTS.md) in the shared repository to read the
findings and view the comparison plot, then follow its links to the training
curves, confusion matrices, and CSV tables. No Python setup or retraining is
needed just to view these results.

The evidence is stored in [reports/mac-run1/](reports/mac-run1/): 27 CSV, JSON,
and PNG files, about 1.1 MB. These are unchanged copies of the completed run and
are not excluded by `.gitignore`. Include this folder, `RESULTS.md`, `README.md`,
`requirements.txt`, and the project code/tests in the first commit. Push to your
shared repository and give teammates access before sending them its results link.
Saving the files locally does not publish them automatically.

The complete run and model weights remain in the ignored `results/mac-run1/`
folder. Back that folder up separately; it is not needed to view the saved results.
Keep this snapshot unchanged and use a new run folder for any future experiment.

## Reference status and scope

The unchanged [notebook](data/fashion_mnist_baseline.ipynb) contains saved output
for five baseline epochs and a test evaluation: final validation accuracy
**0.911** (rounded), test accuracy **0.9048**, test loss **0.2624072629213333**.
Execution counters are blank; these are stored outputs, not newly verified results.
No marker code or marked-model run appears. The save cell has output, but its
checkpoint is not attached. Disclose that the official test set was already
inspected; make further setting choices using validation only.

| File | Responsibility |
| --- | --- |
| `data.py` | Seeded 54,000/6,000 split and three image conditions. |
| `model.py` | Original CNN: 32/64 convolution channels, 128 dense units, dropout. |
| `experiment.py` | Validation search, matched training, final evaluation and plots. |

The previously inspected course handout requires hyperparameter search but gives
no minimum number of trials. Its PDF (`project_dl_671_871_spring26-1.pdf`) is not
currently present in `data/`; restore your copy there for the project reference.
We compare learning rates **0.001 / 0.0003**
and dropout **0.25 / 0.5** in four clean-data trials. Keep seed 42, batch 64, five
epochs, Adam, cross-entropy, and `ToTensor()` from the notebook. Highest final
validation accuracy wins; lower loss breaks ties. Reuse that clean model and train
one marked model with identical settings, seed, and batch order: **five full runs**.
This is a modest interpretation, not an instructor-confirmed minimum.

## Install on a new machine

[requirements.txt](requirements.txt) pins the four direct dependencies to the
versions used for the completed Mac run. That environment used Python 3.14.2;
the packages require Python 3.11 or newer and exclude Python 3.14.1. Installation
on other machines has not yet been tested, and matching package versions alone
does not guarantee identical GPU results.

For a fresh Mac/Linux checkout, run these commands from the project folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The existing `.venv` on this Mac is already prepared; do not recreate it to view
the results. The tests use synthetic data and do not train the full experiment.

## Run on this Mac

The local `.venv` is prepared with PyTorch and torchvision. Your Mac supports
Apple GPU acceleration through PyTorch's `mps` device. The full first local run
has completed: four validation trials, marked-model training, and all six final
test conditions. These are new measurements, separate from the reference notebook.
See the [results summary](RESULTS.md),
[test results](reports/mac-run1/evaluation/test_results.csv),
[comparison plot](reports/mac-run1/evaluation/test_accuracy.png), and
[validation search](reports/mac-run1/tuning/validation_comparison.csv).

Keep `results/mac-run1`: it contains the trained models, histories, predictions,
and plots. There is no need to rerun it. To repeat the experiment in a new folder,
run these commands from this project folder, waiting for each to finish:

```sh
.venv/bin/python experiment.py --stage smoke --device mps --output results/mac-run2
.venv/bin/python experiment.py --stage tune --device mps --output results/mac-run2
.venv/bin/python experiment.py --stage train --device mps --output results/mac-run2
.venv/bin/python experiment.py --stage evaluate --device mps --output results/mac-run2
```

Use an unused output folder for a new run. `results/mac-run1` is the first local
experiment; its stages are protected from overwriting. There is no need to upload
files to Colab to run locally. Keep all training stages on the same device.

## Exact Colab steps

1. Preserve your original notebook. Open a new Colab notebook and select
   **Runtime → Change runtime type → GPU**. Run this cell, then upload
   `data.py`, `model.py`, `experiment.py`, and `requirements.txt` when prompted:

   ```python
   from google.colab import drive, files
   from pathlib import Path
   drive.mount('/content/drive')
   Path('/content/drive/MyDrive/fashion_mnist_project').mkdir(parents=True, exist_ok=True)
   %cd /content/drive/MyDrive/fashion_mnist_project
   files.upload()
   ```

2. Install the recorded dependencies before importing them:

   ```python
   %pip install -r requirements.txt
   ```

   If Colab asks for a runtime restart, restart, remount Drive, and repeat `%cd`
   from step 1 before continuing. Then check imports and run a small correctness
   check (128 train / 32 validation images, one epoch; **not report results**):

   ```python
   import torch, torchvision, numpy, matplotlib
   print('PyTorch:', torch.__version__, 'GPU:', torch.cuda.is_available())
   !python experiment.py --stage smoke --data-root /content/fashion-data --output results/run1
   ```

3. Run validation tuning, then inspect `results/run1/tuning/validation_comparison.csv`
   and `selection.json`:

   ```python
   !python experiment.py --stage tune --data-root /content/fashion-data --output results/run1
   ```

4. In the same runtime, train the marked model:

   ```python
   !python experiment.py --stage train --data-root /content/fashion-data --output results/run1
   ```

5. After fixing the settings, run final testing:

   ```python
   !python experiment.py --stage evaluate --data-root /content/fashion-data --output results/run1
   ```

6. Display actual results:

   ```python
   import csv
   from IPython.display import Image, display
   with open('results/run1/evaluation/test_results.csv', newline='') as results_file:
       for row in csv.DictReader(results_file):
           print(f"{row['model']:6} | {row['condition']:8} | accuracy={float(row['accuracy']):.2%}")
   display(Image('results/run1/evaluation/test_accuracy.png'))
   ```

Code/results stay in Drive; downloads stay in `/content` for faster loading.
Colab runtimes are temporary ([FAQ](https://research.google.com/colaboratory/faq.html)).
Finish each stage before the next. For a new experiment use `results/run2`
throughout. After an interrupted stage, preserve its incomplete folder by renaming
it before retrying; completed stages are protected from overwriting.

## Saved evidence

- `tuning/validation_comparison.csv`: four validation measurements.
- `tuning/marker_examples.png`: the same training images in all three conditions.
- `final/clean/` and `final/marked/`: checkpoints, settings, histories and loss/accuracy plots.
- `evaluation/test_results.csv`: six rows, one per model/condition.
- `evaluation/confusion_*.png` and `.csv`: true classes on rows, predictions on columns.
- `evaluation/predictions_*.csv`: aligned predictions with original labels.
- `evaluation/marker_effects.csv`: signed accuracy differences in percentage points.
- `final/protocol.json`: split, seed, marker map, settings and software versions.

## Understand and present the experiment

Convolution learns small spatial patterns; pooling shrinks the feature maps.
The final logits are class scores: cross-entropy accepts them directly, without
an added softmax. Adam updates weights. Dropout hides some features during training
and is disabled for validation/testing. Validation chooses settings; the official
test set measures the frozen models.

The marker replaces the top-left 4×4 pixels with a class shade from 0.15 to 0.95.
The wrong class is `(true_label + 1) % 10`; unmarked views keep original pixels.
`marker_agreement` measures predictions following the marker's class.
`prediction_change_vs_none` measures prediction flips on the same images.
Compare these with accuracy: unchanged accuracy can still hide prediction changes.

A score drop shows sensitivity to this intervention, but the square also covers
image pixels. The clean model is a control for that perturbation. Avoid claiming
the model learned no clothing features. One seed and one wrong-shade mapping
limit generality. A small effect is a valid result, not a reason to tune on test data.

Remaining coursework: interpret the completed local experiment, write the
handout's research-style report, and prepare its presentation (suggested 10–15
slides, approximately 15–20 minutes). No tasks are assigned to teammates.

Keep the Python code, tests, `requirements.txt`, reference notebook, handout,
and `RESULTS.md` in Git.
The compact evidence snapshot in `reports/mac-run1/` is also eligible for Git;
it contains measured tables, predictions, plots, and records, but no model weights.
Downloaded datasets, local environments, weights, and raw `results/` folders are
ignored. Back up complete run folders separately and record which run produced
each figure in your report.

## Small fixes

- `No module named data/model`: upload all three files together and repeat `%cd`.
- Missing dependency in Colab: upload `requirements.txt`, run
  `%pip install -r requirements.txt`, then restart the runtime if requested and
  repeat imports. Locally, use `python -m pip install -r requirements.txt` inside
  the project environment.
- GPU unavailable: choose a GPU runtime, or use `--device cpu` for the smoke check.
  Keep tuning and paired training on the same device/software.
- Missing `selection.json` or `protocol.json`: finish the preceding stage.
- Existing stage folder: preserve it by renaming it before retrying.

Optional local checks: `python -m unittest discover -s tests -v`.
These use synthetic data, not scientific results.
