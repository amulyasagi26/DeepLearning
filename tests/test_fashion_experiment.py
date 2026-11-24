"""Synthetic checks of the experiment mechanics; these are not project results."""

import contextlib
import csv
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import data
import experiment
from model import SmallCNN


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class FashionExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)  # Tiny synthetic batches need no large thread pool.

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_marker_views_preserve_labels_and_source_pixels(self):
        images = torch.rand(10, 1, 28, 28)
        original = images.clone()
        base = TensorDataset(images, torch.arange(10))
        for condition in data.CONDITIONS:
            view = data.MarkerDataset(base, condition)
            for label in range(10):
                with self.subTest(condition=condition, label=label):
                    image, actual_label = view[label]
                    self.assertEqual(int(actual_label), label)
                    self.assertNotEqual(image.data_ptr(), images[label].data_ptr())
                    if condition == "none":
                        self.assertTrue(torch.equal(image, original[label]))
                    else:
                        shade_class = label if condition == "matching" else (label + 1) % 10
                        self.assertTrue(torch.all(image[:, :4, :4] == data.SHADES[shade_class]))
                        self.assertTrue(torch.equal(image[:, 4:, :], original[label, :, 4:, :]))
                        self.assertTrue(torch.equal(image[:, :4, 4:], original[label, :, :4, 4:]))
        self.assertTrue(torch.equal(images, original))
        self.assertEqual(data.MarkerDataset(base, "wrong")[9][0][0, 0, 0], data.SHADES[0])

    def test_training_split_is_exact_disjoint_and_reproducible(self):
        # A stub avoids importing torchvision or downloading the real dataset.
        source = TensorDataset(torch.arange(60_000), torch.arange(60_000) % 10)
        factory = Mock(return_value=source)
        torchvision = types.ModuleType("torchvision")
        torchvision.datasets = types.SimpleNamespace(FashionMNIST=factory)
        torchvision.transforms = types.SimpleNamespace(ToTensor=lambda: "tensor-transform")
        with patch.dict(sys.modules, {"torchvision": torchvision}):
            train, validation = data.load_training_data("unused", seed=42)
            repeated, _ = data.load_training_data("unused", seed=42)
            changed, _ = data.load_training_data("unused", seed=43)
        self.assertEqual((len(train), len(validation)), (54_000, 6_000))
        self.assertFalse(set(train.indices) & set(validation.indices))
        self.assertEqual(set(train.indices) | set(validation.indices), set(range(60_000)))
        self.assertEqual(train.indices, repeated.indices)
        self.assertNotEqual(train.indices, changed.indices)
        self.assertTrue(all(call.kwargs["train"] for call in factory.call_args_list))

    def test_cnn_preserves_notebook_architecture_and_disables_dropout_for_evaluation(self):
        model = SmallCNN(dropout=0.25)
        convolutions = [layer for layer in model.modules() if isinstance(layer, nn.Conv2d)]
        self.assertEqual([(c.in_channels, c.out_channels, c.kernel_size, c.padding)
                          for c in convolutions], [(1, 32, (3, 3), (1, 1)), (32, 64, (3, 3), (1, 1))])
        linear = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in linear],
                         [(64 * 7 * 7, 128), (128, 10)])
        dropout = next(layer for layer in model.modules() if isinstance(layer, nn.Dropout))
        self.assertEqual(dropout.p, 0.25)
        model.eval()
        images = torch.rand(3, 1, 28, 28)
        with torch.no_grad():
            scores = model(images)
            self.assertEqual(scores.shape, (3, 10))
            self.assertTrue(torch.equal(scores, model(images)))
        self.assertFalse(dropout.training)

    def test_validation_keeps_weights_and_training_updates_them(self):
        model = SmallCNN()
        loader = DataLoader(TensorDataset(torch.rand(4, 1, 28, 28), torch.arange(4)), batch_size=2)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        loss, accuracy = experiment.run_one_epoch(model, loader, nn.CrossEntropyLoss(), "cpu")
        self.assertGreater(loss, 0)
        self.assertTrue(0 <= accuracy <= 1)
        self.assertFalse(model.training)
        self.assertTrue(all(torch.equal(before[name], value) for name, value in model.state_dict().items()))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        experiment.run_one_epoch(model, loader, nn.CrossEntropyLoss(), "cpu", optimizer)
        self.assertTrue(any(not torch.equal(before[name], value)
                            for name, value in model.state_dict().items()))

    def test_staged_workflow_uses_test_data_only_for_final_evaluation(self):
        train = TensorDataset(torch.rand(128, 1, 28, 28), torch.arange(128) % 10)
        validation = TensorDataset(torch.rand(32, 1, 28, 28), torch.arange(32) % 10)
        true_labels = [9, 0, 3, 6, 1, 8, 4, 2, 7, 5]
        test = TensorDataset(torch.rand(10, 1, 28, 28), torch.tensor(true_labels))
        load_test = Mock(return_value=test)
        settings = {**experiment.DEFAULT_SETTINGS, "epochs": 1, "batch_size": 32}
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()), patch.multiple(
            experiment, load_training_data=Mock(return_value=(train, validation)),
            load_test_data=load_test, DEFAULT_SETTINGS=settings,
            LEARNING_RATES=(0.001,), DROPOUT_RATES=(0.25,),
        ):
            output = Path(temporary)
            for stage in (experiment.smoke, experiment.tune, experiment.train):
                stage("unused", output, torch.device("cpu"))
                load_test.assert_not_called()
            with patch.object(experiment, "MARKER_SIZE", 3):
                with self.assertRaisesRegex(ValueError, "Marker settings changed"):
                    experiment.evaluate("unused", output, torch.device("cpu"))
            load_test.assert_not_called()
            experiment.evaluate("unused", output, torch.device("cpu"))
            load_test.assert_called_once_with("unused")
            evaluation = output / "evaluation"
            results = read_csv(evaluation / "test_results.csv")
            self.assertEqual(len(results), 6)
            self.assertEqual({(row["model"], row["condition"]) for row in results},
                             {(name, condition) for name in ("clean", "marked") for condition in data.CONDITIONS})
            self.assertTrue(all(int(row["images"]) == len(test) for row in results))
            for name in ("clean", "marked"):
                predictions = read_csv(evaluation / f"predictions_{name}.csv")
                self.assertEqual([int(row["test_index"]) for row in predictions], list(range(10)))
                self.assertEqual([int(row["true_label"]) for row in predictions], true_labels)
                self.assertEqual([int(row["wrong_marker_class"]) for row in predictions],
                                 [(label + 1) % 10 for label in true_labels])
                self.assertTrue((output / "final" / name / "learning_curves.png").is_file())
                for condition in data.CONDITIONS:
                    confusion = read_csv(evaluation / f"confusion_{name}_{condition}.csv")
                    matrix = [[int(row[label]) for label in data.CLASS_NAMES] for row in confusion]
                    self.assertEqual(sum(map(sum, matrix)), len(test))
                    self.assertEqual([sum(row) for row in matrix], [true_labels.count(i) for i in range(10)])
                    for row in predictions:
                        self.assertEqual(matrix[int(row["true_label"])][int(row[condition])], 1)
                    accuracy = sum(int(row[condition]) == int(row["true_label"]) for row in predictions) / len(test)
                    result = next(row for row in results if row["model"] == name and row["condition"] == condition)
                    self.assertAlmostEqual(float(result["accuracy"]), accuracy)
                    self.assertTrue((evaluation / f"confusion_{name}_{condition}.png").is_file())
            self.assertTrue((evaluation / "test_accuracy.png").is_file())
            self.assertEqual(len(read_csv(evaluation / "marker_effects.csv")), 2)
            with self.assertRaises(FileExistsError):
                experiment.evaluate("unused", output, torch.device("cpu"))
            load_test.assert_called_once()


if __name__ == "__main__":
    unittest.main()
