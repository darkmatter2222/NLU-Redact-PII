import os
import json
import math
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # Registers custom op 'CaseFoldUTF8'
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw, ImageFont

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, hamming_loss,
    average_precision_score
)


class ModelEvaluator:
    def __init__(self, root_dir, model_save_path, labels_save_path, data_file, eval_dir=None):
        """
        Initialize the evaluator with file paths and directories.
        :param root_dir: Root directory for the project.
        :param model_save_path: Path to the saved model file.
        :param labels_save_path: Path to the JSON file with category labels.
        :param data_file: Path to the JSON file containing the raw data.
        :param eval_dir: (Optional) Directory where evaluation outputs will be saved.
        """
        self.root_dir = root_dir
        self.model_save_path = model_save_path
        self.labels_save_path = labels_save_path
        self.data_file = data_file
        self.eval_dir = eval_dir or os.path.join(root_dir, "EVAL")
        os.makedirs(self.eval_dir, exist_ok=True)

    def load_data(self):
        """Load data from the specified JSON file."""
        print("[INFO] Loading data from:", self.data_file)
        with open(self.data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[INFO] Loaded {len(data)} samples.")
        return data

    def preprocess_data(self, data, categories):
        """
        Preprocess the raw data to produce numpy arrays of sentences and label vectors.
        :param data: List of raw data samples.
        :param categories: List of category labels.
        :return: Tuple (sentences, labels) as numpy arrays.
        """
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}
        num_categories = len(categories)
        sentences = []
        labels = []
        for sample in data:
            sentence = sample["sentence"]
            label_vector = np.zeros(num_categories, dtype=np.float32)
            for entity in sample.get("entities", []):
                cat = entity.get("category")
                if cat in cat_to_idx:
                    label_vector[cat_to_idx[cat]] = 1.0
            sentences.append(sentence)
            labels.append(label_vector)
        return np.array(sentences), np.array(labels)

    def create_datasets(self, sentences, labels, batch_size=16):
        """
        Split the data into training and validation tf.data.Datasets.
        :param sentences: Array of sentence strings.
        :param labels: Array of label vectors.
        :param batch_size: Batch size for the datasets.
        :return: Tuple of (train_dataset, val_dataset, sentences_train, sentences_val, labels_train, labels_val)
        """
        sentences_train, sentences_val, labels_train, labels_val = train_test_split(
            sentences, labels, test_size=0.20, random_state=42
        )
        train_dataset = tf.data.Dataset.from_tensor_slices((sentences_train, labels_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(sentences_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((sentences_val, labels_val))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return train_dataset, val_dataset, sentences_train, sentences_val, labels_train, labels_val

    def evaluate_model(self, model, val_dataset, categories, sentences, labels,
                       train_labels, val_labels, sentences_val, raw_data, threshold=0.5):
        """
        Evaluate the model on the validation dataset, compute detailed metrics,
        save the results to a JSON file, and generate charts using ChartPlotter.
        :param model: Trained TensorFlow model.
        :param val_dataset: Validation tf.data.Dataset.
        :param categories: List of category labels.
        :param sentences: All sentences used (numpy array).
        :param labels: All label vectors.
        :param train_labels: Label vectors for training data.
        :param val_labels: Label vectors for validation data.
        :param sentences_val: Sentences used for validation.
        :param raw_data: The raw data samples.
        :param threshold: Threshold for converting predictions to binary values.
        """
        eval_results = model.evaluate(val_dataset)
        print("\n[RESULTS] Overall Evaluation on Validation Set:")
        print(f"         Loss:      {eval_results[0]:.4f}")
        print(f"         Accuracy:  {eval_results[1]:.4f}")
        print(f"         Precision: {eval_results[2]:.4f}")
        print(f"         Recall:    {eval_results[3]:.4f}")

        # Collect predictions for detailed metrics.
        y_true = []
        y_pred = []
        for texts, batch_labels in val_dataset:
            predictions = model.predict(texts)
            y_true.extend(batch_labels.numpy())
            y_pred.extend(predictions)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_binary = (y_pred >= threshold).astype(int)

        report_dict = classification_report(
            y_true, y_pred_binary, target_names=categories, zero_division=0, output_dict=True
        )
        hl = hamming_loss(y_true, y_pred_binary)
        macro_f1 = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
        avg_precision_macro = average_precision_score(y_true, y_pred, average='macro')
        avg_precision_micro = average_precision_score(y_true, y_pred, average='micro')

        performance_metrics = {
            "overall": {
                "loss": eval_results[0],
                "accuracy": eval_results[1],
                "precision": eval_results[2],
                "recall": eval_results[3],
                "hamming_loss": hl,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "macro_average_precision": avg_precision_macro,
                "micro_average_precision": avg_precision_micro
            },
            "per_category": report_dict
        }
        metrics_path = os.path.join(self.eval_dir, "performance_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(performance_metrics, f, indent=4)
        print(f"[INFO] Performance metrics saved to {metrics_path}")

        # Create a dummy history object since no training history is available.
        dummy_history = {
            "loss": [eval_results[0]],
            "accuracy": [eval_results[1]],
            "precision": [eval_results[2]],
            "recall": [eval_results[3]],
            "val_loss": [eval_results[0]],
            "val_accuracy": [eval_results[1]],
            "val_precision": [eval_results[2]],
            "val_recall": [eval_results[3]]
        }
        class DummyHistory:
            history = dummy_history
        dummy_history_obj = DummyHistory()

        # Use the ChartPlotter class (from charting.py) to generate charts.
        from charting.charting import ChartPlotter
        plotter = ChartPlotter(bin_size=5)
        plotter.plot_combined_post_training_charts(
            dummy_history_obj, sentences, labels, y_true, y_pred, categories,
            train_labels, val_labels, sentences_val, model, raw_data,
            self.eval_dir, threshold
        )

    def run_evaluation(self, threshold=0.5, batch_size=16):
        """
        Run the full evaluation:
          - Loads the saved model.
          - Loads and preprocesses the data.
          - Splits the data into training and validation sets.
          - Evaluates the model and generates charts.
        :param threshold: Threshold for binary predictions.
        :param batch_size: Batch size for creating tf.data.Datasets.
        """
        print("[INFO] Starting evaluation...")

        # Load the saved model.
        print("[INFO] Loading model from:", self.model_save_path)
        model = tf.keras.models.load_model(self.model_save_path, custom_objects={'KerasLayer': hub.KerasLayer})

        # Load the category labels.
        with open(self.labels_save_path, "r") as f:
            categories = json.load(f)

        # Load and preprocess the raw data.
        raw_data = self.load_data()
        sentences, labels = self.preprocess_data(raw_data, categories)
        train_dataset, val_dataset, sentences_train, sentences_val, labels_train, labels_val = self.create_datasets(
            sentences, labels, batch_size=batch_size
        )

        # Evaluate the model and generate charts and metrics.
        self.evaluate_model(
            model, val_dataset, categories, sentences, labels,
            labels_train, labels_val, sentences_val, raw_data, threshold
        )


# -----------------------------
# Command Line Execution
# -----------------------------
if __name__ == "__main__":
    ROOT_DIR = r"O:\master_model_collection\redact\v1"
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "final_model.h5")
    LABELS_SAVE_PATH = os.path.join(ROOT_DIR, "labels.json")
    DATA_FILE = r"O:\master_data_collection\redact\synthetic_data.json"

    evaluator = ModelEvaluator(ROOT_DIR, MODEL_SAVE_PATH, LABELS_SAVE_PATH, DATA_FILE)
    evaluator.run_evaluation(threshold=0.5, batch_size=16)
