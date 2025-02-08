import os
import json
import re
import math
import time
import collections
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # Registers the custom op 'CaseFoldUTF8'
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score, hamming_loss, average_precision_score
)

# -----------------------------
# Global Output Directory
# -----------------------------
OUTPUT_DIR = r"O:\master_model_collection\redact\v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Helper Function for Sentence Length Bins
# -----------------------------
def get_bin_label(sentence_length, bin_size=5):
    """
    Returns a bin label (as a string) for a given sentence length.
    For example, a sentence with 12 words with bin_size=5 will fall into "10-14".
    """
    low = (sentence_length // bin_size) * bin_size
    high = low + bin_size - 1
    return f"{low}-{high}"

# -----------------------------
# Data Loading and Preprocessing
# -----------------------------
def load_data(data_file):
    print("[INFO] Loading synthetic data from:", data_file)
    with open(data_file, "r") as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} samples.")
    return data

def preprocess_data(data, categories):
    print("[INFO] Preprocessing data...")
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    num_categories = len(categories)
    sentences = []
    labels = []
    for sample in data:
        sentence = sample["sentence"]
        # Create a multi-hot vector for each sentence
        label_vector = np.zeros(num_categories, dtype=np.float32)
        for entity in sample.get("entities", []):
            cat = entity.get("category")
            if cat in cat_to_idx:
                label_vector[cat_to_idx[cat]] = 1.0
        sentences.append(sentence)
        labels.append(label_vector)
    sentences = np.array(sentences)
    labels = np.array(labels)
    print(f"[INFO] Preprocessing complete. Total sentences: {len(sentences)}. Labels shape: {labels.shape}")
    return sentences, labels

def create_datasets(sentences, labels, batch_size=16):
    print("[INFO] Splitting data into training and validation sets (80/20 split)...")
    sentences_train, sentences_val, labels_train, labels_val = train_test_split(
        sentences, labels, test_size=0.20, random_state=42
    )
    print(f"[INFO] Training samples: {len(sentences_train)}, Validation samples: {len(sentences_val)}")
    
    train_dataset = tf.data.Dataset.from_tensor_slices((sentences_train, labels_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(sentences_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((sentences_val, labels_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    print("[INFO] tf.data.Datasets created.")
    # Also return the raw training/validation arrays for later use in tables/bar charts
    return train_dataset, val_dataset, sentences_train, sentences_val, labels_train, labels_val

# -----------------------------
# Model Building
# -----------------------------
def build_model(num_categories):
    print("[INFO] Building the BERT + CNN model...")
    preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"

    # Define input layer for raw text
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text_input")
    
    # BERT Preprocessing
    print("[INFO] Loading BERT preprocessing layer from TF Hub...")
    preprocessing_layer = hub.KerasLayer(preprocess_url, name="preprocessing")
    encoder_inputs = preprocessing_layer(text_input)
    
    # BERT Encoder
    print("[INFO] Loading BERT encoder layer from TF Hub...")
    bert_encoder = hub.KerasLayer(encoder_url, trainable=True, name="BERT_encoder")
    bert_outputs = bert_encoder(encoder_inputs)
    
    # Use the sequence output for CNN
    sequence_output = bert_outputs["sequence_output"]
    
    # CNN layers
    conv = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation="relu", name="conv1d")(sequence_output)
    pool = tf.keras.layers.GlobalMaxPooling1D(name="global_max_pool")(conv)
    dropout = tf.keras.layers.Dropout(0.1, name="dropout")(pool)
    output = tf.keras.layers.Dense(num_categories, activation="sigmoid", name="output")(dropout)
    
    model = tf.keras.Model(inputs=text_input, outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-6),
        loss="binary_crossentropy",
        metrics=[
            "accuracy", 
            tf.keras.metrics.Precision(name="precision"), 
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    
    print("[INFO] Model built successfully. Model summary:")
    model.summary(print_fn=lambda x: print("[MODEL] " + x))
    return model

# -----------------------------
# Training
# -----------------------------
def train_model(model, train_dataset, val_dataset, epochs=25):
    print(f"[INFO] Starting training for {epochs} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )
    print("[INFO] Training complete.")
    return history

# -----------------------------
# Combined Post-Training Charts and Tables
# -----------------------------
def plot_combined_post_training_charts(history, sentences, labels, y_true, y_pred, categories,
                                       train_labels, val_labels, val_sentences, model, raw_data,
                                       output_dir, threshold=0.5, bin_size=5):
    print("[INFO] Creating combined post-training charts...")

    y_pred_binary = (y_pred >= threshold).astype(int)
    num_categories = len(categories)
    num_conf_rows = math.ceil(num_categories / 4)
    # Fixed rows:
    # Row 0: Training History
    # Row 1: Category Distribution
    # Row 2: Binned Sentence Length Heatmap
    # Row 3: Per-Category Metrics Table
    # Row 4: Training/Validation Dataset & Category Balance Bar Chart (updated)
    # Row 5: Inference Performance Bar Chart
    # Row 6: Sample Predictions Visualization
    # Rows 7+ for Confusion Matrices.
    total_rows = 7 + num_conf_rows

    fig = plt.figure(figsize=(20, 5 * total_rows))
    gs = gridspec.GridSpec(total_rows, 4, figure=fig)

    # ---- Row 0: Training History ----
    metrics_list = ["loss", "accuracy", "precision", "recall"]
    for i, metric in enumerate(metrics_list):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(history.history[metric], marker='o', label=f"Train {metric.capitalize()}")
        ax.plot(history.history["val_" + metric], marker='x', label=f"Val {metric.capitalize()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.legend()

    # ---- Row 1: Category Distribution ----
    ax_dist = fig.add_subplot(gs[1, :])
    counts = np.sum(labels, axis=0)
    sns.barplot(x=categories, y=counts, ax=ax_dist)
    # Use existing tick labels to avoid warning.
    ax_dist.set_xticklabels(ax_dist.get_xticklabels(), rotation=45, ha="right")
    ax_dist.set_xlabel("Category")
    ax_dist.set_ylabel("Count")
    ax_dist.set_title("Sample Count per Category")

    # ---- Row 2: Entity Category vs. Binned Sentence Length Heatmap ----
    counts_dict = {}
    for sample in raw_data:
        sentence = sample.get("sentence", "")
        words = sentence.split()
        length = len(words)
        bin_label = get_bin_label(length, bin_size)
        if bin_label not in counts_dict:
            counts_dict[bin_label] = {}
        for entity in sample.get("entities", []):
            category = entity.get("category", "Unknown")
            counts_dict[bin_label][category] = counts_dict[bin_label].get(category, 0) + 1
    df = pd.DataFrame.from_dict(counts_dict, orient="index").fillna(0)
    df.index = pd.CategoricalIndex(
        df.index,
        categories=sorted(df.index, key=lambda x: int(x.split('-')[0])),
        ordered=True
    )
    df = df.sort_index()
    ax_heat = fig.add_subplot(gs[2, :])
    sns.heatmap(df, annot=True, fmt=".0f", cmap="viridis", ax=ax_heat)
    ax_heat.set_xlabel("Entity Category")
    ax_heat.set_ylabel("Sentence Length Bins (words)")
    ax_heat.set_title("Heatmap: Entity Category vs. Binned Sentence Length")
    
    # ---- Row 3: Per-Category Metrics Table ----
    per_cat_data = []
    for i, cat in enumerate(categories):
        cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])
        if cm.shape == (1, 1):
            cm = np.array([[cm[0, 0], 0], [0, 0]])
        elif cm.shape == (1, 2):
            cm = np.vstack([cm, [0, 0]])
        elif cm.shape == (2, 1):
            cm = np.hstack([cm, [[0], [0]]])
        if cm.size == 4:
            TN, FP, FN, TP = cm.ravel()
        else:
            TN = FP = FN = TP = 0
        total = TN + FP + FN + TP
        accuracy = (TP + TN) / total if total > 0 else 0
        precision_val = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall_val = TP / (TP + FN) if (TP + FN) > 0 else 0
        per_cat_data.append([cat, f"{precision_val:.2f}", f"{recall_val:.2f}", f"{accuracy:.2f}"])
    ax_table1 = fig.add_subplot(gs[3, :])
    ax_table1.axis('tight')
    ax_table1.axis('off')
    col_labels = ["Category", "Precision", "Recall", "Accuracy"]
    table1 = ax_table1.table(cellText=per_cat_data, colLabels=col_labels, loc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    ax_table1.set_title("Per-Category Metrics", fontweight="bold")

    # ---- Row 4: Training/Validation Dataset & Category Balance Bar Chart ----
    # Compute counts for training and validation splits.
    train_counts = np.sum(train_labels, axis=0)
    val_counts = np.sum(val_labels, axis=0)
    ax_bar = fig.add_subplot(gs[4, :])
    x = np.arange(len(categories))
    width = 0.35
    ax_bar.bar(x - width/2, train_counts, width, label="Training", color="blue")
    ax_bar.bar(x + width/2, val_counts, width, label="Validation", color="orange")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(categories, rotation=45, ha="right")
    ax_bar.set_xlabel("Category")
    ax_bar.set_ylabel("Count")
    ax_bar.set_title("Training/Validation Dataset & Category Balance")
    ax_bar.legend()

    # ---- Row 5: Inference Performance Bar Chart ----
    bin_times = {}
    bin_cat_counts = {}
    for sentence in val_sentences:
        if sentence is None:
            continue
        sentence_str = str(sentence)
        words = sentence_str.split()
        length = len(words)
        bin_label = get_bin_label(length, bin_size)
        start = time.perf_counter()
        pred = model.predict([sentence_str])
        end = time.perf_counter()
        time_ms = (end - start) * 1000  # in milliseconds
        pred_bin = (pred >= threshold).astype(int)
        cat_count = int(np.sum(pred_bin))
        bin_times.setdefault(bin_label, []).append(time_ms)
        bin_cat_counts.setdefault(bin_label, []).append(cat_count)
    bins_sorted = sorted(bin_times.keys(), key=lambda x: int(x.split('-')[0]))
    avg_times = [np.mean(bin_times[b]) for b in bins_sorted]
    avg_cats = [np.mean(bin_cat_counts[b]) for b in bins_sorted]
    
    ax_perf = fig.add_subplot(gs[5, :])
    width = 0.35
    x_bins = np.arange(len(bins_sorted))
    ax_perf.bar(x_bins - width/2, avg_times, width, label="Avg Inference Time (ms)")
    ax_perf.bar(x_bins + width/2, avg_cats, width, label="Avg Predicted Category Count")
    ax_perf.set_xticks(x_bins)
    ax_perf.set_xticklabels(bins_sorted)
    ax_perf.set_xlabel("Sentence Length Bin")
    ax_perf.set_title("Inference Performance Metrics by Sentence Length Bin")
    ax_perf.legend()

    # ---- Row 6: Sample Predictions Visualization ----
    sentence_to_sample = {sample.get("sentence", ""): sample for sample in raw_data if "sentence" in sample}
    num_val = len(val_sentences)
    sample_indices = random.sample(range(num_val), min(5, num_val))
    ax_sample = fig.add_subplot(gs[6, :])
    ax_sample.axis("off")
    current_y = 1.0  # normalized coordinates
    for idx in sample_indices:
        sentence = val_sentences[idx]
        sentence_disp = sentence if len(sentence) < 120 else sentence[:117] + "..."
        sample_info = sentence_to_sample.get(sentence, None)
        true_entities = []
        if sample_info is not None:
            for entity in sample_info.get("entities", []):
                ent_text = entity.get("text", entity.get("category", ""))
                true_entities.append(f"{ent_text} ({entity.get('category', '')})")
        true_entities_str = ", ".join(true_entities) if true_entities else "N/A"
        pred_bin = y_pred_binary[idx]
        predicted_cats = [categories[i] for i, flag in enumerate(pred_bin) if flag == 1]
        
        ax_sample.text(0.01, current_y, f"Sentence: {sentence_disp}", fontsize=10, color="black", wrap=True)
        current_y -= 0.05
        ax_sample.text(0.02, current_y, f"True Entities: {true_entities_str}", fontsize=10, color="blue", wrap=True)
        current_y -= 0.05
        x_text = 0.02
        ax_sample.text(x_text, current_y, "Predicted Categories: ", fontsize=10, color="black")
        x_text += 0.15
        for cat in predicted_cats:
            is_correct = False
            if sample_info is not None:
                for entity in sample_info.get("entities", []):
                    if entity.get("category", "") == cat:
                        is_correct = True
                        break
            color = "green" if is_correct else "red"
            ax_sample.text(x_text, current_y, cat + " ", fontsize=10, color=color)
            x_text += 0.1
        current_y -= 0.1
        ax_sample.axhline(y=current_y + 0.05, color="gray", linewidth=0.5)
        current_y -= 0.05

    # ---- Rows 7 and onward: Confusion Matrices for Each Category ----
    for idx, cat in enumerate(categories):
        row_idx = 7 + idx // 4
        col_idx = idx % 4
        ax_cm = fig.add_subplot(gs[row_idx, col_idx])
        cm = confusion_matrix(y_true[:, idx], y_pred_binary[:, idx])
        if cm.shape == (1, 1):
            cm = np.array([[cm[0, 0], 0], [0, 0]])
        elif cm.shape == (1, 2):
            cm = np.vstack([cm, [0, 0]])
        elif cm.shape == (2, 1):
            cm = np.hstack([cm, [[0], [0]]])
        if cm.size == 4:
            TN, FP, FN, TP = cm.ravel()
        else:
            TN = FP = FN = TP = 0
        total = TN + FP + FN + TP
        acc = (TP + TN) / total if total > 0 else 0
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        ax_cm.set_title(f"{cat}")
        ax_cm.text(0.5, -0.3, f"Acc: {acc:.2f}\nPrec: {prec:.2f}\nRec: {rec:.2f}",
                   fontsize=10, ha="center", transform=ax_cm.transAxes)

    total_conf_plots = num_conf_rows * 4
    for extra in range(num_categories, total_conf_plots):
        row_idx = 7 + extra // 4
        col_idx = extra % 4
        ax_empty = fig.add_subplot(gs[row_idx, col_idx])
        ax_empty.axis('off')

    combined_path = os.path.join(output_dir, "combined_post_training_charts.png")
    plt.tight_layout()
    plt.savefig(combined_path)
    plt.close()
    print(f"[INFO] Combined post-training charts saved to {combined_path}")

# -----------------------------
# Evaluation with Detailed Metrics & Combined Charts
# -----------------------------
def evaluate_model(model, val_dataset, categories, output_dir, history, sentences, labels,
                   train_labels, val_labels, val_sentences, raw_data, threshold=0.5):
    print("[INFO] Evaluating the model on the validation set...")
    eval_results = model.evaluate(val_dataset)
    print("\n[RESULTS] Overall Evaluation on Validation Set:")
    print(f"         Loss:      {eval_results[0]:.4f}")
    print(f"         Accuracy:  {eval_results[1]:.4f}")
    print(f"         Precision: {eval_results[2]:.4f}")
    print(f"         Recall:    {eval_results[3]:.4f}")
    
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
    metrics_path = os.path.join(output_dir, "performance_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(performance_metrics, f, indent=4)
    print(f"[INFO] Performance metrics saved to {metrics_path}")
    
    plot_combined_post_training_charts(history, sentences, labels, y_true, y_pred, categories,
                                       train_labels, val_labels, val_sentences, model, raw_data,
                                       output_dir, threshold)

# -----------------------------
# Main Execution
# -----------------------------
def main():
    data_file = r"O:\master_data_collection\redact\synthetic_data.json"
    model_save_path = os.path.join(OUTPUT_DIR, "final_model.h5")
    labels_save_path = os.path.join(OUTPUT_DIR, "labels.json")
    
    categories = [
        "People Name", "Card Number", "Account Number", "Social Security Number",
        "Government ID Number", "Date of Birth", "Password", "Tax ID Number",
        "Phone Number", "Residential Address", "Email Address", "IP Number",
        "Passport", "Driver License"
    ]
    
    raw_data = load_data(data_file)
    sentences, labels = preprocess_data(raw_data, categories)
    train_dataset, val_dataset, sentences_train, sentences_val, labels_train, labels_val = create_datasets(sentences, labels, batch_size=16)
    
    with open(labels_save_path, "w") as f:
        json.dump(categories, f, indent=4)
    print(f"[INFO] Labels saved to {labels_save_path}")
    
    model = build_model(num_categories=len(categories))
    history = train_model(model, train_dataset, val_dataset, epochs=1)
    
    model.save(model_save_path)
    print(f"[INFO] Final model saved to {model_save_path}")
    
    evaluate_model(model, val_dataset, categories, OUTPUT_DIR, history, sentences, labels,
                   labels_train, labels_val, sentences_val, raw_data, threshold=0.5)

if __name__ == "__main__":
    main()
