import os
import json
import math
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # Registers the custom op 'CaseFoldUTF8'
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, hamming_loss,
    average_precision_score
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
    low = (sentence_length // bin_size) * bin_size
    high = low + bin_size - 1
    return f"{low}-{high}"

# -----------------------------
# Utility: Get sample from raw data by sentence string
# -----------------------------
def get_sample_by_sentence(sentence, raw_data):
    for sample in raw_data:
        if sample.get("sentence", "") == sentence:
            return sample
    return None

# -----------------------------
# PIL-Based Highlighted Text Image Generator
# -----------------------------
def get_highlighted_text_image(text, entities):
    """
    Creates and returns a PIL Image with the given text rendered word-by-word.
    If a word exactly matches an entity (using a fallback on available fields),
    that word is drawn over a pastel rounded rectangle with the category label below.
    """
    # Define image dimensions
    img_width = 1000
    img_height = 300
    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)
    
    # Load fonts (try Arial; if not available, use default)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Define pastel colors for categories
    colors = {
        "People Name": "#FFB6C1",   # Light Pink
        "Card Number": "#FFD700",   # Gold
        "Account Number": "#FFA07A",   # Light Salmon
        "Social Security Number": "#FA8072",   # Salmon
        "Government ID Number": "#FF8C00",   # Dark Orange
        "Date of Birth": "#98FB98",   # Pale Green
        "Password": "#8A2BE2",   # Blue Violet
        "Tax ID Number": "#DC143C",   # Crimson
        "Phone Number": "#32CD32",   # Lime Green
        "Residential Address": "#4682B4",   # Steel Blue
        "Email Address": "#87CEEB",   # Sky Blue
        "IP Number": "#20B2AA",   # Light Sea Green
        "Passport": "#A020F0",   # Purple
        "Driver License": "#D2691E"   # Chocolate
    }
    
    # Build a list of entity tuples: (entity_text, category)
    # If the entity dictionary has "start", use it; otherwise, try "word" or "text".
    entity_tuples = []
    if entities and isinstance(entities[0], dict):
        if "start" in entities[0]:
            entity_tuples = [(text[ent["start"]:ent["end"]], ent["category"]) for ent in entities]
        elif "word" in entities[0]:
            entity_tuples = [(ent["word"], ent["category"]) for ent in entities]
        elif "text" in entities[0]:
            entity_tuples = [(ent["text"], ent["category"]) for ent in entities]
        else:
            entity_tuples = []
    else:
        # Assume entities is already a list of tuples
        entity_tuples = entities

    # Split text into words
    words = text.split()
    x, y = 20, 50  # Starting position
    line_height = 40  # Space between lines
    
    for word in words:
        # Check for an exact match with one of the entity words
        category = next((cat for w, cat in entity_tuples if w == word), None)
        
        # Get word size using textbbox
        word_bbox = draw.textbbox((0, 0), word, font=font)
        word_width = word_bbox[2] - word_bbox[0]
        word_height = word_bbox[3] - word_bbox[1]
        
        if category is not None:
            # Draw a rounded rectangle behind the word
            rect_x1, rect_y1 = x - 5, y - 5
            rect_x2, rect_y2 = x + word_width + 5, y + word_height + 5
            draw.rounded_rectangle(
                [(rect_x1, rect_y1), (rect_x2, rect_y2)],
                fill=colors.get(category, "#FFB6C1"),
                radius=8
            )
            draw.text((x, y), word, fill="black", font=font)
            # Draw the category label below the word in a smaller gray font
            cat_bbox = draw.textbbox((0, 0), category, font=font_small)
            draw.text((x, y + word_height + 5), category, fill="gray", font=font_small)
        else:
            draw.text((x, y), word, fill="black", font=font)
        
        x += word_width + 20
        # Wrap text to next line if needed
        if x > img_width - 100:
            x = 20
            y += line_height + 15
            
    return img

# -----------------------------
# Data Loading and Preprocessing
# -----------------------------
def load_data(data_file):
    print("[INFO] Loading synthetic data from:", data_file)
    with open(data_file, "r", encoding="utf-8") as f:
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
    return train_dataset, val_dataset, sentences_train, sentences_val, labels_train, labels_val

# -----------------------------
# Model Building
# -----------------------------
def build_model(num_categories):
    print("[INFO] Building the BERT + CNN model...")
    preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text_input")
    print("[INFO] Loading BERT preprocessing layer from TF Hub...")
    preprocessing_layer = hub.KerasLayer(preprocess_url, name="preprocessing")
    encoder_inputs = preprocessing_layer(text_input)
    print("[INFO] Loading BERT encoder layer from TF Hub...")
    bert_encoder = hub.KerasLayer(encoder_url, trainable=True, name="BERT_encoder")
    bert_outputs = bert_encoder(encoder_inputs)
    sequence_output = bert_outputs["sequence_output"]
    
    conv = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation="relu", name="conv1d")(sequence_output)
    pool = tf.keras.layers.GlobalMaxPooling1D(name="global_max_pool")(conv)
    dropout = tf.keras.layers.Dropout(0.1, name="dropout")(pool)
    output = tf.keras.layers.Dense(num_categories, activation="sigmoid", name="output")(dropout)
    
    model = tf.keras.Model(inputs=text_input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-6),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )
    print("[INFO] Model built successfully. Model summary:")
    model.summary(print_fn=lambda x: print("[MODEL] " + x))
    return model

# -----------------------------
# Training
# -----------------------------
def train_model(model, train_dataset, val_dataset, epochs=25):
    print(f"[INFO] Starting training for {epochs} epochs...")
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
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
    total_rows = 7 + num_conf_rows

    fig = plt.figure(figsize=(20, 6 * total_rows))
    gs = gridspec.GridSpec(total_rows, 4, figure=fig)

    # Row 0: Training History
    metrics_list = ["loss", "accuracy", "precision", "recall"]
    for i, metric in enumerate(metrics_list):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(history.history[metric], marker='o', label=f"Train {metric.capitalize()}")
        ax.plot(history.history["val_" + metric], marker='x', label=f"Val {metric.capitalize()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.legend()

    # Row 1: Category Distribution
    ax_dist = fig.add_subplot(gs[1, :])
    counts = np.sum(labels, axis=0)
    sns.barplot(x=categories, y=counts, ax=ax_dist)
    ax_dist.set_xticklabels(ax_dist.get_xticklabels(), rotation=45, ha="right")
    ax_dist.set_xlabel("Category")
    ax_dist.set_ylabel("Count")
    ax_dist.set_title("Sample Count per Category")

    # Row 2: Entity Category vs. Binned Sentence Length Heatmap
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
    df.index = pd.CategoricalIndex(df.index, categories=sorted(df.index, key=lambda x: int(x.split('-')[0])), ordered=True)
    df = df.sort_index()
    ax_heat = fig.add_subplot(gs[2, :])
    sns.heatmap(df, annot=True, fmt=".0f", cmap="viridis", ax=ax_heat)
    ax_heat.set_xlabel("Entity Category")
    ax_heat.set_ylabel("Sentence Length Bins (words)")
    ax_heat.set_title("Heatmap: Entity Category vs. Binned Sentence Length")
    
    # Row 3: Per-Category Metrics Table
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

    # Row 4: Training/Validation Dataset & Category Balance Bar Chart
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

    # Row 5: Inference Performance Bar Chart
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
        time_ms = (end - start) * 1000
        pred_bin = (pred >= threshold).astype(int)
        cat_count = int(np.sum(pred_bin))
        bin_times.setdefault(bin_label, []).append(time_ms)
        bin_cat_counts.setdefault(bin_label, []).append(cat_count)
    bins_sorted = sorted(bin_times.keys(), key=lambda x: int(x.split('-')[0]))
    avg_times = [np.mean(bin_times[b]) for b in bins_sorted]
    avg_cats = [np.mean(bin_cat_counts[b]) for b in bins_sorted]
    ax_perf = fig.add_subplot(gs[5, :])
    width_bar = 0.35
    x_bins = np.arange(len(bins_sorted))
    ax_perf.bar(x_bins - width_bar/2, avg_times, width_bar, label="Avg Inference Time (ms)")
    ax_perf.bar(x_bins + width_bar/2, avg_cats, width_bar, label="Avg Predicted Category Count")
    ax_perf.set_xticks(x_bins)
    ax_perf.set_xticklabels(bins_sorted)
    ax_perf.set_xlabel("Sentence Length Bin")
    ax_perf.set_title("Inference Performance Metrics by Sentence Length Bin")
    ax_perf.legend()

    # Row 6: Embed the PIL-generated Highlighted Text Image into the Master JPEG
    ax_highlight = fig.add_subplot(gs[6, :])
    ax_highlight.axis("off")
    # Look for a sample sentence in the validation set that has entity info.
    sample_sentence = None
    sample_entities = None
    for sent in val_sentences:
        sample_info = get_sample_by_sentence(sent, raw_data)
        if sample_info and "entities" in sample_info and sample_info["entities"]:
            sample_sentence = sent
            sample_entities = sample_info["entities"]
            break
    if sample_sentence is not None:
        pil_img = get_highlighted_text_image(sample_sentence, sample_entities)
        img_array = np.array(pil_img)
        ax_highlight.imshow(img_array)
        ax_highlight.set_title("Highlighted Sample Text", fontsize=12, fontweight="bold")
    else:
        ax_highlight.text(0.5, 0.5, "No sample with entity info", horizontalalignment="center", verticalalignment="center")
    
    # Rows 7+: Confusion Matrices for Each Category
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
    
    plt.subplots_adjust(hspace=1.0, wspace=0.5)
    combined_path = os.path.join(output_dir, "combined_post_training_charts.png")
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
    history = train_model(model, train_dataset, val_dataset, epochs=25)
    
    model.save(model_save_path)
    print(f"[INFO] Final model saved to {model_save_path}")
    
    evaluate_model(model, val_dataset, categories, OUTPUT_DIR, history, sentences, labels,
                   labels_train, labels_val, sentences_val, raw_data, threshold=0.5)

if __name__ == "__main__":
    main()
