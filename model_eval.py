import os
import json
import math
import time
import textwrap
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # Registers the custom op 'CaseFoldUTF8'
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw, ImageFont

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, hamming_loss,
    average_precision_score
)

# -------------------------------------------------------------------
# Directories and File Paths
# -------------------------------------------------------------------
ROOT_DIR = r"O:\master_model_collection\redact\v1"
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "final_model.h5")
LABELS_SAVE_PATH = os.path.join(ROOT_DIR, "labels.json")
DATA_FILE = r"O:\master_data_collection\redact\synthetic_data.json"

# New folder for evaluation outputs
EVAL_DIR = os.path.join(ROOT_DIR, "EVAL")
os.makedirs(EVAL_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Helper Functions (Adapted from your training code)
# -------------------------------------------------------------------
def get_bin_label(sentence_length, bin_size=5):
    low = (sentence_length // bin_size) * bin_size
    high = low + bin_size - 1
    return f"{low}-{high}"

def get_sample_by_sentence(sentence, raw_data):
    for sample in raw_data:
        if sample.get("sentence", "") == sentence:
            return sample
    return None

def convert_entities(entities):
    """
    Converts a list of entity dictionaries into a list of tuples (entity_text, category).
    Now also checks for an "address" key if the category is Residential Address.
    """
    converted = []
    for ent in entities:
        if isinstance(ent, dict):
            if "word" in ent:
                converted.append((ent["word"], ent["category"]))
            elif "text" in ent:
                converted.append((ent["text"], ent["category"]))
            # If the entity is a residential address, check for an "address" key.
            elif ent.get("category") == "Residential Address" and "address" in ent:
                converted.append((ent["address"], ent["category"]))
            else:
                continue
        else:
            converted.append(ent)
    return converted

def get_highlighted_text_image_wrapped(text, entities):
    """
    Returns a PIL Image rendering the text with wrapped lines.
    Any substring matching an entity is highlighted with a pastel-colored rounded rectangle,
    and its category is drawn below the text.
    
    Updates:
      - Increased wrap_width from 80 to 120.
      - Increased line_height from 30 to 40.
      - Increased font sizes (primary from 18 to 24 and secondary from 14 to 18).
    """
    img_width = 1000
    wrap_width = 120  # increased to reduce unwanted wrapping for longer texts (e.g., addresses)
    wrapped_lines = textwrap.wrap(text, width=wrap_width)
    line_height = 40  # increased for bigger font spacing
    img_height = 40 + line_height * len(wrapped_lines)
    
    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 24)  # bigger primary font
        font_small = ImageFont.truetype("arial.ttf", 18)  # bigger secondary font
    except IOError:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Define pastel colors for categories.
    colors = {
        "People Name": "#FFB6C1",
        "Card Number": "#FFD700",
        "Account Number": "#FFA07A",
        "Social Security Number": "#FA8072",
        "Government ID Number": "#FF8C00",
        "Date of Birth": "#98FB98",
        "Password": "#8A2BE2",
        "Tax ID Number": "#DC143C",
        "Phone Number": "#32CD32",
        "Residential Address": "#4682B4",
        "Email Address": "#87CEEB",
        "IP Number": "#20B2AA",
        "Passport": "#A020F0",
        "Driver License": "#D2691E",
        "Organization": "#C0C0C0"
    }
    
    y = 20
    for line in wrapped_lines:
        x = 20
        draw.text((x, y), line, fill="black", font=font)
        # For each entity, check if it appears in this line.
        for entity_text, category in entities:
            index = line.find(entity_text)
            if index != -1:
                prefix = line[:index]
                prefix_bbox = draw.textbbox((0, 0), prefix, font=font)
                x_start = x + (prefix_bbox[2] - prefix_bbox[0])
                entity_bbox = draw.textbbox((0, 0), entity_text, font=font)
                entity_width = entity_bbox[2] - entity_bbox[0]
                entity_height = entity_bbox[3] - entity_bbox[1]
                padding = 2
                rect = [x_start - padding, y - padding, x_start + entity_width + padding, y + entity_height + padding]
                draw.rounded_rectangle(rect, fill=colors.get(category, "#FFB6C1"), radius=5)
                draw.text((x_start, y), entity_text, fill="black", font=font)
                draw.text((x_start, y + entity_height + 2), category, fill="gray", font=font_small)
        y += line_height
    return img

def load_data(data_file):
    print("[INFO] Loading data from:", data_file)
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} samples.")
    return data

def preprocess_data(data, categories):
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

def create_datasets(sentences, labels, batch_size=16):
    sentences_train, sentences_val, labels_train, labels_val = train_test_split(
        sentences, labels, test_size=0.20, random_state=42
    )
    train_dataset = tf.data.Dataset.from_tensor_slices((sentences_train, labels_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(sentences_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((sentences_val, labels_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset, val_dataset, sentences_train, sentences_val, labels_train, labels_val

# -------------------------------------------------------------------
# Updated Plotting Function with an Even Grid for Highlighted Samples
# -------------------------------------------------------------------
def plot_combined_post_training_charts(history, sentences, labels, y_true, y_pred, categories,
                                       train_labels, val_labels, val_sentences, model, raw_data,
                                       output_dir, threshold=0.5, bin_size=5):
    num_categories = len(categories)
    y_pred_binary = (y_pred >= threshold).astype(int)
    num_conf_rows = math.ceil(num_categories / 4)

    # Define number of rows for each section:
    TOP_CHART_ROWS = 6    # Rows 0-5: training history, distribution, heatmap, metrics table, balance, inference performance
    HIGHLIGHT_ROWS = 4    # Rows 6-9: highlighted samples (4 rows x 2 columns = 8 cells)
    base_conf = TOP_CHART_ROWS + HIGHLIGHT_ROWS  # Confusion matrices start from this row index
    total_rows = TOP_CHART_ROWS + HIGHLIGHT_ROWS + num_conf_rows

    # Create figure and gridspec with the new total rows.
    fig = plt.figure(figsize=(20, 6 * total_rows))
    gs = gridspec.GridSpec(total_rows, 4, figure=fig)

    # --- Top Charts (Rows 0-5) ---
    # Row 0: Training History (4 subplots)
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

    # Row 2: Heatmap of Entity Category vs. Binned Sentence Length
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

    # Row 5: Inference Performance Bar Chart by Sentence Length Bin
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

    # --- Highlighted Samples Grid (Rows 6-9) ---
    max_highlights = HIGHLIGHT_ROWS * 2  # 2 columns * 4 rows = 8 samples
    highlight_samples = []
    for sent in val_sentences:
        info = get_sample_by_sentence(sent, raw_data)
        if info and "entities" in info and info["entities"]:
            highlight_samples.append((sent, info))
        if len(highlight_samples) >= max_highlights:
            break

    for idx in range(max_highlights):
        row_idx = TOP_CHART_ROWS + (idx // 2)  # Rows 6 to 9
        col_idx = idx % 2
        # Each sample spans 2 columns.
        if col_idx == 0:
            ax = fig.add_subplot(gs[row_idx, 0:2])
        else:
            ax = fig.add_subplot(gs[row_idx, 2:4])
        ax.axis("off")
        if idx < len(highlight_samples):
            sample_sentence, sample_info = highlight_samples[idx]
            sample_sentence = str(sample_sentence)
            pred = model.predict([sample_sentence])[0]
            pred_binary = (pred >= threshold).astype(int)
            predicted_categories = [categories[i] for i, val in enumerate(pred_binary) if val == 1]
            orig_entities = sample_info["entities"]
            filtered_entities = [ent for ent in orig_entities if ent.get("category") in predicted_categories]
            if not filtered_entities:
                filtered_entities = orig_entities
            filtered_entities = convert_entities(filtered_entities)
            pil_img = get_highlighted_text_image_wrapped(sample_sentence, filtered_entities)
            img_array = np.array(pil_img)
            ax.imshow(img_array)
            ax.set_title(f"Sample {idx+1}", fontsize=12)
        else:
            ax.text(0.5, 0.5, "No sample", horizontalalignment="center", verticalalignment="center")

    # --- Confusion Matrices (Rows base_conf onward) ---
    for idx, cat in enumerate(categories):
        row_idx = base_conf + (idx // 4)
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
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"], ax=ax_cm)
        ax_cm.set_title(f"{cat}")
        ax_cm.text(0.5, -0.3, f"Acc: {acc:.2f}\nPrec: {prec:.2f}\nRec: {rec:.2f}",
                   fontsize=10, ha="center", transform=ax_cm.transAxes)
    
    # Turn off any extra subplots in the confusion matrix area.
    num_conf_plots = num_conf_rows * 4
    for extra in range(num_categories, num_conf_plots):
        row_idx = base_conf + (extra // 4)
        col_idx = extra % 4
        ax_empty = fig.add_subplot(gs[row_idx, col_idx])
        ax_empty.axis('off')
    
    # Adjust spacing to reduce padding while preserving all sections.
    plt.subplots_adjust(hspace=0.8, wspace=0.5)
    
    combined_path = os.path.join(output_dir, "combined_post_training_charts.png")
    plt.savefig(combined_path)
    plt.close()
    print(f"[INFO] Combined post-training charts saved to {combined_path}")


# -------------------------------------------------------------------
# Evaluation Function
# -------------------------------------------------------------------
def evaluate_model(model, val_dataset, categories, output_dir, sentences, labels,
                   train_labels, val_labels, val_sentences, raw_data, threshold=0.5):
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
    metrics_path = os.path.join(output_dir, "performance_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(performance_metrics, f, indent=4)
    print(f"[INFO] Performance metrics saved to {metrics_path}")
    
    # Since no training history is available now, create a dummy history object.
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
    
    plot_combined_post_training_charts(
        dummy_history_obj, sentences, labels, y_true, y_pred, categories,
        train_labels, val_labels, val_sentences, model, raw_data,
        output_dir, threshold
    )

# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------
def main():
    print("[INFO] Starting evaluation...")

    # Load the saved model.
    print("[INFO] Loading model from:", MODEL_SAVE_PATH)
    model = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects={'KerasLayer': hub.KerasLayer})
    
    # Load the category labels.
    with open(LABELS_SAVE_PATH, "r") as f:
        categories = json.load(f)
    
    # Load and preprocess the raw data.
    raw_data = load_data(DATA_FILE)
    sentences, labels = preprocess_data(raw_data, categories)
    train_dataset, val_dataset, sentences_train, sentences_val, labels_train, labels_val = create_datasets(sentences, labels, batch_size=16)
    
    # Evaluate the model and generate output files in the EVAL folder.
    evaluate_model(
        model, val_dataset, categories, EVAL_DIR, sentences, labels,
        labels_train, labels_val, sentences_val, raw_data, threshold=0.5
    )

if __name__ == "__main__":
    main()
