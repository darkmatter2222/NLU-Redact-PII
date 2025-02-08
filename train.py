import os
import json
import re
import math
import collections
import numpy as np
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
    # Also return the raw training/validation arrays for later use in tables
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
# Combined Post-Training Charts (with two additional tables)
# -----------------------------
def plot_combined_post_training_charts(history, sentences, labels, y_true, y_pred, categories, train_labels, val_labels, output_dir, threshold=0.5):
    print("[INFO] Creating combined post-training charts...")
    y_pred_binary = (y_pred >= threshold).astype(int)
    num_categories = len(categories)
    num_conf_rows = math.ceil(num_categories / 4)
    # We now add two extra rows (one for per-category metrics table and one for training/val dataset table)
    total_rows = 5 + num_conf_rows  # rows: 0=training history, 1=category distribution, 2=word density, 3=per-cat metrics table, 4=dataset table, rows 5+ for confusion matrices

    fig = plt.figure(figsize=(20, 5 * total_rows))
    gs = gridspec.GridSpec(total_rows, 4, figure=fig)
    
    # 1. Training History (Row 0: 4 subplots)
    metrics = ["loss", "accuracy", "precision", "recall"]
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(history.history[metric], label=f"Train {metric.capitalize()}", marker='o')
        ax.plot(history.history["val_" + metric], label=f"Val {metric.capitalize()}", marker='x')
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.legend()
    
    # 2. Category Distribution (Row 1, span all columns)
    ax_dist = fig.add_subplot(gs[1, :])
    counts = np.sum(labels, axis=0)
    sns.barplot(x=categories, y=counts, ax=ax_dist)
    ax_dist.set_xticklabels(categories, rotation=45, ha="right")
    ax_dist.set_xlabel("Category")
    ax_dist.set_ylabel("Count")
    ax_dist.set_title("Sample Count per Category")
    
    # 3. Word Density Heatmap (Row 2, span all columns)
    # For each category, count word frequencies from sentences labeled with that category.
    category_word_counts = {cat: collections.Counter() for cat in categories}
    overall_counter = collections.Counter()
    for sentence, label_vector in zip(sentences, labels):
        # Simple tokenizer: lower-case and remove non-alphanumerics
        words = re.sub(r"[^a-zA-Z0-9\s]", "", sentence.lower()).split()
        overall_counter.update(words)
        for i, cat_flag in enumerate(label_vector):
            if cat_flag == 1:
                category_word_counts[categories[i]].update(words)
    top_n = 20
    top_words = [word for word, _ in overall_counter.most_common(top_n)]
    heatmap_data = []
    for cat in categories:
        row = [category_word_counts[cat][word] for word in top_words]
        heatmap_data.append(row)
    ax_heat = fig.add_subplot(gs[2, :])
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", 
                yticklabels=categories, xticklabels=top_words, ax=ax_heat)
    ax_heat.set_title("Word Density Heatmap (Top {} Words) per Category".format(top_n))
    ax_heat.set_xlabel("Words")
    ax_heat.set_ylabel("Category")
    
    # 4. Per-Category Metrics Table (Row 3, span all columns)
    # Compute precision, recall, and accuracy for each category using y_true and y_pred_binary
    per_cat_data = []
    for i, cat in enumerate(categories):
        cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i])
        # Ensure the confusion matrix is 2x2
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
    
    # Create a table (using a subplot with axis turned off)
    ax_table1 = fig.add_subplot(gs[3, :])
    ax_table1.axis('tight')
    ax_table1.axis('off')
    col_labels = ["Category", "Precision", "Recall", "Accuracy"]
    table1 = ax_table1.table(cellText=per_cat_data, colLabels=col_labels, loc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    ax_table1.set_title("Per-Category Metrics", fontweight="bold")
    
    # 5. Training/Validation Dataset & Category Balance Table (Row 4, span all columns)
    # Compute counts for training and validation splits:
    train_total = len(train_labels)
    val_total = len(val_labels)
    train_counts = np.sum(train_labels, axis=0)
    val_counts = np.sum(val_labels, axis=0)
    # Create table data with headers: "Dataset", "Total", followed by each category
    header = ["Dataset", "Total"] + categories
    train_row = ["Training", str(train_total)] + [str(int(count)) for count in train_counts]
    val_row = ["Validation", str(val_total)] + [str(int(count)) for count in val_counts]
    dataset_table_data = [train_row, val_row]
    
    ax_table2 = fig.add_subplot(gs[4, :])
    ax_table2.axis('tight')
    ax_table2.axis('off')
    table2 = ax_table2.table(cellText=dataset_table_data, colLabels=header, loc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    ax_table2.set_title("Training/Validation Dataset & Category Balance", fontweight="bold")
    
    # 6. Confusion Matrices for Each Category (Rows 5 to end)
    for idx, cat in enumerate(categories):
        row_idx = 5 + idx // 4
        col_idx = idx % 4
        ax_cm = fig.add_subplot(gs[row_idx, col_idx])
        
        # Compute confusion matrix for this category
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
        
        # Plot confusion matrix heatmap
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        ax_cm.set_title(f"{cat}")
        ax_cm.text(0.5, -0.3, f"Acc: {acc:.2f}\nPrec: {prec:.2f}\nRec: {rec:.2f}",
                   fontsize=10, ha="center", transform=ax_cm.transAxes)
    
    # If there are extra empty subplots in the confusion matrix grid, hide them.
    total_conf_plots = num_conf_rows * 4
    for extra in range(num_categories, total_conf_plots):
        row_idx = 5 + extra // 4
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
def evaluate_model(model, val_dataset, categories, output_dir, history, sentences, labels, train_labels, val_labels, threshold=0.5):
    print("[INFO] Evaluating the model on the validation set...")
    # Overall evaluation via model.evaluate
    eval_results = model.evaluate(val_dataset)
    print("\n[RESULTS] Overall Evaluation on Validation Set:")
    print(f"         Loss:      {eval_results[0]:.4f}")
    print(f"         Accuracy:  {eval_results[1]:.4f}")
    print(f"         Precision: {eval_results[2]:.4f}")
    print(f"         Recall:    {eval_results[3]:.4f}")
    
    # Collect true labels and predictions from the entire validation set
    y_true = []
    y_pred = []
    for texts, batch_labels in val_dataset:
        predictions = model.predict(texts)
        y_true.extend(batch_labels.numpy())
        y_pred.extend(predictions)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Additional Metrics using scikit-learn
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
    
    # Generate the combined chart image containing all post-training plots
    plot_combined_post_training_charts(history, sentences, labels, y_true, y_pred, categories, train_labels, val_labels, output_dir, threshold)

# -----------------------------
# Main Execution
# -----------------------------
def main():
    # File paths
    data_file = r"O:\master_data_collection\redact\synthetic_data.json"
    model_save_path = os.path.join(OUTPUT_DIR, "final_model.h5")
    labels_save_path = os.path.join(OUTPUT_DIR, "labels.json")
    
    # Define all possible PII categories
    categories = [
        "People Name", "Card Number", "Account Number", "Social Security Number",
        "Government ID Number", "Date of Birth", "Password", "Tax ID Number",
        "Phone Number", "Residential Address", "Email Address", "IP Number",
        "Passport", "Driver License"
    ]
    
    # Load, preprocess, and split the data
    data = load_data(data_file)
    sentences, labels = preprocess_data(data, categories)
    train_dataset, val_dataset, sentences_train, sentences_val, labels_train, labels_val = create_datasets(sentences, labels, batch_size=16)
    
    # Save label list as JSON for later use
    with open(labels_save_path, "w") as f:
        json.dump(categories, f, indent=4)
    print(f"[INFO] Labels saved to {labels_save_path}")
    
    # Build and train the model for 25 epochs
    model = build_model(num_categories=len(categories))
    history = train_model(model, train_dataset, val_dataset, epochs=25)
    
    # Save the final trained model
    model.save(model_save_path)
    print(f"[INFO] Final model saved to {model_save_path}")
    
    # Evaluate the model with extended metrics and generate combined charts.
    # Note: We pass the full sentences/labels (for word density) and the training/validation arrays for the dataset table.
    evaluate_model(model, val_dataset, categories, OUTPUT_DIR, history, sentences, labels, labels_train, labels_val, threshold=0.5)

if __name__ == "__main__":
    main()
