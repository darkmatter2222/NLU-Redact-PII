import os
import json
import math
import time
import random
import textwrap
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # Registers the custom op 'CaseFoldUTF8'

from sklearn.model_selection import train_test_split

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
    
    # Load and preprocess the data.
    raw_data = load_data(data_file)
    sentences, labels = preprocess_data(raw_data, categories)
    train_dataset, val_dataset, sentences_train, sentences_val, labels_train, labels_val = create_datasets(sentences, labels, batch_size=16)
    
    # Save the category labels.
    with open(labels_save_path, "w") as f:
        json.dump(categories, f, indent=4)
    print(f"[INFO] Labels saved to {labels_save_path}")
    
    # Build and train the model.
    model = build_model(num_categories=len(categories))
    history = train_model(model, train_dataset, val_dataset, epochs=40)
    
    # Save the training statistics to a JSON file.
    training_stats_path = os.path.join(OUTPUT_DIR, "training_stats.json")
    with open(training_stats_path, "w") as f:
        json.dump(history.history, f, indent=4)
    print(f"[INFO] Training statistics saved to {training_stats_path}")
    
    # Save the final trained model.
    model.save(model_save_path)
    print(f"[INFO] Final model saved to {model_save_path}")
    
    # Use the evaluation class from evaluate.py to run evaluation and charting.
    from evaluate.evaluate import ModelEvaluator
    evaluator = ModelEvaluator(
        root_dir=OUTPUT_DIR,
        model_save_path=model_save_path,
        labels_save_path=labels_save_path,
        data_file=data_file
    )
    evaluator.run_evaluation(threshold=0.5, batch_size=16)

if __name__ == "__main__":
    main()
