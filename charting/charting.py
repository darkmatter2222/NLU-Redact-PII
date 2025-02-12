import os
import math
import time
import textwrap
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix


class ChartPlotter:
    """
    A chart plotting utility for post-training visualization.
    Generates several groups of figures including training history, category distribution,
    heatmaps, per-category metrics, dataset balance, inference performance, highlighted samples,
    and confusion matrices. Each group is saved as a separate image file.
    """

    def __init__(self, bin_size: int = 5):
        self.bin_size = bin_size
        # Pastel colors for entity categories.
        self.colors = {
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

    def _get_bin_label(self, sentence_length: int) -> str:
        """Return a string label for the bin corresponding to the sentence length."""
        low = (sentence_length // self.bin_size) * self.bin_size
        high = low + self.bin_size - 1
        return f"{low}-{high}"

    def _get_sample_by_sentence(self, sentence: str, raw_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return the first sample from raw_data matching the provided sentence."""
        for sample in raw_data:
            if sample.get("sentence", "") == sentence:
                return sample
        return {}

    def _convert_entities(self, entities: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """
        Converts a list of entity dictionaries into a list of tuples (entity_text, category).
        """
        converted = []
        for ent in entities:
            if isinstance(ent, dict):
                if "word" in ent:
                    converted.append((ent["word"], ent["category"]))
                elif "text" in ent:
                    converted.append((ent["text"], ent["category"]))
                elif ent.get("category") == "Residential Address" and "address" in ent:
                    converted.append((ent["address"], ent["category"]))
            else:
                converted.append(ent)
        return converted

    def _create_gradient_background(
        self, width: int, height: int, start_color: Tuple[int, int, int], end_color: Tuple[int, int, int]
    ) -> Image.Image:
        """
        Creates a vertical gradient background image from start_color (top) to end_color (bottom).
        """
        base = Image.new("RGB", (width, height), start_color)
        draw = ImageDraw.Draw(base)
        for y in range(height):
            ratio = y / float(height)
            r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
            g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
            b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        return base

    def _get_highlighted_text_images_wrapped(
        self, text: str, entities: List[Tuple[str, str]], num_variants: int = 5
    ) -> List[Image.Image]:
        """
        Generates a list of `num_variants` PIL Images that render the provided text with
        highlighted entity substrings. Each variant uses a different modern, clean style.
        The text is wrapped and each line is horizontally centered.
        
        For each entity (a tuple of (entity_text, category)), the text is "protected"
        (spaces replaced with non-breaking spaces) so that textwrap does not split it.
        """
        images = []
        container_width = 1500
        container_height = 600

        for variant in range(num_variants):
            # Set default margins and padding.
            margin_x = 20
            margin_y = 20
            padding = 4  # padding around highlighted entities

            # Set style parameters based on the variant.
            if variant == 0:
                # Variant 0: Modern Clean
                bg_color = "white"
                text_color = "black"
                highlight_style = "rounded"  # filled, rounded rectangle
                try:
                    font = ImageFont.truetype("arial.ttf", 32)
                    font_small = ImageFont.truetype("arial.ttf", 24)
                except IOError:
                    font = ImageFont.load_default()
                    font_small = ImageFont.load_default()

            elif variant == 1:
                # Variant 1: Classic Paper Look
                bg_color = "#f5f5dc"  # beige parchment
                text_color = "#4B3621"  # dark brown text
                highlight_style = "border"  # rectangle with border
                try:
                    font = ImageFont.truetype("times.ttf", 32)
                    font_small = ImageFont.truetype("times.ttf", 24)
                except IOError:
                    font = ImageFont.load_default()
                    font_small = ImageFont.load_default()

            elif variant == 2:
                # Variant 2: Dark Mode
                bg_color = "#333333"
                text_color = "white"
                highlight_style = "rounded"
                try:
                    font = ImageFont.truetype("arial.ttf", 32)
                    font_small = ImageFont.truetype("arial.ttf", 24)
                except IOError:
                    font = ImageFont.load_default()
                    font_small = ImageFont.load_default()

            elif variant == 3:
                # Variant 3: Minimalist Underline
                bg_color = "#f0f0f0"
                text_color = "black"
                highlight_style = "underline"
                try:
                    font = ImageFont.truetype("arial.ttf", 32)
                    font_small = ImageFont.truetype("arial.ttf", 24)
                except IOError:
                    font = ImageFont.load_default()
                    font_small = ImageFont.load_default()

            elif variant == 4:
                # Variant 4: Elegant with a Gradient Background
                start_color = (52, 152, 219)   # light blue
                end_color = (41, 128, 185)       # darker blue
                bg_img = self._create_gradient_background(container_width, container_height, start_color, end_color)
                text_color = "white"
                highlight_style = "translucent"  # draw a translucent highlight
                try:
                    font = ImageFont.truetype("arial.ttf", 32)
                    font_small = ImageFont.truetype("arial.ttf", 24)
                except IOError:
                    font = ImageFont.load_default()
                    font_small = ImageFont.load_default()

            # Create the image background.
            if variant == 4:
                img = bg_img.copy()
            else:
                img = Image.new("RGB", (container_width, container_height), bg_color)
            draw = ImageDraw.Draw(img)

            # Protect each entity: replace spaces with non-breaking spaces.
            protected_map = {}
            proc_text = text
            for entity_text, category in entities:
                protected = entity_text.replace(" ", "\u00A0")
                protected_map[protected] = (entity_text, category)
                proc_text = proc_text.replace(entity_text, protected)

            # Compute available width and wrap text.
            available_width = container_width - 2 * margin_x
            bbox = font.getbbox("A")
            avg_char_width = bbox[2] - bbox[0]
            wrap_chars = max(1, available_width // avg_char_width)
            wrapped_lines = textwrap.wrap(proc_text, width=wrap_chars, break_long_words=False, break_on_hyphens=False)

            # Determine line height.
            _, _, _, line_height = draw.textbbox((0, 0), "Ag", font=font)
            line_height = max(line_height, 50)
            total_text_height = line_height * len(wrapped_lines)
            vertical_offset = margin_y + (container_height - total_text_height - 2 * margin_y) // 2

            # Draw each line of text with horizontal centering.
            y = vertical_offset
            for line in wrapped_lines:
                # Compute the width of this line.
                line_bbox = draw.textbbox((0, 0), line, font=font)
                line_width = line_bbox[2] - line_bbox[0]
                # Center horizontally.
                x = (container_width - line_width) / 2
                draw.text((x, y), line, fill=text_color, font=font)
                # Look for protected entity substrings and apply highlights.
                for protected, (original_entity, category) in protected_map.items():
                    index = line.find(protected)
                    if index != -1:
                        prefix = line[:index]
                        prefix_bbox = draw.textbbox((0, 0), prefix, font=font)
                        x_start = x + (prefix_bbox[2] - prefix_bbox[0])
                        entity_bbox = draw.textbbox((0, 0), protected, font=font)
                        entity_width = entity_bbox[2] - entity_bbox[0]
                        entity_height = entity_bbox[3] - entity_bbox[1]
                        if highlight_style in ["rounded", "border", "translucent"]:
                            rect = [x_start - padding, y - padding, x_start + entity_width + padding, y + entity_height + padding]
                            highlight_color = self.colors.get(category, "#FFB6C1")
                            if highlight_style == "rounded":
                                draw.rounded_rectangle(rect, fill=highlight_color, radius=8)
                            elif highlight_style == "border":
                                draw.rectangle(rect, fill=highlight_color, outline=text_color, width=2)
                            elif highlight_style == "translucent":
                                overlay = Image.new("RGBA", (container_width, container_height), (0, 0, 0, 0))
                                overlay_draw = ImageDraw.Draw(overlay)
                                hex_color = self.colors.get(category, "#FFB6C1").lstrip("#")
                                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                                overlay_draw.rounded_rectangle(rect, fill=(r, g, b, 150), radius=8)
                                img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
                                draw = ImageDraw.Draw(img)
                            draw.text((x_start, y), original_entity, fill=text_color, font=font)
                            if highlight_style != "underline":
                                draw.text((x_start, y + entity_height + 2), category, fill="gray", font=font_small)
                        elif highlight_style == "underline":
                            underline_y = y + entity_height + 2
                            draw.line([(x_start, underline_y), (x_start + entity_width, underline_y)],
                                      fill=self.colors.get(category, "blue"), width=2)
                y += line_height

            images.append(img)
        return images

    def _standardize_confusion_matrix(self, cm: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:
        """
        Ensures the confusion matrix is 2x2 and returns it along with its flattened components:
        (TN, FP, FN, TP).
        """
        if cm.shape == (1, 1):
            cm = np.array([[cm[0, 0], 0], [0, 0]])
        elif cm.shape == (1, 2):
            cm = np.vstack([cm, [0, 0]])
        elif cm.shape == (2, 1):
            cm = np.hstack([cm, np.array([[0], [0]])])
        if cm.size == 4:
            TN, FP, FN, TP = cm.ravel()
        else:
            TN = FP = FN = TP = 0
        return cm, TN, FP, FN, TP

    def plot_combined_post_training_charts(
        self,
        history: Any,
        sentences: List[str],
        labels: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        categories: List[str],
        train_labels: np.ndarray,
        val_labels: np.ndarray,
        val_sentences: List[str],
        model: Any,
        raw_data: List[Dict[str, Any]],
        output_dir: str,
        threshold: float = 0.5
    ) -> None:
        """
        Generates and saves separate figures for the following chart groups:
          1. Training History (loss, accuracy, precision, recall)
          2. Category Distribution
          3. Heatmap: Entity Category vs. Sentence Length Bin
          4. Per-Category Metrics Table
          5. Training/Validation Dataset & Category Balance
          6. (Optional Inference Performance Metrics by Sentence Length Bin)
          7. Highlighted Sample Texts with Entity Highlights (saved as 6 individual images)
          8. Confusion Matrices for each Category

        All images are saved to output_dir.
        """
        sns.set_theme(style="whitegrid")

        # ---------------------------
        # Group 1: Training History
        # ---------------------------
        metrics_list = ["loss", "accuracy", "precision", "recall"]
        fig, axes = plt.subplots(1, len(metrics_list), figsize=(20, 5))
        for i, metric in enumerate(metrics_list):
            axes[i].plot(history.history[metric], marker='o', label=f"Train {metric.capitalize()}")
            axes[i].plot(history.history["val_" + metric], marker='x', label=f"Val {metric.capitalize()}")
            axes[i].set_xlabel("Epoch", fontsize=12)
            axes[i].set_ylabel(metric.capitalize(), fontsize=12)
            axes[i].set_title(metric.capitalize(), fontsize=14, pad=10)
            axes[i].legend()
            axes[i].grid(True)
        fig.suptitle("Training History", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        training_history_path = os.path.join(output_dir, "training_history.png")
        plt.savefig(training_history_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Training History saved to {training_history_path}")

        # -------------------------------
        # Group 2: Category Distribution
        # -------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        counts = np.sum(labels, axis=0)
        sns.barplot(x=categories, y=counts, palette="pastel", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
        ax.set_xlabel("Category", fontsize=14, labelpad=10)
        ax.set_ylabel("Count", fontsize=14, labelpad=10)
        ax.set_title("Sample Count per Category", fontsize=16, pad=10)
        fig.tight_layout()
        category_distribution_path = os.path.join(output_dir, "category_distribution.png")
        plt.savefig(category_distribution_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Category Distribution saved to {category_distribution_path}")

        # -----------------------------------------------------------
        # Group 3: Heatmap of Entity Category vs. Sentence Length Bin
        # -----------------------------------------------------------
        counts_dict: Dict[str, Dict[str, int]] = {}
        for sample in raw_data:
            sentence = sample.get("sentence", "")
            words = sentence.split()
            length = len(words)
            bin_label = self._get_bin_label(length)
            counts_dict.setdefault(bin_label, {})
            for entity in sample.get("entities", []):
                cat = entity.get("category", "Unknown")
                counts_dict[bin_label][cat] = counts_dict[bin_label].get(cat, 0) + 1
        df = pd.DataFrame.from_dict(counts_dict, orient="index").fillna(0)
        df.index = pd.CategoricalIndex(
            df.index,
            categories=sorted(df.index, key=lambda x: int(x.split('-')[0])),
            ordered=True
        )
        df = df.sort_index()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df, annot=True, fmt=".0f", cmap="viridis", ax=ax)
        ax.set_xlabel("Entity Category", fontsize=14, labelpad=10)
        ax.set_ylabel("Sentence Length Bins (words)", fontsize=14, labelpad=10)
        ax.set_title("Heatmap: Entity Category vs. Binned Sentence Length", fontsize=16, pad=20)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
        fig.tight_layout()
        heatmap_path = os.path.join(output_dir, "entity_category_heatmap.png")
        plt.savefig(heatmap_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Entity Category Heatmap saved to {heatmap_path}")

        # --------------------------------------
        # Group 4: Per-Category Metrics Table
        # --------------------------------------
        y_pred_binary = (y_pred >= threshold).astype(int)
        per_cat_data = []
        for idx, cat in enumerate(categories):
            cm = confusion_matrix(y_true[:, idx], y_pred_binary[:, idx])
            cm, TN, FP, FN, TP = self._standardize_confusion_matrix(cm)
            total = TN + FP + FN + TP
            accuracy = (TP + TN) / total if total > 0 else 0
            precision_val = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall_val = TP / (TP + FN) if (TP + FN) > 0 else 0
            per_cat_data.append([cat, f"{precision_val:.2f}", f"{recall_val:.2f}", f"{accuracy:.2f}"])
        fig, ax = plt.subplots(figsize=(12, 1 + len(categories) * 0.5))
        ax.axis('off')
        table = ax.table(cellText=per_cat_data,
                         colLabels=["Category", "Precision", "Recall", "Accuracy"],
                         loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        ax.set_title("Per-Category Metrics", fontsize=16, pad=10)
        fig.tight_layout()
        metrics_table_path = os.path.join(output_dir, "per_category_metrics.png")
        plt.savefig(metrics_table_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Per-Category Metrics Table saved to {metrics_table_path}")

        # ---------------------------------------------------------
        # Group 5: Training/Validation Dataset & Category Balance
        # ---------------------------------------------------------
        train_counts = np.sum(train_labels, axis=0)
        val_counts = np.sum(val_labels, axis=0)
        x = np.arange(len(categories))
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, train_counts, width, label="Training", color="blue", alpha=0.7)
        ax.bar(x + width / 2, val_counts, width, label="Validation", color="orange", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=12)
        ax.set_xlabel("Category", fontsize=14, labelpad=10)
        ax.set_ylabel("Count", fontsize=14, labelpad=10)
        ax.set_title("Training/Validation Dataset & Category Balance", fontsize=16, pad=10)
        ax.legend(fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()
        dataset_balance_path = os.path.join(output_dir, "dataset_category_balance.png")
        plt.savefig(dataset_balance_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Dataset & Category Balance Chart saved to {dataset_balance_path}")

        # --------------------------------------------------
        # Group 7: Highlighted Sample Texts as Individual Images
        # --------------------------------------------------
        max_highlights = 6
        highlight_samples = []
        for sent in val_sentences:
            sample_info = self._get_sample_by_sentence(sent, raw_data)
            if sample_info and sample_info.get("entities"):
                highlight_samples.append((sent, sample_info))
            if len(highlight_samples) >= max_highlights:
                break
        if len(highlight_samples) == 0:
            print("[INFO] No highlighted samples available.")
        else:
            for i, (sample_sentence, sample_info) in enumerate(highlight_samples):
                sample_sentence = str(sample_sentence)
                pred = model.predict([sample_sentence])[0]
                pred_binary = (pred >= threshold).astype(int)
                predicted_categories = [categories[k] for k, val in enumerate(pred_binary) if val == 1]
                orig_entities = sample_info.get("entities", [])
                filtered_entities = [ent for ent in orig_entities if ent.get("category") in predicted_categories]
                if not filtered_entities:
                    filtered_entities = orig_entities
                filtered_entities = self._convert_entities(filtered_entities)
                # Generate five highlighted variants; choose the first for saving.
                pil_images = self._get_highlighted_text_images_wrapped(sample_sentence, filtered_entities, num_variants=5)
                output_path = os.path.join(output_dir, f"highlighted_sample_{i+1}.png")
                pil_images[0].save(output_path)
                print(f"[INFO] Highlighted sample {i+1} saved to {output_path}")

        # --------------------------------------------------
        # Group 8: Confusion Matrices for each Category
        # --------------------------------------------------
        num_categories = len(categories)
        num_cols = 4
        num_rows = math.ceil(num_categories / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
        axes = axes.flatten()
        for idx, cat in enumerate(categories):
            ax_cm = axes[idx]
            cm = confusion_matrix(y_true[:, idx], y_pred_binary[:, idx])
            cm, TN, FP, FN, TP = self._standardize_confusion_matrix(cm)
            total = TN + FP + FN + TP
            acc = (TP + TN) / total if total > 0 else 0
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"],
                        ax=ax_cm, square=True)
            ax_cm.set_title(f"{cat}", fontsize=14, pad=10)
            ax_cm.text(0.5, -0.25, f"Acc: {acc:.2f}\nPrec: {prec:.2f}\nRec: {rec:.2f}",
                       fontsize=10, ha="center", va="top", transform=ax_cm.transAxes)
        for extra in range(num_categories, len(axes)):
            axes[extra].axis('off')
        fig.suptitle("Confusion Matrices by Category", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        confusion_matrices_path = os.path.join(output_dir, "confusion_matrices.png")
        plt.savefig(confusion_matrices_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Confusion Matrices saved to {confusion_matrices_path}")
