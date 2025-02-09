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
    Generates a combined figure including training history, category distribution,
    heatmaps, per-category metrics, sample highlights, and confusion matrices.
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

    def _get_highlighted_text_image_wrapped(self, text: str, entities: List[Tuple[str, str]]) -> Image.Image:
        """
        Render the provided text into a PIL image with wrapped lines.
        Substrings matching an entity are highlighted with a pastel-colored rounded rectangle.
        The wrap width is computed dynamically based on the available image width.
        """
        # Image configuration
        img_width = 1000
        margin_x = 20
        margin_y = 20

        # Attempt to load fonts; fall back to default if unavailable.
        try:
            font = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except IOError:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Compute average character width and calculate wrap width.
        bbox = font.getbbox("A")
        char_width = bbox[2] - bbox[0]
        available_width = img_width - 2 * margin_x
        wrap_chars = max(1, available_width // char_width)

        # Wrap text without breaking long words.
        wrapped_lines = textwrap.wrap(
            text,
            width=wrap_chars,
            break_long_words=False,
            break_on_hyphens=False
        )

        # Compute line height (using representative text "Ay") and determine image height.
        bbox_line = font.getbbox("Ay")
        line_height = (bbox_line[3] - bbox_line[1]) + 10  # extra spacing
        img_height = 2 * margin_y + line_height * len(wrapped_lines)

        # Create a white canvas and prepare for drawing.
        img = Image.new("RGB", (img_width, img_height), "white")
        draw = ImageDraw.Draw(img)

        # Draw text line by line with entity highlighting.
        y = margin_y
        for line in wrapped_lines:
            x = margin_x
            # First, draw the plain text.
            draw.text((x, y), line, fill="black", font=font)

            # Then, overlay entity highlights.
            for entity_text, category in entities:
                index = line.find(entity_text)
                if index != -1:
                    # Calculate x-offset for the entity within the line.
                    prefix = line[:index]
                    prefix_bbox = draw.textbbox((0, 0), prefix, font=font)
                    x_start = x + (prefix_bbox[2] - prefix_bbox[0])
                    entity_bbox = draw.textbbox((0, 0), entity_text, font=font)
                    entity_width = entity_bbox[2] - entity_bbox[0]
                    entity_height = entity_bbox[3] - entity_bbox[1]
                    padding = 2
                    rect = [
                        x_start - padding, y - padding,
                        x_start + entity_width + padding, y + entity_height + padding
                    ]
                    draw.rounded_rectangle(rect, fill=self.colors.get(category, "#FFB6C1"), radius=5)
                    # Redraw the entity text over the highlight.
                    draw.text((x_start, y), entity_text, fill="black", font=font)
                    # Optionally, draw the category label below the entity text.
                    draw.text((x_start, y + entity_height + 2), category, fill="gray", font=font_small)
            y += line_height

        return img

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
        Generates a figure with multiple subplots:
          - Training history
          - Category distribution
          - Heatmap (Entity Category vs. Sentence Length Bin)
          - Per-category metrics table
          - Training/Validation category balance
          - Inference performance metrics
          - Highlighted sample texts with entity highlights
          - Confusion matrices for each category

        The resulting image is saved to output_dir.
        """
        # Use Seaborn's clean whitegrid style.
        sns.set_theme(style="whitegrid")

        num_categories = len(categories)
        y_pred_binary = (y_pred >= threshold).astype(int)
        num_conf_rows = math.ceil(num_categories / 4)

        # Define grid layout rows.
        TOP_CHART_ROWS = 6    # Rows 0-5: various charts.
        HIGHLIGHT_ROWS = 4    # Rows 6-9: highlighted sample images.
        base_conf = TOP_CHART_ROWS + HIGHLIGHT_ROWS  # Starting row for confusion matrices.
        total_rows = TOP_CHART_ROWS + HIGHLIGHT_ROWS + num_conf_rows

        # Custom height ratios.
        top_ratios = [1, 1, 2, 2.5, 1, 1]           # Rows 0-5
        highlight_ratios = [0.7] * HIGHLIGHT_ROWS     # Rows 6-9
        confusion_ratios = [1] * num_conf_rows        # Confusion matrix rows
        height_ratios = top_ratios + highlight_ratios + confusion_ratios

        # Create the figure with gridspec.
        fig = plt.figure(figsize=(20, 6 * total_rows))
        gs = gridspec.GridSpec(total_rows, 4, figure=fig, height_ratios=height_ratios)

        # ----------------------------
        # Row 0: Training History
        # ----------------------------
        metrics_list = ["loss", "accuracy", "precision", "recall"]
        for i, metric in enumerate(metrics_list):
            ax = fig.add_subplot(gs[0, i])
            ax.plot(history.history[metric], marker='o', label=f"Train {metric.capitalize()}")
            ax.plot(history.history["val_" + metric], marker='x', label=f"Val {metric.capitalize()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(metric.capitalize(), pad=10)
            ax.legend()

        # ----------------------------
        # Row 1: Category Distribution
        # ----------------------------
        ax_dist = fig.add_subplot(gs[1, :])
        counts = np.sum(labels, axis=0)
        sns.barplot(x=categories, y=counts, ax=ax_dist, palette="pastel")
        ax_dist.set_xticklabels(ax_dist.get_xticklabels(), rotation=45, ha="right")
        ax_dist.set_xlabel("Category", labelpad=10)
        ax_dist.set_ylabel("Count", labelpad=10)
        ax_dist.set_title("Sample Count per Category", pad=10)

        # ---------------------------------------------------------------
        # Row 2: Heatmap of Entity Category vs. Binned Sentence Length
        # ---------------------------------------------------------------
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
        # Ensure bins are ordered numerically.
        df.index = pd.CategoricalIndex(
            df.index,
            categories=sorted(df.index, key=lambda x: int(x.split('-')[0])),
            ordered=True
        )
        df = df.sort_index()
        ax_heat = fig.add_subplot(gs[2, :])
        sns.heatmap(df, annot=True, fmt=".0f", cmap="viridis", ax=ax_heat)
        ax_heat.set_xlabel("Entity Category", labelpad=10)
        ax_heat.set_ylabel("Sentence Length Bins (words)", labelpad=10)
        ax_heat.set_title("Heatmap: Entity Category vs. Binned Sentence Length", pad=20)
        ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha="right")

        # ----------------------------
        # Row 3: Per-Category Metrics Table
        # ----------------------------
        per_cat_data = []
        for cat in categories:
            cm = confusion_matrix(y_true[:, categories.index(cat)], y_pred_binary[:, categories.index(cat)])
            cm, TN, FP, FN, TP = self._standardize_confusion_matrix(cm)
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
        table1.scale(1, 1.5)
        ax_table1.text(0.5, 1.02, "Per-Category Metrics", transform=ax_table1.transAxes,
                       ha='center', va='bottom', fontsize=12, fontweight="bold")

        # ----------------------------------------------------------
        # Row 4: Training/Validation Dataset & Category Balance Bar Chart
        # ----------------------------------------------------------
        train_counts = np.sum(train_labels, axis=0)
        val_counts = np.sum(val_labels, axis=0)
        ax_bar = fig.add_subplot(gs[4, :])
        x = np.arange(len(categories))
        width = 0.35
        ax_bar.bar(x - width/2, train_counts, width, label="Training", color="blue", alpha=0.7)
        ax_bar.bar(x + width/2, val_counts, width, label="Validation", color="orange", alpha=0.7)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(categories, rotation=45, ha="right")
        ax_bar.set_xlabel("Category", labelpad=10)
        ax_bar.set_ylabel("Count", labelpad=10)
        ax_bar.set_title("Training/Validation Dataset & Category Balance", pad=10)
        ax_bar.legend()

        # ----------------------------------------------------------
        # Row 5: Inference Performance Bar Chart by Sentence Length Bin
        # ----------------------------------------------------------
        bin_times: Dict[str, List[float]] = {}
        bin_cat_counts: Dict[str, List[int]] = {}
        for sentence in val_sentences:
            if sentence is None:
                continue
            sentence_str = str(sentence)
            words = sentence_str.split()
            length = len(words)
            bin_label = self._get_bin_label(length)
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
        ax_perf.bar(x_bins - width_bar/2, avg_times, width_bar, label="Avg Inference Time (ms)", color="green", alpha=0.7)
        ax_perf.bar(x_bins + width_bar/2, avg_cats, width_bar, label="Avg Predicted Category Count", color="purple", alpha=0.7)
        ax_perf.set_xticks(x_bins)
        ax_perf.set_xticklabels(bins_sorted)
        ax_perf.set_xlabel("Sentence Length Bin", labelpad=10)
        ax_perf.set_title("Inference Performance Metrics by Sentence Length Bin", pad=20)
        ax_perf.legend()

        # ----------------------------------------------------------
        # Rows 6-9: Highlighted Samples Grid (8 samples max)
        # ----------------------------------------------------------
        max_highlights = HIGHLIGHT_ROWS * 2  # 2 columns per row
        highlight_samples = []
        for sent in val_sentences:
            sample_info = self._get_sample_by_sentence(sent, raw_data)
            if sample_info and sample_info.get("entities"):
                highlight_samples.append((sent, sample_info))
            if len(highlight_samples) >= max_highlights:
                break

        for idx in range(max_highlights):
            row_idx = TOP_CHART_ROWS + (idx // 2)
            col_idx = idx % 2
            # Allocate two columns per sample.
            if col_idx == 0:
                ax = fig.add_subplot(gs[row_idx, 0:2])
            else:
                ax = fig.add_subplot(gs[row_idx, 2:4])
            ax.axis("off")
            if idx < len(highlight_samples):
                sample_sentence, sample_info = highlight_samples[idx]
                sample_sentence = str(sample_sentence)
                # Get model prediction and corresponding predicted categories.
                pred = model.predict([sample_sentence])[0]
                pred_binary = (pred >= threshold).astype(int)
                predicted_categories = [categories[i] for i, val in enumerate(pred_binary) if val == 1]
                # Filter entities based on predicted categories.
                orig_entities = sample_info.get("entities", [])
                filtered_entities = [ent for ent in orig_entities if ent.get("category") in predicted_categories]
                if not filtered_entities:
                    filtered_entities = orig_entities
                filtered_entities = self._convert_entities(filtered_entities)
                pil_img = self._get_highlighted_text_image_wrapped(sample_sentence, filtered_entities)
                ax.imshow(np.array(pil_img))
                ax.set_title(f"Sample {idx + 1}", fontsize=12, pad=5)
            else:
                ax.text(0.5, 0.5, "No sample", horizontalalignment="center", verticalalignment="center")

        # ----------------------------------------------------------
        # Remaining Rows: Confusion Matrices for each Category
        # ----------------------------------------------------------
        for idx, cat in enumerate(categories):
            row_idx = base_conf + (idx // 4)
            col_idx = idx % 4
            ax_cm = fig.add_subplot(gs[row_idx, col_idx])
            cm = confusion_matrix(y_true[:, idx], y_pred_binary[:, idx])
            cm, TN, FP, FN, TP = self._standardize_confusion_matrix(cm)
            total = TN + FP + FN + TP
            acc = (TP + TN) / total if total > 0 else 0
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"], ax=ax_cm)
            ax_cm.set_title(f"{cat}", pad=10)
            ax_cm.text(0.5, -0.3, f"Acc: {acc:.2f}\nPrec: {prec:.2f}\nRec: {rec:.2f}",
                         fontsize=10, ha="center", transform=ax_cm.transAxes)

        # Turn off extra confusion matrix subplots if any.
        num_conf_plots = num_conf_rows * 4
        for extra in range(num_categories, num_conf_plots):
            row_idx = base_conf + (extra // 4)
            col_idx = extra % 4
            ax_empty = fig.add_subplot(gs[row_idx, col_idx])
            ax_empty.axis('off')

        # Adjust overall spacing.
        plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.4, wspace=0.5)

        combined_path = os.path.join(output_dir, "combined_post_training_charts.png")
        plt.savefig(combined_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Combined post-training charts saved to {combined_path}")
