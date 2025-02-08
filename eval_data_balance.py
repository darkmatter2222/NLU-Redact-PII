import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_bin_label(sentence_length, bin_size=5):
    """
    Returns a bin label (as a string) for a given sentence length.
    For example, a sentence with 12 words with bin_size=5 will fall into "10-14".
    """
    low = (sentence_length // bin_size) * bin_size
    high = low + bin_size - 1
    return f"{low}-{high}"

def create_heatmap_from_json(json_file, bin_size=5):
    # Load JSON data from file.
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Initialize a dictionary to accumulate counts:
    #   { bin_label: { entity_category: count, ... }, ... }
    counts = {}
    
    for sample in data:
        sentence = sample.get("sentence", "")
        # Compute sentence length as number of words.
        words = sentence.split()
        length = len(words)
        bin_label = get_bin_label(length, bin_size)
        
        # Ensure there is an entry for this sentence length bin.
        if bin_label not in counts:
            counts[bin_label] = {}
        
        # Iterate over the entities in the sample.
        for entity in sample.get("entities", []):
            category = entity.get("category", "Unknown")
            counts[bin_label][category] = counts[bin_label].get(category, 0) + 1
    
    # Convert the nested dictionary into a DataFrame.
    df = pd.DataFrame.from_dict(counts, orient="index").fillna(0)
    
    # Sort the bins in increasing order based on the lower bound of the bin.
    # The index labels are strings like "0-4", "5-9", etc.
    df.index = pd.CategoricalIndex(
        df.index, 
        categories=sorted(df.index, key=lambda x: int(x.split('-')[0])), 
        ordered=True
    )
    df = df.sort_index()
    
    # Create the heatmap.
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, fmt=".0f", cmap="viridis")
    plt.xlabel("Entity Category")
    plt.ylabel("Sentence Length Bins (words)")
    plt.title("Heatmap: Entity Category vs. Binned Sentence Length")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    json_file = r"O:\master_data_collection\redact\synthetic_data.json"  # Replace with the path to your JSON file.
    create_heatmap_from_json(json_file)
