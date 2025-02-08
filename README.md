# Synthetic Data Redaction & Generation

⭐️ If you like this project, please give it a star on GitHub! ⭐️

This repository contains a suite of tools for generating synthetic data useful for testing redaction and data anonymization pipelines. The project is designed to produce diverse and realistic data entities (such as names, addresses, passwords, and more) that are embedded within coherent sentences. It also integrates with Llama, a language model, to generate creative outputs, and includes utilities for validating and appending data entries.

## Key Components

- **Data Generators:**  
  Functions to produce synthetic data for various fields (e.g., People Name, Card Number, Social Security Number, etc.) found in [`redact/data_generators.py`](redact/data_generators.py).

- **Llama Integration:**  
  Integrates with the Llama language model to generate sentences incorporating the generated data, as seen in [`redact/llama_integration.py`](redact/llama_integration.py).

- **Synthetic Data Writer:**  
  An entry writer that appends generated data entries to a JSON file located at `O:\master_data_collection\redact\synthetic_data.json`, implemented in [`redact/synthetic_data.py`](redact/synthetic_data.py).

- **Main Execution Script:**  
  The driver script (`main.py`) which ties everything together by generating fields, building prompts, validating outputs, and appending entries to the synthetic data file.

- **Evaluation & Training Modules:**  
  Additional scripts like [`eval_data_balance.py`](eval_data_balance.py) and [`train.py`](train.py) for evaluating data distributions and training models on the synthetic dataset.

## How to Run

1. **Data Generation:**  
   Run the main script to start generating synthetic data entries:
   ```sh
   python main.py