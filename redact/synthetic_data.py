import os
import json

class SyntheticDataWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        # Ensure the directory exists.
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def append_entry(self, entry):
        """
        Appends the given entry (a dictionary) to the JSON file.
        If the file exists and contains a JSON array, the entry is appended;
        otherwise, a new array is created.
        """
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except Exception:
                data = []
        else:
            data = []
        data.append(entry)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
