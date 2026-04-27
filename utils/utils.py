import json
import pickle

def read_md_file(file_path):
    """
    Read the content of a Markdown file and return it as a string.

    Args:
        file_path (str): Path to the Markdown file.

    Returns:
        str: The entire content of the file. Returns None if the file does not exist or reading fails.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: File does not exist - {file_path}")
        return None
    except Exception as e:
        print(f"Error: Failed to read file - {e}")
        return None
    
def to_str(obj: dict, sign: bool) -> str:
    res = {}
    res["sentence"] = obj["sentence"]
    res["base form"] = obj["label"]
    if sign:
        res["prefer form"] = obj["prefer_label"]
    return  json.dumps(res)

def load_pickle(path):
    # Load pickle file and return the data
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data