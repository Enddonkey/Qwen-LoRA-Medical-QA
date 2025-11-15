import json
import os
import glob

def clean_response(text):
    """
    Remove content after </s> marker in a text string.
    
    Args:
        text (str): The text to clean
        
    Returns:
        str: Text with content after </s> removed
    """
    if isinstance(text, str) and '</s>' in text:
        # Find the position of </s> and truncate everything after it
        end_pos = text.find('</s>')
        return text[:end_pos]
    return text

def clean_json_data(data):
    """
    Recursively clean JSON data by removing content after </s> markers.
    
    Args:
        data: JSON data (dict, list, or primitive type)
        
    Returns:
        Cleaned JSON data
    """
    if isinstance(data, dict):
        # Process each key-value pair in dictionary
        return {key: clean_json_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Process each item in list
        return [clean_json_data(item) for item in data]
    elif isinstance(data, str):
        # Clean string content
        return clean_response(data)
    else:
        # Return primitive types as-is
        return data

def process_json_file(filepath):
    """
    Read, clean, and write back a JSON file.
    
    Args:
        filepath (str): Path to the JSON file
    """
    try:
        # Read the JSON file
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Processing: {filepath}")
        
        # Clean the data
        cleaned_data = clean_json_data(data)
        
        # Write back to the same file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Successfully cleaned: {filepath}")
        
    except json.JSONDecodeError as e:
        print(f"✗ Error decoding JSON in {filepath}: {e}")
    except Exception as e:
        print(f"✗ Error processing {filepath}: {e}")

def main():
    """
    Main function to process all JSON files in the evaluate folder.
    """
    # Define the evaluate folder path
    evaluate_folder = 'evaluate'
    
    # Check if evaluate folder exists
    if not os.path.exists(evaluate_folder):
        print(f"Error: '{evaluate_folder}' folder not found!")
        return
    
    # Find all JSON files in the evaluate folder
    json_files = glob.glob(os.path.join(evaluate_folder, '*.json'))
    
    if not json_files:
        print(f"No JSON files found in '{evaluate_folder}' folder.")
        return
    
    print(f"Found {len(json_files)} JSON file(s) to process.\n")
    
    # Process each JSON file
    for json_file in json_files:
        process_json_file(json_file)
        print()  # Add blank line between files
    
    print("All files processed!")

if __name__ == "__main__":
    main()
