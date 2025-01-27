import fileinput
import json 
import os

file_path = "sample_data.jsonl"


# Function to split JSONL into separate JSON files
def split_jsonl_to_json(jsonl_file, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    with open(jsonl_file, 'r', encoding='utf-8') as infile:
        for index, line in enumerate(infile):
            try:
                data = json.loads(line)  # Parse JSON from each line
                output_file = os.path.join(output_folder, data["id"] + '.json')

                with open(output_file, 'w', encoding='utf-8') as outfile:
                    json.dump(data, outfile, indent=4)  # Save JSON with pretty formatting

                print(f"Saved: {output_file}")
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at line {index + 1}")


jsonl_file = ".rawdatasets/arxiv-metadata-oai-snapshot.jsonl"  
output_folder = "arXiv_metadata"  

split_jsonl_to_json(jsonl_file, output_folder)