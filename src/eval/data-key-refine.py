import os
import json
import argparse
from tqdm import tqdm
from src.utils.utils import article_key_function

def main(input_dir, output_dir):
    # Get all filenames in input_dir
    file_names = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    for f in file_names:
        try:
            input_path = os.path.join(input_dir, f)
            output_path = os.path.join(output_dir, f)

            processed_rows = []

            # Read input file
            with open(input_path, "r", encoding="utf-8") as infile:
                for line in tqdm(infile):
                    line = line.strip()
                    if not line:
                        continue
                    # Parse JSON line by line
                    data = json.loads(line)
                    
                    # Apply article_key_function to article_to_check
                    data["article_to_check"] = article_key_function(data["article_to_check"])
                    # Apply article_key_function to each element in articles list
                    data["articles"] = [article_key_function(article) for article in data.get("articles", [])]

                    processed_rows.append(data)

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Save results to JSONL file
            with open(output_path, "w", encoding="utf-8") as outfile:
                for row in processed_rows:
                    json_line = json.dumps(row, ensure_ascii=False)
                    outfile.write(json_line + "\n")
        except:
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./outputs/retrieval_results/article_key",
                        help="Input directory path")
    parser.add_argument("--output_dir", type=str, default="./outputs/retrieval_results/article_key",
                        help="Output directory path") 
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir)
