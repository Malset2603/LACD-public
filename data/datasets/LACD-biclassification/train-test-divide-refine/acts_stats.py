import json
from collections import Counter
from tqdm import tqdm
import sys

from src.utils.utils import article_key_function

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

if __name__ == "__main__":
    # File paths
    file_paths = ["./data/datasets/LACD-biclassification/train-test-divide-refine/"+p for p in ["test.jsonl", "train.jsonl", "val.jsonl"]]

    # Set to hold unique elements
    unique_elements = set()

    # Set of pre86 laws
    pre86_laws = {f"형법 제{i}조" for i in range(1, 86)}

    # Process each file to extract the specified elements
    for file_path in file_paths:
        data = read_jsonl(file_path)
        for row in tqdm(data):
            unique_elements.add((article_key_function(row["article1"]), "".join(article_key_function(row["article1"]).split(" 제")[:-1])))
            unique_elements.add((article_key_function(row["article2"]), "".join(article_key_function(row["article2"]).split(" 제")[:-1])))

            law_id1 = article_key_function(row["article1"])
            law_id2 = article_key_function(row["article2"])
            # print(law_id1)
            if row.get("answer") and (law_id1 in pre86_laws or law_id2 in pre86_laws):
                print(row)

    # Count the distribution of law_id
    law_id_counter = Counter(element[1] for element in unique_elements)

    criminal_act_list = list(set(article_key_function(u[0]) for u in unique_elements if "형법" in u[1]))
    criminal_act_list.sort()

    # Calculate the threshold for "Others" grouping (5% of total)
    total_count = sum(law_id_counter.values())
    threshold = 0.02 * total_count

    # Separate items into "Others" if they fall below the threshold
    filtered_counter = Counter()
    others_count = 0

    for law_id, count in law_id_counter.items():
        if count >= threshold:
            filtered_counter[law_id] = count
        else:
            others_count += count

    # Add "Others" to the counter if there are any
    if others_count > 0:
        filtered_counter["Others"] = others_count

    # Sort by count in descending order
    sorted_law_id_distribution = filtered_counter.most_common()

    # Print the sorted distribution of law_id
    print("\nDistribution of law_id (with 'Others' for less frequent items):")
    for law_id, count in sorted_law_id_distribution:
        print(f"  {law_id}: {count} ({count*100/total_count})")

    print(total_count)
    print(len(law_id_counter))