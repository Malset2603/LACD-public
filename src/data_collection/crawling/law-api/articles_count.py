import os
import pandas as pd
from src.preprocessing.utils import is_law, generate_law_list
import csv

if __name__ == "__main__":

    def count_rows(directory, law_list):
        total_rows = 0
        for source_law in os.listdir(directory):
            if source_law.endswith('.csv') and is_law(law_list, source_law[:-4]):
            # if source_law.endswith('.csv'):
                file_path = os.path.join(directory, source_law)
                df = pd.read_csv(file_path)
                row_count = len(df)
                total_rows += row_count
        # Subtracting the number of CSV files to account for the header row in each file
        total_rows -= len([f for f in os.listdir(directory) if (f.endswith('.csv') and is_law(law_list, f[:-4]))])
        # total_rows -= len([f for f in os.listdir(directory) if (f.endswith('.csv'))])
        return total_rows


    directory_path = './data/database/laws'
    # Replace with your directory path



    # law list 생셩
    law_list = generate_law_list()

    total_rows = count_rows(directory_path, law_list)
    print(f"Total number of rows (excluding headers) in all CSV files: {total_rows}")
