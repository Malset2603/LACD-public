import os
import time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd


def extract_metrics_from_tfevent(file_path):
    event_acc = EventAccumulator(file_path)
    event_acc.Reload()

    # Extracting scalar values for eval/f1, eval/roc_auc, eval/accuracy
    f1_values = event_acc.Scalars('test/f1')
    roc_auc_values = event_acc.Scalars('test/roc_auc')
    accuracy_values = event_acc.Scalars('test/accuracy')

    # Finding the step with the maximum F1 score
    max_f1_step = max(f1_values, key=lambda x: x.value).step

    # Extracting metrics at the step with the maximum F1 score
    max_f1 = round(next(x.value for x in f1_values if x.step == max_f1_step), 3)
    max_roc_auc = round(next(x.value for x in roc_auc_values if x.step == max_f1_step), 3)
    max_accuracy = round(next(x.value for x in accuracy_values if x.step == max_f1_step), 3)

    return max_f1_step, max_f1, max_roc_auc, max_accuracy

def find_and_display_metrics(root_dir):
    metrics_list = []

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.startswith("events.out.tfevents") and file[-1] != "2":
                try:
                    file_path = os.path.join(subdir, file)
                    step, f1, roc_auc, accuracy = extract_metrics_from_tfevent(file_path)

                    # Get the creation time of the file
                    creation_time = os.path.getctime(file_path)
                    creation_time_readable = time.ctime(creation_time)

                    metrics_list.append({
                        'Directory': subdir.split("/")[-1],
                        'Step': step,
                        'Accuracy(%)': accuracy * 100,
                        'F1(%)': f1 * 100,
                        'ROC AUC': roc_auc,
                        'File': file,
                        'Creation Time': creation_time_readable
                    })
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue

    # Convert list of metrics to DataFrame for display
    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df


if __name__ == "__main__":
    # Example usage
    root_dir = './outputs/LACD-cross/'
    metrics_df = find_and_display_metrics(root_dir)

    # Display the DataFrame
    print(metrics_df)