import csv
import os
from collections import OrderedDict


def save_metrics_to_csv(output_dir: str, filename: str, metrics_dict: OrderedDict):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    file_exists = os.path.isfile(filepath)

    with open(filepath, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics_dict)
