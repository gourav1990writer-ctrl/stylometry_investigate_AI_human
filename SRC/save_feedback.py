import os
import csv
from datetime import datetime

def save_feedback(text, prediction, feedback):
    folder_path = "data/feedback"
    file_path = os.path.join(folder_path, "user_feedback.csv")

    # Make sure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Check if file already exists
    file_exists = os.path.isfile(file_path)

    # Open file in append mode
    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header only if file is new
        if not file_exists:
            writer.writerow(["timestamp", "text", "prediction", "feedback"])

        # Write one feedback row
        writer.writerow([
            datetime.now().isoformat(),
            text,
            prediction,
            feedback
        ])