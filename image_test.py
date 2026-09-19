import cv2
import csv
import re
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = r"runs\detect\SwachhSetu_Runs\waste_segregation_v1-2\weights\best.pt"
model = YOLO(MODEL_PATH)  # Load model

# Some more paths
INPUT_PATH = Path(r"SwachhSetu_dataset")
OUTPUT_FOLDER = Path(r"test_results")
OUTPUT_FOLDER.mkdir(exist_ok=True)
CSV_FILE_PATH = OUTPUT_FOLDER / "waste_classification_log.csv"

# Confidence threshold for classification
CONF_THRESHOLD = 0.10  # inc for sctrictness, dec for leniency

# for image overwrite prevention
def get_starting_index(folder_path):
    max_idx = 0
    for file in folder_path.glob("I*.*"):
        match = re.search(r"I(\d+)\.", file.name)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1

# Index search starts here
current_idx = get_starting_index(OUTPUT_FOLDER)

# Load images from SwachhSetu_dataset folder
if INPUT_PATH.is_file():
    image_files = [INPUT_PATH]
else:
    image_files = list(INPUT_PATH.glob("*.jpg")) + list(INPUT_PATH.glob("*.png")) + list(INPUT_PATH.glob("*.jpeg"))

print(f"Found {len(image_files)} test images.")
print(f"Starting incremental index from: I{current_idx}\n")

# CSV existence check
file_exists = CSV_FILE_PATH.exists()

# Open CSV file - append mode
with open(CSV_FILE_PATH, mode='a', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)

    # If new file, create header 
    if not file_exists:
        csv_writer.writerow(["Index_Name", "Original_Filename", "Final_Category", "Confidence_Score", "Detected_Boxes_Count"])

    # Read existing original filenames from CSV to skip re-testing
    processed_files = set()
    if CSV_FILE_PATH.exists():
        with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip CSV header
            for row in reader:
                if len(row) >= 2:
                    processed_files.add(row[1])  # Column index 1 is Original_Filename

        
    # image loop for checks
    for img_path in image_files[:10]:  # Adjust slice to process more/fewer images
        if img_path.name in processed_files:
            print(f"Skipping: {img_path.name:<20} (Already processed)") # no dobara annotation
            continue
        results = model.predict(str(img_path), conf=CONF_THRESHOLD, verbose=False)[0]

        # if "pta nahi" give that tag as mixed
        if len(results.boxes) == 0:
            final_category = "MIXED"
            confidence_str = "N/A"
        else:
            best_box = max(results.boxes, key=lambda b: float(b.conf[0]))
            cls_id = int(best_box.cls[0])
            confidence_val = float(best_box.conf[0])
            
            detected_class = model.names[cls_id]  # 'dry' or 'wet'
            final_category = detected_class.upper()
            confidence_str = f"{confidence_val:.2f}"

        # IP addressing for images
        save_name = f"I{current_idx}{img_path.suffix}"
        output_filepath = OUTPUT_FOLDER / save_name

        # I'm tired of giving proofs, khair, let's save these in CSV
        csv_writer.writerow([
            save_name,
            img_path.name,
            final_category,
            confidence_str,
            len(results.boxes)
        ])

        print(f"Saved: {save_name:<10} | Original: {img_path.name:<20} | Category: {final_category:<6} | Conf: {confidence_str}")

        # Ye thodi si drawing 
        annotated_img = results.plot()
        status_color = (0, 0, 255) if final_category == "MIXED" else (0, 255, 0)
        cv2.putText(
            annotated_img,
            f"{save_name} | Category: {final_category}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            2
        )

        # thode proof aur save kar lete hain
        cv2.imwrite(str(output_filepath), annotated_img)

        # pahele wale proof overwrite nahi hone chahiye
        current_idx += 1

print(f"\nCompleted!")
print(f"Annotated images saved in: {OUTPUT_FOLDER.resolve()}")
print(f"CSV records updated at: {CSV_FILE_PATH.resolve()}")

