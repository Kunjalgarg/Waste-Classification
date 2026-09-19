import cv2
import os
from pathlib import Path
from ultralytics import YOLO

# 1. Load your trained model
MODEL_PATH = r"runs\detect\SwachhSetu_Runs\waste_segregation_v1-2\weights\best.pt"
model = YOLO(MODEL_PATH)

# 2. Input/Output configuration
# Can point to a single image or an entire folder of images
INPUT_PATH = Path(r"merged_dataset\val\images")
OUTPUT_FOLDER = Path(r"test_results")
OUTPUT_FOLDER.mkdir(exist_ok=True)

CONF_THRESHOLD = 0.50  # Confidence threshold for classification

# Get list of images
if INPUT_PATH.is_file():
    image_files = [INPUT_PATH]
else:
    image_files = list(INPUT_PATH.glob("*.jpg")) + list(INPUT_PATH.glob("*.png"))

print(f"Found {len(image_files)} images to process...")

# 3. Inference Loop
for img_path in image_files[:10]:  # Change index range to test more images
    # Run inference
    results = model.predict(str(img_path), conf=CONF_THRESHOLD, verbose=False)[0]

    detected_label = "mixed"
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        print(f"Image: {img_path.name} -> Detected: {label} (Conf: {conf:.2f})")
        detected_label = label

    if len(results.boxes) == 0:
        print(f"Image: {img_path.name} -> No high-confidence detection. Classified as: MIXED")

    # Save output image with visual bounding boxes
    annotated_img = results.plot()
    output_filepath = OUTPUT_FOLDER / f"result_{img_path.name}"
    cv2.imwrite(str(output_filepath), annotated_img)

print(f"\nProcessing complete! Visual results saved in: {OUTPUT_FOLDER.resolve()}")
