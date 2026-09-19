from ultralytics import YOLO

def train_model():
    # Load pre-trained YOLOv8 small weights
    model = YOLO("yolov8s.pt")

    # Start training
    results = model.train(
        data="./merged_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,  # Use GPU 0 (set to 'cpu' if no GPU available)
        project="SwachhSetu_Runs",
        name="waste_segregation_v1"
    )

if __name__ == "__main__":
    train_model()

    