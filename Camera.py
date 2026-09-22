import cv2
from ultralytics import YOLO
from pymodbus.client import ModbusTcpClient

# Initialize model
model = YOLO(r"runs\detect\SwachhSetu_Runs\waste_segregation_v1-2\weights\best.pt")

# Connect to PLC via Ethernet (Modbus TCP)
# Update IP address according to your PLC configuration
plc_client = ModbusTcpClient('192.168.0.10', port=502)
plc_connected = plc_client.connect() 
print("PLC Connection Status:", plc_connected)

CONF_THRESHOLD = 0.10  # Detection confidence threshold
cap = cv2.VideoCapture(0)  # USB Camera

def send_to_plc(status_code):
    # 1 = Dry, 2 = Wet, 0 = Mixed (Default/Low Confidence)
    if plc_connected:
        plc_client.write_register(address=100, value=status_code)

while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
    detected_category = "mixed"
    status_code = 0  # Default to mixed

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        if class_name == "dry":
            detected_category = "dry"
            status_code = 1
            break
        elif class_name == "wet":
            detected_category = "wet"
            status_code = 2
            break

    send_to_plc(status_code)

    # Draw overlays
    annotated_frame = results.plot()
    cv2.putText(annotated_frame, f"PLC Signal: {detected_category.upper()}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("SwachhSetu Conveyor Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if plc_connected:
    plc_client.close()
    