import glob
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Temporary fix for deprecated numpy int
np.int = int

# ---------------- CONFIG ----------------
IMAGE_PATH = r"anpr.jpg"   # change if needed
MODEL_PATH = r"alpnr_best.pt"                       # model not pushed to git
CONF_THRESHOLD = 0.6
# ----------------------------------------

# Load models
model = YOLO(MODEL_PATH)
ocr = PaddleOCR(use_angle_cls=True, lang="en")

for path in glob.glob(IMAGE_PATH):
    print(f"Processing: {path}")
    image = cv2.imread(path)

    results = model(image, verbose=False)

    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf

        for i in range(len(boxes)):
            if confs[i] < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, boxes[i])

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Crop license plate
            crop = image[y1:y2, x1:x2]

            # OCR
            ocr_result = ocr.ocr(crop, cls=True)
            print("OCR Raw Output:", ocr_result)

            plate_text = []
            for line in ocr_result:
                text = line[1][0]
                if text == "IND":
                    continue
                plate_text.append(text)

            number_plate = "".join(plate_text)
            print("Detected Number Plate:", number_plate)

            cv2.imshow("Number Plate", crop)
            cv2.waitKey(0)

cv2.destroyAllWindows()
