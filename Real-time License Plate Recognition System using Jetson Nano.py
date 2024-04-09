import cv2
import pytesseract
net = cv2.dnn.readNet('path_to_model')
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur
def detect_license_plate(image):
    processed_image = preprocess_image(image)
    roi = processed_image[100:300, 100:500]
    blob = cv2.dnn.blobFromImage(roi, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    confidences = []
    boxes = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = detection[0:4] * np.array([400, 200, 400, 200]) + np.array([100, 100, 100, 100])
                (x, y, w, h) = box.astype("int")
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    recognized_text = ""

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            plate = roi[y:y+h, x:x+w]
            recognized_text += pytesseract.image_to_string(plate)

    return recognized_text
def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        cv2.imshow('Frame', frame)
        recognized_plate = detect_license_plate(frame)
        print("Recognized License Plate:", recognized_plate)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
