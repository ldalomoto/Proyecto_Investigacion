import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

emociones_traducciones = {
    'angry': 'enojado',
    'disgust': 'disgustado',
    'fear': 'miedoso',
    'happy': 'feliz',
    'sad': 'triste',
    'surprise': 'sorprendido',
    'neutral': 'neutral'
}

while True:
    
    ret, frame = cap.read()
    frame_resized = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(frame_resized, 1.0, (300, 300), (104, 117, 123))
    net.setInput(blob)
    detections = net.forward()

    for detection in detections[0, 0]:
        confidence = detection[2]

        if confidence > 0.5:
            box = detection[3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x_start, y_start, x_end, y_end) = box.astype("int")
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            label = f"Confianza: {confidence * 100:.2f}%"
            cv2.putText(frame, label, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            roi = frame[y_start:y_end, x_start:x_end]

            try:
                info = DeepFace.analyze(roi, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
                emociones = info[0]['dominant_emotion']
                emociones = emociones_traducciones.get(emociones, emociones)
                cv2.putText(frame, emociones, (x_start, y_start - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            except ValueError as e:
                print(f"Error: {e}")
            except IndexError:
                print("No se pudo acceder a la información de emociones. Asegúrate de que se detecte un rostro.")

    cv2.imshow("Detección y Emociones en Tiempo Real", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
