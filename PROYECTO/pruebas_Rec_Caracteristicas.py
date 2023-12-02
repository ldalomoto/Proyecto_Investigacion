import cv2
import numpy as np
from deepface import DeepFace

cap = cv2.VideoCapture(0)
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

emociones_traducciones = {
    'angry': 'enojado',
    'disgust': 'disgustado',
    'fear': 'miedo',
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
        confidence, x_start, y_start, x_end, y_end = detection[2], *map(int, detection[3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]))

        if confidence > 0.5:
      
            box = detection[3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x_start, y_start, x_end, y_end) = box.astype("int")

            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            label = f"Confianza: {confidence * 100:.2f}%"
            cv2.putText(frame, label, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            roi = frame[y_start:y_end, x_start:x_end]

            try:
                info = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)
                emociones = emociones_traducciones.get(info[0]['dominant_emotion'], info[0]['dominant_emotion'])
                cv2.putText(frame, emociones, (x_start, y_start - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if str(emociones) == "miedo":

                    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
                    alerta_texto = "ALERTA"
                    text_size = cv2.getTextSize(alerta_texto, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = (frame.shape[0] + text_size[1]) // 2
                    cv2.putText(frame, alerta_texto, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, emociones, (x_start, y_start - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    label = f"Confianza: {confidence * 100:.2f}%"
                    cv2.putText(frame, label, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        #info2 = DeepFace.analyze(roi, actions=['age', 'gender', 'race'], enforce_detection=False)

                        #info2_dict = info2[0]
                        #edad = info2_dict['age']
                        #race = info2_dict['dominant_race']
                        #gen = info2_dict['dominant_gender']

                        #print(f"""
                        #                            ALERTA
                        #    genero: {str(gen)}
                        #    edad aproximada: {str(edad)}
                        #    color de piel: {str(race)}

                        #    """)

            except (ValueError, IndexError):
                pass

    cv2.imshow("Detección y Emociones en Tiempo Real", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
