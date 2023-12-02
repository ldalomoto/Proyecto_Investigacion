import cv2
import numpy as np
from deepface import DeepFace
import time
import Obtener_Caracteristicas as OC
import os

def camara_unica(indice):

    cap = cv2.VideoCapture(indice)
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

    tiempo_inicial = None
    tiempo_limite = 3

    while True:
        ret, frame = cap.read()
        frame_resized = cv2.resize(frame, (300, 300))
        frame_copy = frame.copy()

        blob = cv2.dnn.blobFromImage(frame_resized, 1.0, (300, 300), (104, 117, 123))
        net.setInput(blob)
        detections = net.forward()

        expresion_sospechosa = False

        for detection in detections[0, 0]:
            confidence, x_start, y_start, x_end, y_end = detection[2], *map(int, detection[3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]))

            if confidence > 0.5:
                box = detection[3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x_start, y_start, x_end, y_end) = box.astype("int")

                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                label = f"Confianza: {confidence * 100:.2f}%"
                cv2.putText(frame, label, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                roi = frame[y_start:y_end, x_start:x_end]
                roi2 = frame_copy[y_start-35:y_end+35, x_start-35:x_end+35]

                try:
                    info = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)
                    emociones = emociones_traducciones.get(info[0]['dominant_emotion'], info[0]['dominant_emotion'])
                    cv2.putText(frame, emociones, (x_start, y_start - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if emociones == "miedo":
                        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
                        cv2.putText(frame, emociones, (x_start, y_start - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        label = f"Confianza: {confidence * 100:.2f}%"
                        cv2.putText(frame, label, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        expresion_sospechosa = True
                        if tiempo_inicial is None:
                            tiempo_inicial = time.time()
                except (ValueError, IndexError):
                    pass

        if expresion_sospechosa:
            tiempo_actual = time.time()
            tiempo_transcurrido = tiempo_actual - tiempo_inicial

            if tiempo_transcurrido >= tiempo_limite:
                
                cv2.putText(frame, "ALERTA", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                output_folder = "Fotos_sospechosos"
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                timestamp = int(time.time())
                output_filename = f"{output_folder}/rostro_analizado_{timestamp}.jpg"

                cv2.imwrite(output_filename, roi2)
                
                OC.procesar(output_filename)

                tiempo_inicial = None
        else:
            tiempo_inicial = None

        cv2.imshow("Deteccion y Emociones en Tiempo Real", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
