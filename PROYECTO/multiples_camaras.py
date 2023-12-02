import cv2
import numpy as np
from deepface import DeepFace
import time
import Obtener_Caracteristicas as OC
import os
import reconocimiento_facial as RF

def multiples_camaras(num_camaras):

    cap_list = [cv2.VideoCapture(i) for i in range(num_camaras)]

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

    tiempo_inicial_dict = {i: None for i in range(num_camaras)}
    tiempo_limite = 1

    for camara_id in range(num_camaras):
        output_folder = f"Imagenes_camara_{camara_id}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    while True:
        for camara_id, cap in enumerate(cap_list):
            ret, frame = cap.read()

            if not ret or frame is None:
                print(f"No se pudo obtener un fotograma de la cámara {camara_id}")
                continue

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
                            if tiempo_inicial_dict[camara_id] is None:
                                tiempo_inicial_dict[camara_id] = time.time()
                    except (ValueError, IndexError):
                        pass

            if expresion_sospechosa:
                tiempo_actual = time.time()
                tiempo_transcurrido = tiempo_actual - tiempo_inicial_dict[camara_id]

                if tiempo_transcurrido >= tiempo_limite:
                
                    cv2.putText(frame, "ALERTA", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    timestamp = int(time.time())

                    output_folder = f"Imagenes_camara_{camara_id}"
                    output_filename = f"{output_folder}/rostro_analizado_{camara_id}_{timestamp}.jpg"

                    output_folder2 = "Imagenes"
                    output_filename2 = f"{output_folder2}/rostro_analizado_{camara_id}_{timestamp}.jpg"

                    cv2.imwrite(output_filename, roi2)
                    cv2.imwrite(output_filename2, roi2)

                    if RF.reconocimiento(output_filename, output_folder):
                        os.remove(output_filename)
                        os.remove(output_filename2)

                    RF.reconocimiento_facial(output_filename2)

                    tiempo_inicial_dict[camara_id] = None
            else:
                tiempo_inicial_dict[camara_id] = None

            cv2.imshow(f"Cámara {camara_id}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in cap_list:
        cap.release()
    cv2.destroyAllWindows()
