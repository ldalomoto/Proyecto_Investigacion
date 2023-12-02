import multiples_camaras as mc
import camara_unica as tm
import cv2

def obtener_camaras_disponibles():
    total_camaras = 0
    for indice_camara in range(10):
        try:
            cap = cv2.VideoCapture(indice_camara)
            if not cap.isOpened():
                break
            cap.release()
            total_camaras += 1 
        except cv2.error:
            pass

    if total_camaras == 1:
        tm.camara_unica(0)
    else:
        mc.multiples_camaras(total_camaras)

obtener_camaras_disponibles()

