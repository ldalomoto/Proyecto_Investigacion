import cv2
import face_recognition
import os
import Obtener_Caracteristicas as OC
import shutil

def reconocimiento_facial(imag):
     try:
          nombre_archivo = os.path.basename(imag)
          image = cv2.imread(imag)
          face_loc = face_recognition.face_locations(image)[0]
          face_image_encodings = face_recognition.face_encodings(image, known_face_locations=[face_loc])[0]
          archivos_en_carpeta = os.listdir("Imagenes")
          extensiones_imagen = [".jpg", ".jpeg", ".png", ".bmp"]
          imagenes = [archivo for archivo in archivos_en_carpeta if any(archivo.lower().endswith(extension) for extension in extensiones_imagen)]

          if imagenes != []:
               for imagen in imagenes:
                    ruta_completa = os.path.join("Imagenes", imagen)
                    imagen2 = cv2.imread(ruta_completa)

                    face_location = face_recognition.face_locations(imagen2)[0]
                    face_imagen2_encodings = face_recognition.face_encodings(imagen2, known_face_locations=[face_location])[0]
                    result = face_recognition.compare_faces([face_image_encodings], face_imagen2_encodings)

                    if result[0] == True and imagen != nombre_archivo:
                         carpeta_sospechosos = "Fotos_sospechosos"
                         if not os.path.exists(carpeta_sospechosos):
                              os.makedirs(carpeta_sospechosos)

                         shutil.copy(imag, os.path.join(carpeta_sospechosos, f"imagen_sospechosa_{imagen}"))
                         OC.procesar(imag)
                         break
          else:
               print("archivo vacio")

     except Exception as e:
          print("No se pudo realizar el reconocimiento facial.")

def reconocimiento(imag, archivo):
     encontrado = False
     try:
          nombre_archivo = os.path.basename(imag)
          image = cv2.imread(imag)
          face_loc = face_recognition.face_locations(image)[0]
          face_image_encodings = face_recognition.face_encodings(image, known_face_locations=[face_loc])[0]

          archivos_en_carpeta = os.listdir(archivo)
          extensiones_imagen = [".jpg", ".jpeg", ".png", ".bmp"]
          imagenes = [archivo for archivo in archivos_en_carpeta if any(archivo.lower().endswith(extension) for extension in extensiones_imagen)]
          
          if imagenes != []:
               for imagen in imagenes:
                    ruta_completa = os.path.join(archivo, imagen)
                    imagen2 = cv2.imread(ruta_completa)

                    face_location = face_recognition.face_locations(imagen2)[0]
                    face_imagen2_encodings = face_recognition.face_encodings(imagen2, known_face_locations=[face_location])[0]
                    result = face_recognition.compare_faces([face_image_encodings], face_imagen2_encodings)

                    if result[0] == True:
                         encontrado = True
          else:
               print("archivo vacio")

          return encontrado  
        
     except Exception as e:
          print("No se pudo realizar el reconocimiento")

