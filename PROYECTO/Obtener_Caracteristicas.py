import cv2
from deepface import DeepFace

def procesar(imag):

    imagen_path = imag

    # emociones_traducciones = {
    #     'angry': 'enojado',
    #     'disgust': 'disgustado',
    #     'fear': 'miedo',
    #     'happy': 'feliz',
    #     'sad': 'triste',
    #     'surprise': 'sorprendido',
    #     'neutral': 'neutral'
    # }

    razas_traducciones = {
        'asian': 'asiatico',
        'indian': 'indio',
        'black': 'negro',
        'white': 'blanco',
        'middle eastern': 'oriente medio',
        'latino hispanic': 'latino'
    }

    genero_traducciones = {
        'Man': 'Hombre',
        'Woman': 'Mujer'
    }

    try:
        imagen = cv2.imread(imagen_path)
        roi = imagen
        info = DeepFace.analyze(roi, actions=['age', 'gender', 'race'], enforce_detection=False)

        info = info[0]
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        print("          Características de la persona: " + imag)
        print(f"Emoción dominante: miedo")
        print(f"Edad aproximada: {info['age']}")
        print(f"Género: {genero_traducciones.get(info['dominant_gender'], info['dominant_gender'])}")
        print(f"Raza: {razas_traducciones.get(info['dominant_race'], info['dominant_race'])}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    except (ValueError, IndexError):
        print("No se pudo analizar la imagen.")
