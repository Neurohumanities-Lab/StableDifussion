import os
#import face_recognition
import numpy as np
import time
import openai
import torch
import csv
#import random
import tempfile
#import matplotlib.pyplot as plt
import cv2
import sys
#import mediapipe as mp
import PIL
import argparse
import pandas as pd
import subprocess
import shutil
#from collections import deque
#from statistics import mode
#from matplotlib.animation import FuncAnimation
from datetime import datetime
from feat import Detector, utils
#from feat.utils.io import get_test_data_path
#from feat.plotting import imshow
from PIL import Image
#from audiocraft.models import MusicGen
#from audiocraft.utils.notebook import display_audio
#from audiocraft.data.audio import audio_write
#from diffusers import StableDiffusionInpaintPipeline
#from diffusers import StableDiffusionImageVariationPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from torch import autocast

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, default="emociones.csv", help='Nombre del archivo CSV')
parser.add_argument('--video', type=str, help='Nombre del archivo de video')
parser.add_argument('--gpt_prompts', type=str,default="false", help='Forma en que se generan los prompts')
parser.add_argument('--record', type=str, default="false", help='Grabar video: true o false')
parser.add_argument('--century', type=int, default=16, help='Definir siglo a adaptar')
args = parser.parse_args()

nombre_archivo = os.path.join('csv', args.csv)
#century = args.century

if os.path.isfile(nombre_archivo):
    with open(nombre_archivo, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)
        if data:
            last_row = data[-1]
            num_test = int(last_row[0]) + 1
        else:
            num_test = 1
else:
    num_test = 1


utils.set_torch_device(device='cuda')


emotions = ""
emocion_objetivo = ""
emociones = []
emocionFrec = ""

img_path_clock = os.path.join("imagenes","new_image_clock.jpg")
img_path_vase = os.path.join("imagenes","new_image_vase.jpg")
orig_path_clock = os.path.join("imagenes","originals","clock_black.jpg")
orig_path_vase =  os.path.join("imagenes","originals","vase1.jpg")

shutil.copy(orig_path_clock,img_path_clock)
shutil.copy(orig_path_vase,img_path_vase)


detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="svm",
        facepose_model="img2pose",
        device="cuda"
)

def crear_carpetas():
    if not os.path.exists('imagenes'):
        os.makedirs('imagenes')
    if not os.path.exists('csv'):
        os.makedirs('csv')

def init_camera():
    video_capture = cv2.VideoCapture(0)
    ret = video_capture.set(3,640)
    ret = video_capture.set(4,480)
    return video_capture

def acquire_image(video_capture, max_attempts=3):
    attempts = 0

    while attempts < max_attempts:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        
        if ret:
            scaled_rgb_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            scaled_rgb_frame = np.ascontiguousarray(scaled_rgb_frame[:, :, ::-1])
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, "temp_frame.jpg")
            cv2.imwrite(temp_file, scaled_rgb_frame)
            return frame, scaled_rgb_frame, temp_file
        else:
            attempts += 1

    print("--------No se pudo capturar la imagen / Fin del video------")
    return None, None, None

def show_frame(frame):
    # Display the resulting image frame in the PAC
    cv2.imshow('Video',frame)

def find_face_emotion(frame):
    single_face_prediction = detector.detect_image(frame)
    data = single_face_prediction
    df = single_face_prediction.emotions
    if len(df) == 1 and df.isnull().all().all():
        emotion_list = []
    else:
        dict = df.idxmax(axis=1).to_dict()
        emotion_list = list(dict.values())
    return emotion_list, data

def save_data(emociones,dataframe=None):
    #Verificar si las listas están vacías
    if not emociones:
        return
    #Obtener la fecha y hora actual
    fecha_actual = datetime.now().strftime("%Y-%m-%d")
    hora_actual = datetime.now().strftime("%H:%M:%S")

    #Abrir el archivo CSV en modo de agregado ('a')
    with open(nombre_archivo, 'a', newline='') as archivo_csv:
        writer = csv.writer(archivo_csv)

        #Escribir cada elemento en una nueva fila
        for emocion in emociones:
            if dataframe is not None:
                data_row = [num_test, fecha_actual, hora_actual, emocion]
                data_row += list(dataframe.iloc[0]) #Agregar datos del DataFrame
                writer.writerow(data_row)
            else:
                writer.writerow([fecha_actual,hora_actual,emocion])
    print("--------Datos guardados correctamente------------")

def load_image(file_path):
    return PIL.Image.open(file_path).convert("RGB")

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def infer(prompt, init_image, strength, img_path):
    #base_generated_folder = os.path.join("imagenes")
    #generated_image_name = f"Result_image.jpg"
    generated_image_path = img_path
    if init_image != None:
        init_image = init_image.resize((512, 512))
        init_image = preprocess(init_image)
        with autocast("cuda"):
            images = pipeimg(prompt=prompt, image=init_image, strength=strength, guidance_scale=7.5).images[0]#["sample"]
    else: 
        pass
    
    images.save(generated_image_path)
    subprocess.Popen(["start", generated_image_path],shell=True)
    return images

def generate_inpainting_prompt(emotion, element):
    emotion_values = {
        "anger": (-1, -1, -1),
        "fear": (-1, 1, -1),
        "disgust": (None, None, None),
        "happiness": (1, 1, 0),
        "sadness": (-1, -1, -1),
        "surprise": (0, 1, -1),
        "neutral": (0, 0, 0) 
    }
     # Create a dictionary with modifications for each element based on valence, arousal, and dominance values
    element_modifications = {
        "flower": {
            (-1, -1, -1): "Darker colors. The flower withers and shrinks. Petals and leaves harden. More pointed shapes and thorns.",
            (-1, 1, -1): "Darker colors. The flower blooms and grows. Petals and leaves harden. More pointed shapes and thorns.",
            (None, None, None): "Dull colors. The flower remains the same. Petals and leaves wrinkle. Irregular shapes.",
            (1, 1, 0): "Brighter colors. The flower blooms and grows. Petals and leaves soften. Rounded and smooth shapes.",
            (-1, -1, -1): "Darker colors. The flower withers and shrinks. Petals and leaves harden. More pointed shapes and thorns.",
            (0, 1, -1): "Varied colors. The flower blooms and grows. Petals and leaves harden. Unexpected and surprising shapes.",
            (0, 0, 0): "Neutral colors. The flower remains unchanged. No significant changes in shape or size."  # Prompt for "neutral" emotion
        },
        "hourglass": {
            (-1, -1, -1): "Darker colors. The hourglass becomes older, deteriorates. Time passes more slowly.",
            (-1, 1, -1): "Darker colors. The hourglass becomes older, deteriorates. Time passes more quickly.",
            (None, None, None): "Dull colors. The hourglass remains the same. Time passes randomly.",
            (1, 1, 0): "Brighter colors. The hourglass becomes more modern, new, and shiny. Time passes at the desired pace.",
            (-1, -1, -1): "Darker colors. The hourglass becomes older, deteriorates. Time passes more slowly.",
            (0, 1, -1): "Varied colors. The hourglass becomes randomly more modern or older. Time passes unpredictably.",
            (0, 0, 0): "Neutral colors. The hourglass remains unchanged. Time stands still."  # Prompt for "neutral" emotion
        }
    }
    #Get valence, arousal, and dominance values for the given emotion
    valence, arousal, dominance = emotion_values[emotion]

    #Get the modification for the given element based on valence, arousal, and dominance values
    modification = element_modifications[element][(valence,arousal,dominance)]

    #Create the prompt using the modification
    prompt = f"{modification}"

    return prompt
    
def generate_elements_dict(archivo_csv,mode):
    elements = {}
    columns = ['sadness', 'neutral', 'fear', 'happiness', 'surprise', 'anger', 'disgust']
    initial_df = pd.DataFrame(columns=columns, index=range(10))
    with open(archivo_csv , 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if len(row) == 0:
                break
            
            if len(row) >= 3:
                key = row[1]
                value = row[2]
                if (mode == 1):
                    elements[key] = {'value': value, 'count': 0,'dataframe': initial_df.copy()}
                elif(mode == 2):
                    elements[key] = {'value': value, 'count': 0,'prompt': ""}
    if not elements:
        return None 
    return elements

pipeimg = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16,
).to("cuda")

elements_dict = generate_elements_dict('descriptions.csv',2)

crear_carpetas()
video_capture = init_camera()

try:
    while (True):
        #############################################################
        # SENSING LAYER
        rgb_frame, scaled_rgb_frame, temp_file = acquire_image(video_capture)
        if rgb_frame is None:
            break
        #Emotion recognition
        face_emotions,data = find_face_emotion(temp_file)
        print(face_emotions[0])
        if len(emociones) < 10:
            emociones.append(face_emotions[0])
            print(emociones)
            emocionFrec = None
        else:
            emocionFrec = str(max(emociones, key=emociones.count))
            emociones=[]

        if emocionFrec is not None:
            print("La emoción más frecuente es: " + emocionFrec)
            image_prompt1 = generate_inpainting_prompt(emocionFrec,'hourglass')
            image_prompt2 = generate_inpainting_prompt(emocionFrec,'flower')
            print(image_prompt1)
            print(image_prompt2)
            im_vase = Image.open(r"C:\Users\Neurohumanities\Documents\Python_Scripts\StableDifussion\imagenes\new_image_vase.jpg")
            im_clock = Image.open(r"C:\Users\Neurohumanities\Documents\Python_Scripts\StableDifussion\imagenes\new_image_clock.jpg")
            images = infer(image_prompt1, im_clock, 0.5, img_path_clock)
            images = infer(image_prompt2, im_vase, 0.5, img_path_vase)

            
        else:
            pass     
        
        show_frame(rgb_frame)
        #############################################################

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # END OF THE GAME/LIFE
            break
        lastPublication = time.time()
        
except KeyboardInterrupt:

    # Cerrar las ventanas de OpenCV
    cv2.destroyAllWindows()



cv2.destroyAllWindows()