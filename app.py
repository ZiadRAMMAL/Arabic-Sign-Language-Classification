# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:58:21 2026

@author: Ziad RAMMAL et Melissa YESGUER
"""

from __future__ import annotations
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import mediapipe as mp
from tensorflow.keras.models import model_from_json, Sequential
import tensorflow as tf

# Importation des modules pour la nouvelle API MediaPipe Tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os
import gdown

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="path/to/hand_landmarker.task",
        delegate=BaseOptions.Delegate.CPU  # <-- Explicitly set CPU delegate
    ),
    running_mode=RunningMode.IMAGE
)

# Function to download models from Google Drive if not present locally
@st.cache_resource
def download_model_files():
    h5_file_id = "1KFUQ7b4LW7HGwr2eHirdWAIgTmGKxoxj"
    json_file_id = "1jX7dpkWKsh6hP2tWgeiOnLmy4Xr-_Jt8"
    
    h5_url = f"https://drive.google.com/uc?id={h5_file_id}"
    json_url = f"https://drive.google.com/uc?id={json_file_id}"
    
    if not os.path.exists("model.h5"):
        with st.spinner("Downloading model weight file (model.h5)..."):
            gdown.download(h5_url, "model.h5", quiet=False)
            
    if not os.path.exists("model.json"):
        with st.spinner("Downloading model structure file (model.json)..."):
            gdown.download(json_url, "model.json", quiet=False)

# Execute the download before loading the model
download_model_files()

# --- 1. INITIALISATION ET MÉMOIRE ---

if 'phrase' not in st.session_state:
    st.session_state.phrase = ""

# Configuration du nouveau détecteur MediaPipe Tasks (HandLandmarker)
# Assurez-vous d'avoir téléchargé le fichier de modèle 'hand_landmarker.task' 
# et de l'avoir placé dans le même dossier que votre script app.py.
@st.cache_resource
def load_detector():
    base_options = python.BaseOptions(
        model_asset_path='hand_landmarker.task',
        delegate=python.BaseOptions.Delegate.CPU  # <-- Forcer l'exécution CPU sur le serveur
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.IMAGE
    )
    return vision.HandLandmarker.create_from_options(options)

detector = load_detector()

nom_classe = [
    "Ain", "Al", "Alef", "Beh", "Dad", "Dal", "Feh", "Ghain", "Hah", "Heh",
    "Jeem", "Kaf", "Khah", "Laa", "Lam", "Meem", "Noon", "Qaf", "Reh", "Sad",
    "Seen", "Sheen", "Tah", "Teh", "Teh_Marbuta", "Thal", "Theh", "Waw", "Yeh", 
    "Zah", "Zain"
]

mapping_arabe = {
    "Ain": "ع", "Al": "ال", "Alef": "أ", "Beh": "ب", "Dad": "ض",
    "Dal": "د", "Feh": "ف", "Ghain": "غ", "Hah": "ح", "Heh": "ه",
    "Jeem": "ج", "Kaf": "ك", "Khah": "خ", "Laa": "لا", "Lam": "ل",
    "Meem": "م", "Noon": "ن", "Qaf": "ق", "Reh": "ر", "Sad": "ص",
    "Seen": "س", "Sheen": "ش", "Tah": "ط", "Teh": "ت", "Teh_Marbuta": "ة",
    "Thal": "ذ", "Theh": "ث", "Waw": "و", "Yeh": "ي", "Zah": "ظ",
    "Zain": "ز"
}

@st.cache_resource
def load_trained_model():
    try:
        with open('model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
            
        # Dictionnaire d'objets personnalisés pour assurer la rétrocompatibilité Keras 2 -> Keras 3
        custom_objects = {
            'Sequential': tf.keras.models.Sequential,
            'Functional': tf.keras.models.Model,
            'InputLayer': tf.keras.layers.InputLayer,
            'Conv2D': tf.keras.layers.Conv2D,
            'MaxPooling2D': tf.keras.layers.MaxPooling2D,
            'Flatten': tf.keras.layers.Flatten,
            'Dense': tf.keras.layers.Dense
        }
        
        # Chargement de la structure avec les objets déclarés
        model = tf.keras.models.model_from_json(loaded_model_json, custom_objects=custom_objects)
        model.load_weights('model.h5')  
        return model
    except Exception as e:
        # Affichage détaillé de l'erreur dans la console/UI pour faciliter le débogage
        st.error(f"Erreur modèle : {e}")
        return None
model = load_trained_model()

# --- 2. TRAITEMENT ---

def get_hand_crop(image_pil):
    img = np.array(image_pil.convert('RGB'))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape
    
    # Conversion de l'image pour MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)
    
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            x_coords = [lm.x * w for lm in hand_landmarks]
            y_coords = [lm.y * h for lm in hand_landmarks]
            center_x, center_y = (min(x_coords) + max(x_coords)) / 2, (min(y_coords) + max(y_coords)) / 2
            box_size = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)) * 1.4
            
            x1, y1 = int(max(0, center_x - box_size/2)), int(max(0, center_y - box_size/2))
            x2, y2 = int(min(w, center_x + box_size/2)), int(min(h, center_y + box_size/2))
            
            roi = img_bgr[y1:y2, x1:x2]
            if roi.size == 0: return None, None, False
            
            roi_resized = cv2.resize(roi, (224, 224))
            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
            img_tensor = roi_rgb.astype('float32') / 255.0
            img_tensor = np.expand_dims(img_tensor, axis=0)
            
            return img_tensor, roi_rgb, True
    return None, None, False

# --- 3. INTERFACE ---

st.title("Classification des lettres de signe (version arabe) : Mode Traduction")

st.markdown(
    f"""
    <div style="text-align: right; direction: rtl; font-size: 40px; background-color: #000000; padding: 20px; border-radius: 10px; color: white;">
        {st.session_state.phrase}
    </div>
    """, 
    unsafe_allow_html=True
)

source = st.radio("Source :", ("Webcam", "Upload"))
img_file = st.camera_input("Faites votre signe") if source == "Webcam" else st.file_uploader("Image", type=["jpg", "png", "jpeg"])

if img_file:
    image_pil = Image.open(img_file)
    input_ia, img_display, detected = get_hand_crop(image_pil)
    
    if detected:
        col_img, col_act = st.columns([1, 1])
        
        with col_img:
            st.image(img_display, caption="Signe détecté", width="stretch")
            
        with col_act:
            if model is not None:
                preds = model.predict(input_ia)[0]
                top_idx = np.argmax(preds)
                lettre_detectee = nom_classe[top_idx]
                confiance = preds[top_idx] * 100

                st.subheader(f"Lettre : :blue[{lettre_detectee}]")
                st.write(f"Confiance : {confiance:.1f}%")
                
                st.write("---")
                st.markdown("##### 📊 Top 5 des prédictions :")
                
                top5_indices = np.argsort(preds)[-5:][::-1] 
                
                for i in top5_indices:
                    nom = nom_classe[i]
                    carac = mapping_arabe.get(nom, nom)
                    prob = preds[i] * 100
                    st.write(f"{carac} ({nom}) : {prob:.1f}%")
                    st.progress(int(prob))

                st.write("---")
                caractere_arabe = mapping_arabe.get(lettre_detectee, lettre_detectee)

                if st.button(f"✅ Ajouter '{caractere_arabe}' ({lettre_detectee})"):
                    st.session_state.phrase += caractere_arabe
                    st.experimental_rerun()

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("⌨️ Espace"):
                        st.session_state.phrase += " "
                        st.experimental_rerun()
                with col_btn2:
                    if st.button("🗑️ Effacer"):
                        st.session_state.phrase = ""
                        st.experimental_rerun()
            else:
                st.error("Le modèle n'a pas pu être chargé.")
    else:
        st.warning("Main non détectée.")