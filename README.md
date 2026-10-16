# 🤟 Classification de la Langue des Signes Arabe (Deep Learning & Streamlit)
Projet réalisé dans le cadre du **Master 1 EEA (SIA2)** à l'**Université de Toulouse** par **Ziad RAMMAL** et **Melissa YESGUER**.
Ce projet propose une application interactive capable de reconnaître et traduire les lettres de l'alphabet arabe en langue des signes (31 classes).
---
## 📌 Architecture du projet
Le pipeline de traitement se compose de deux étapes majeures :
1. **Détection & Recadrage (MediaPipe Tasks)** : Utilisation du modèle `HandLandmarker` de MediaPipe pour repérer la main dans la vidéo ou l'image et effectuer un recadrage (*crop*) précis autour des repères (*landmarks*).
2. **Classification (Transfer Learning VGG16)** : Un réseau de neurones convolutif basé sur la structure **VGG16 pré-entraînée sur ImageNet** (avec Fine-Tuning) pour classer le signe recadré parmi 31 catégories.
---
## 📁 Structure du dépôt
```text
.
├── app.py                     # Code principal de l'application Streamlit
├── hand_landmarker.task        # Modèle MediaPipe Task pour la détection des mains
├── requirements.txt            # Dépendances pip pour le déploiement Cloud
├── environment_train.yml       # Environnement Conda pour l'entraînement du modèle
├── environment_app.yml         # Environnement Conda pour l'exécution de l'application
├── docs/                       # Rapport et supports de présentation
└── README.md                   # Documentation du projet
```
---
## 📦 Téléchargement des fichiers lourds (Google Drive)
En raison des limites de taille imposées par GitHub, le modèle entraîné ainsi que le jeu de données complet sont hébergés séparément :
- 🧠 **Modèle entraîné (`model.h5` et `model.json`)** : [Télécharger le modèle sur Google Drive](https://drive.google.com/drive/folders/1oci-zhFr9BcLwQY8KcUoNtknlm9Z5Kry?usp=drive_link)
- 📊 **Dataset (ARSL-No-BackGround-V2)** : [Télécharger le dataset](https://drive.google.com/drive/folders/1oci-zhFr9BcLwQY8KcUoNtknlm9Z5Kry?usp=drive_link)
> **Remarque :** Pour exécuter l'application en local, téléchargez `model.h5` et `model.json` puis placez-les à la racine du projet.
---
## 🛠️ Installation & Exécution locale
### Option 1 : Utiliser les environnements Anaconda (Recommandé)
Pour éviter tout conflit de versions, deux fichiers d'environnement Anaconda sont fournis :
1. **Environnement d'entraînement** (TensorFlow, Keras, etc.) :
   ```bash
   conda env create -f environment_train.yml
   conda activate tf-gpu
   ```
2. **Environnement d'exécution Streamlit** :
   ```bash
   conda env create -f environment_app.yml
   conda activate sign_app
   streamlit run app.py
   ```
---
### Option 2 : Installation via `pip`
```bash
# 1. Cloner le dépôt
git clone https://github.com/ZiadRAMMAL/Arabic-Sign-Language-Classification.git
cd Arabic-Sign-Language-Classification
# 2. Installer les dépendances
pip install -r requirements.txt
# 3. Lancer l'application
streamlit run app.py
```
---
## 📜 Documentation
Consultez le dossier `docs/` pour accéder au rapport détaillé du projet ainsi qu'aux supports de présentation.

👉 **[Accéder à l'application en ligne sur Streamlit Cloud](https://arabic-sign-language-classification-ndsm4odtfjabqqedhhxiwr.streamlit.app/)**