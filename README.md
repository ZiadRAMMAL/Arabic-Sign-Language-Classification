# 🤟 Classification de la Langue des Signes Arabe (Deep Learning & Streamlit)

Ce projet a été réalisé dans le cadre du Master 1 EEA (SIA2) à l'Université de Toulouse par **Ziad RAMMAL** et **Melissa YESGUER**.

Il s'agit d'une application d'apprentissage profond permettant de reconnaître et de traduire en temps réel les lettres de l'alphabet arabe en langue des signes (31 classes).

---

## 🚀 Démo en ligne
👉 **[Tester l'application Streamlit Cloud](URL_DE_VOTRE_APP_STREAMLIT)**

---

## 📌 Architecture & Méthodologie

1. **Prétraitement & Détection de la main** : Utilisation de la nouvelle API **MediaPipe Tasks** (`HandLandmarker`) pour détecter la main, calculer la boîte englobante (bounding box) et effectuer un recadrage dynamique (*crop*).
2. **Classification (Deep Learning)** : Modèle basé sur la recherche de transfert (**VGG16 pré-entraîné sur ImageNet**) avec Fine-Tuning.
3. **Performances** : Précision finale de **~97%** sur l'ensemble de test.
4. **Interface utilisateur** : Déploiement sous forme d'application web interactive avec **Streamlit**.

---

## 📁 Fichiers Lourds & Téléchargements

En raison des limitations de taille de GitHub, les poids du modèle entraîné et le jeu de données complet sont hébergés sur Google Drive :

- 📦 **Modèle entraîné (`model.h5` & `model.json`)** : [Télécharger le modèle sur Google Drive](LIEN_GOOGLE_DRIVE_MODELE)
- 📊 **Dataset (ARSL-No-BackGround-V2)** : [Télécharger le jeu de données](LIEN_GOOGLE_DRIVE_DATASET)

> *Remarque : Pour exécuter l'application localement, téléchargez `model.h5` et `model.json` et placez-les à la racine du projet.*

---

## 💻 Installation et Exécution Locale

```bash
# 1. Cloner le dépôt
git clone [https://github.com/VOTRE_NOM_UTILISATEUR/ARSL-Classification.git](https://github.com/VOTRE_NOM_UTILISATEUR/ARSL-Classification.git)
cd ARSL-Classification

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger le modèle (model.h5 et model.json) depuis le lien Google Drive et les placer dans ce dossier

# 5. Lancer l'application Streamlit
streamlit run app.py