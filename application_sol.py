import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import os  

# Titre de l'application
st.write('''
# Prédiction de Culture Agricole 🌱
Cette application recommande la culture la plus adaptée en fonction de la qualité du sol et des conditions environnementales.
''')

# Chargement des données
df = pd.read_csv('Crop_recommendation.csv')

# Encodage des labels en valeurs numériques
df['target'] = df.label.astype('category').cat.codes
targets = dict(enumerate(df.label.astype('category').cat.categories))  # Mapping cultures

# Dictionnaire associant les cultures aux images (ajoute les images dans le dossier "images/")
image_paths = {
    'rice': 'rice.jpg',
    'maize': 'maize.jpg',
    'chickpea': 'chickpea.jpg',
    'kidneybeans': 'kidneybeans.jpg',
    'pigeonpeas': 'pigeonpeas.jpg',
    'mothbeans': 'mothbeans.png',
    'mungbean': 'mungbean.jpg',
    'blackgram': 'blackgram.jpg',
    'lentil': 'lentil.jpg',
    'pomegranate': 'pomegranate.jpg',
    'banana': 'banana.jpg',
    'mango': 'mango.jpg',
    'grapes': 'grapes.jpg',
    'watermelon': 'watermelon.jpg',
    'muskmelon': 'muskmelon.jpg',
    'apple': 'apple.jpg',
    'orange': 'orange.png',
    'papaya': 'papaya.jpg',
    'coconut': 'coconut.png',
    'cotton': 'cotton.jpg',
    'jute': 'jute.jpg',
    'coffee': 'coffee.jpg'
}

# Séparation des données
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Normalisation des données
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement du modèle Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=1)  # 100 arbres
rf.fit(X_train_scaled, y_train)

# Enregistrement du scaler et du modèle
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(rf, open("rf_model.pkl", "wb"))

# Interface utilisateur dans Streamlit
st.sidebar.header("🔍 Paramètres du sol")

# Saisie des paramètres du sol via la barre latérale
def user_input():
    N = st.sidebar.slider('Azote (N)', 0.0, 140.0, 80.0)
    P = st.sidebar.slider('Phosphore (P)', 5.0, 145.0, 21.0)
    K = st.sidebar.slider('Potassium (K)', 5.0, 205.0, 180.0)
    temperature = st.sidebar.slider('Température (°C)', 8.8, 43.6, 26.0)
    humidity = st.sidebar.slider('Humidité (%)', 14.2, 99.8, 40.0)
    ph = st.sidebar.slider('pH', 3.5, 10.0, 7.0)
    rainfall = st.sidebar.slider('Précipitations (mm)', 20.2, 298.5, 122.0)

    data = {'N': N, 'P': P, 'K': K, 'temperature': temperature,
            'humidity': humidity, 'ph': ph, 'rainfall': rainfall}

    return pd.DataFrame(data, index=[0])

# Récupération des valeurs saisies
user_data = user_input()

# Affichage des valeurs saisies
st.subheader("🌍 Paramètres du sol sélectionnés :")
st.write(user_data)

# Bouton pour prédire
if st.button("🔮 Prédire la culture recommandée"):
    # Normalisation des nouvelles valeurs
    scaler = pickle.load(open("scaler.pkl", "rb"))
    user_data_scaled = scaler.transform(user_data)

    # Chargement du modèle Random Forest et prédiction
    rf_model = pickle.load(open("rf_model.pkl", "rb"))
    predicted_label = rf_model.predict(user_data_scaled)[0]  # Prédiction de la culture
    predicted_crop = targets[predicted_label]  # Nom de la culture prédite

    # Affichage du résultat
    st.subheader("🌾 La culture recommandée est :")
    st.write(f"**{predicted_crop}**")

    # Affichage de l'image si disponible
    image_path = image_paths.get(predicted_crop.lower(), None)
    if image_path and os.path.exists(image_path):
        st.image(image_path, caption=predicted_crop, use_column_width=True)
    else:
        st.write("📌 Aucune image disponible pour cette culture.")
