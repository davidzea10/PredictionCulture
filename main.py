import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# ➕ Intégration Google Analytics
GA_JS = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-D2T0F15YZ8"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-D2T0F15YZ8');
</script>
"""
st.markdown(GA_JS, unsafe_allow_html=True)
# Titre de l'application
st.write('''
# 🌱 Prédiction de Culture Agricole
Cette application recommande la culture la plus adaptée en fonction de la qualité du sol et des conditions environnementales..
''')

# Chargement des données
df = pd.read_csv('Crop_recommendation.csv')

# Encodage des labels en valeurs numériques
df['target'] = df.label.astype('category').cat.codes
targets = dict(enumerate(df.label.astype('category').cat.categories))  # Mapping cultures

# Dictionnaire associant les cultures aux images
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

# Dictionnaire contenant les informations sur chaque culture
crop_info = {
    'coffee': {
        'description': "Le café est une culture tropicale appréciée pour ses grains aromatiques utilisés pour produire du café.",
        'needs': "Sol bien drainé, riche en matière organique, pH légèrement acide (6-6.5), humidité élevée.",
        'cycle': "Plantation → Croissance (3-4 ans) → Floraison → Développement des cerises → Récolte (~9 mois après floraison)."
    },
    'orange': {
        'description': "L'orange est un agrume riche en vitamine C, cultivé dans les climats tempérés et subtropicaux.",
        'needs': "Sol bien drainé, pH légèrement acide à neutre (6-7), températures modérées, bon ensoleillement.",
        'cycle': "Semis → Croissance (3-5 ans) → Floraison → Développement des fruits → Récolte (hiver-été selon la variété)."
    },
    'apple': {
        'description': "La pomme est un fruit à pépins cultivé dans des climats tempérés.",
        'needs': "Climat frais, sol bien drainé, pH légèrement acide (6-6.5), bon ensoleillement.",
        'cycle': "Plantation → Croissance (2-4 ans) → Floraison (printemps) → Développement des fruits → Récolte (automne)."
    },
    'pomegranate': {
        'description': "La grenade est un fruit riche en antioxydants, adapté aux climats arides et subtropicaux.",
        'needs': "Sol bien drainé, pH légèrement alcalin à neutre (6.5-7.5), faible besoin en eau.",
        'cycle': "Plantation → Croissance (2-3 ans) → Floraison → Fructification → Récolte (automne)."
    },
    'chickpea': {
        'description': "Le pois chiche est une légumineuse résistante à la sécheresse, riche en protéines.",
        'needs': "Sol bien drainé, pH légèrement alcalin à neutre (6-7.5), faible humidité, climat chaud.",
        'cycle': "Semis → Croissance → Floraison → Formation des gousses → Récolte (~4-5 mois)."
    },
    'blackgram': {
        'description': "Le blackgram est une légumineuse riche en protéines, cultivée principalement en Asie.",
        'needs': "Sol bien drainé, pH neutre (6-7), climat chaud et humide.",
        'cycle': "Semis → Croissance → Floraison → Développement des graines → Récolte (~3-4 mois)."
    },
    'mango': {
        'description': "La mangue est un fruit tropical apprécié pour sa chair sucrée et juteuse.",
        'needs': "Sol bien drainé, pH légèrement acide (5.5-7), climat chaud, bon ensoleillement.",
        'cycle': "Plantation → Croissance (3-6 ans) → Floraison → Développement des fruits → Récolte (été)."
    },
    'papaya': {
        'description': "La papaye est un fruit tropical riche en enzymes et en vitamines.",
        'needs': "Sol bien drainé, pH neutre à légèrement acide (6-7), climat chaud et humide.",
        'cycle': "Plantation → Croissance (~6 mois) → Floraison → Fructification → Récolte (~9-12 mois)."
    },
    'jute': {
        'description': "Le jute est une fibre naturelle utilisée pour fabriquer des sacs, des cordes et du textile.",
        'needs': "Sol alluvial, pH légèrement acide à neutre (5.5-7), climat chaud et humide.",
        'cycle': "Semis → Croissance rapide (~4-5 mois) → Récolte avant floraison."
    },
    'coconut': {
        'description': "Le cocotier produit des noix de coco utilisées pour leur eau, leur chair et leur huile.",
        'needs': "Sol sableux bien drainé, pH légèrement acide à neutre (5.5-7.5), climat tropical humide.",
        'cycle': "Plantation → Croissance (5-6 ans) → Floraison → Fructification continue → Récolte toute l'année."
    },
    'cotton': {
        'description': "Le coton est une plante fibreuse utilisée dans l'industrie textile.",
        'needs': "Sol bien drainé, pH neutre à légèrement acide (6-7), climat chaud et sec.",
        'cycle': "Semis → Croissance → Floraison → Formation des capsules → Récolte (~5-6 mois)."
    },
    'lentil': {
        'description': "La lentille est une légumineuse riche en protéines et cultivée dans les régions tempérées.",
        'needs': "Sol bien drainé, pH neutre à légèrement alcalin (6-7.5), climat tempéré.",
        'cycle': "Semis → Germination → Croissance → Floraison → Récolte (~4 mois)."
    },
    'grapes': {
        'description': "Le raisin est un fruit cultivé pour la consommation fraîche, le vin et les jus.",
        'needs': "Sol bien drainé, pH légèrement acide à neutre (5.5-7), climat tempéré à chaud.",
        'cycle': "Plantation → Croissance (2-3 ans) → Floraison → Développement des grappes → Récolte (été-automne)."
    },
    'watermelon': {
        'description': "La pastèque est un fruit rafraîchissant riche en eau et en sucres naturels.",
        'needs': "Sol bien drainé, pH légèrement acide à neutre (5.5-7), climat chaud.",
        'cycle': "Semis → Croissance → Floraison → Formation des fruits → Récolte (~3-4 mois)."
    },
    'muskmelon': {
        'description': "Le melon cantaloup est un fruit sucré et parfumé, apprécié en été.",
        'needs': "Sol bien drainé, pH légèrement acide à neutre (5.5-7), climat chaud.",
        'cycle': "Semis → Croissance → Floraison → Formation des fruits → Récolte (~3-4 mois)."
    },
    'mothbeans': {
        'description': "Les haricots mat sont une légumineuse tolérante à la sécheresse, riche en protéines.",
        'needs': "Sol bien drainé, pH légèrement alcalin (6.5-7.5), climat chaud et sec.",
        'cycle': "Semis → Croissance → Floraison → Formation des gousses → Récolte (~3-4 mois)."
    },
    'kidneybeans': {
        'description': "Les haricots rouges sont une légumineuse nutritive utilisée dans de nombreuses cuisines.",
        'needs': "Sol bien drainé, pH légèrement acide à neutre (6-7), climat tempéré à chaud.",
        'cycle': "Semis → Croissance → Floraison → Formation des gousses → Récolte (~3-4 mois)."
    },
    'pigeonpeas': {
        'description': "Le pois d'Angole est une légumineuse résistante à la sécheresse, riche en protéines.",
        'needs': "Sol bien drainé, pH légèrement acide à neutre (5.5-7), climat chaud et sec.",
        'cycle': "Semis → Croissance → Floraison → Formation des gousses → Récolte (~4-5 mois)."
    },
    'maize': {
        'description': "Le maïs est une céréale essentielle utilisée pour l'alimentation humaine et animale.",
        'needs': "Sol bien drainé, pH neutre à légèrement acide (6-7), bon ensoleillement.",
        'cycle': "Semis → Germination → Croissance → Floraison → Récolte (~3-5 mois)."
    }
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
rf = RandomForestClassifier(n_estimators=100, random_state=1)
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
    predicted_label = rf_model.predict(user_data_scaled)[0]
    predicted_crop = targets[predicted_label]

    # Affichage du résultat
    st.subheader("🌾 La culture recommandée est :")
    st.write(f"**{predicted_crop}**")

    # Affichage de l'image et des détails
    image_path = image_paths.get(predicted_crop.lower(), None)
    if image_path and os.path.exists(image_path):
        st.image(image_path, caption=predicted_crop, use_column_width=True)

    # Affichage des détails sur la culture
    if predicted_crop.lower() in crop_info:
        st.write("📌 **Description**:", crop_info[predicted_crop.lower()]['description'])
        st.write("🛠 **Besoins**:", crop_info[predicted_crop.lower()]['needs'])
        st.write("🌱 **Cycle de croissance**:", crop_info[predicted_crop.lower()]['cycle'])
    else:
        st.write("📌 Aucune information détaillée disponible pour cette culture.")

    # Génération d'un graphique pour visualiser l'influence des paramètres du sol
    st.subheader("📊 Influence des paramètres du sol sur la culture")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=user_data.columns, y=user_data.values[0], palette="coolwarm")
    plt.xticks(rotation=45)
    plt.ylabel("Valeurs")
    plt.xlabel("Paramètres du sol")
    plt.title("Impact des paramètres sur la culture")
    st.pyplot(fig)

# Ajout d'une section "À propos"
st.sidebar.subheader("📢 À propos")
st.sidebar.write("Application développée par Débuze David ")
st.sidebar.write("Données issues de Kaggle pour l'entrainement.")
st.sidebar.write("Modèle basé sur un Random Forest entraîné sur un jeu de données de recommandation de culture.")
