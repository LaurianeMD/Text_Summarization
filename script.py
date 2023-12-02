import streamlit as st
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from langdetect import detect
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
import util 
# Charger la fonction generate_summary
from util import set_background
import streamlit as st
from PIL import Image
set_background('./images/Grey2.jpg')
#Charger les données nécessaires pour nltk (si ce n'est pas déjà fait)
nltk.download('punkt')
nltk.download('stopwords')

# Fonction pour obtenir le résumé
def get_summary(text, summarizer, num_sentences, language="english"):
    # Détection automatique de la langue
    detected_language = detect(text)
    if detected_language == "fr":
        language = "french"
    elif detected_language == "en":
        language = "english"
    else:
        pass
        #st.warning("Langue détectée non prise en charge. Utilisation par défaut de la langue anglaise.")

    # Sélection du modèle de résumé
    if selected_summarizer == "TextRank":
        summarizer = TextRankSummarizer()
    elif selected_summarizer == "LexRank":
        summarizer = LexRankSummarizer()
    elif selected_summarizer == "LSA":
        summarizer = LsaSummarizer()
    elif selected_summarizer == "Modèle personnalisé":
        summarizer = generate_summary(user_input)
    else:
        st.error("Modèle de résumé non valide.")

    parser = PlaintextParser.from_string(text, Tokenizer(language))
    summary = summarizer(parser.document, num_sentences)
    text_summary = ""
    for sentence in summary:
        text_summary += str(sentence)
    return text_summary


# Fonction de résumé personnalisée
def generate_summary(article_text):
    # Tokeniser en phrase
    sentence_list = nltk.sent_tokenize(article_text)
    
    # Stopwords
    stopwords = nltk.corpus.stopwords.words('french')  # Change to 'english' for English text
    
    # Dictionnaire de fréquences des mots
    word_frequencies = {}
    for word in nltk.word_tokenize(article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    
    # Fréquence maximale
    maximum_frequency = max(word_frequencies.values())
    
    # Calculer la fréquence pondérée
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / maximum_frequency
    
    # Liste des scores de chaque phrase
    sentence_scores = {}
    
    # Calculer le score de chaque phrase
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    
    # Ordonner les phrases par pondération et récupérer les 10 premières phrases
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:10]
    
    # Regrouper ensemble les phrases qui ont les poids les plus élevés
    summary = ' '.join(summary_sentences)
    
    return summary

# Titre de l'application
st.title("Résumé de Texte avec Streamlit")

# Zone de texte pour l'utilisateur
user_input = st.text_area("Entrez votre texte ici:", key="user_input")

# Boutons pour les modèles de résumé
selected_summarizer = st.radio("Sélectionnez le modèle de résumé :", ["TextRank", "LexRank", "LSA", "Modèle personnalisé"])

# Nombre de phrases dans le résumé
num_sentences = st.slider("Nombre de phrases dans le résumé :", 1, 10, 5)

# ...

# Bouton pour générer le résumé
if st.button("Résumer", key="summarize_button"):
    # Sélection du modèle de résumé
    if selected_summarizer == "TextRank" or selected_summarizer == "LexRank" or selected_summarizer == "LSA":
        # Le modèle est pré-entraîné
        summarizer = None  # Déclarer la variable summarizer
        if selected_summarizer == "TextRank":
            summarizer = TextRankSummarizer()
        elif selected_summarizer == "LexRank":
            summarizer = LexRankSummarizer()
        elif selected_summarizer == "LSA":
            summarizer = LsaSummarizer()

        # Vérifier que le summarizer est défini avant de l'utiliser
        if summarizer is not None:
            summary_text = get_summary(user_input, summarizer, num_sentences)
            # Affichage du résumé avec une police différente (Times New Roman)
            st.markdown(f"<span style='font-family: Times New Roman;'>### Résumé généré avec {selected_summarizer}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-family: Times New Roman;'>{summary_text}</span>", unsafe_allow_html=True)
        else:
            st.error("Modèle de résumé non valide.")
    else:
        summary_text = generate_summary(user_input)
        # Affichage du résumé avec une police différente (Times New Roman)
        st.markdown(f"<span style='font-family: Times New Roman;'>### Résumé généré avec Modèle personnalisé</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='font-family: Times New Roman;'>{summary_text}</span>", unsafe_allow_html=True)
