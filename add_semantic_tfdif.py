import gensim
import pandas as pd
import spacy
import stanza
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# 📌 1. Wczytanie danych z pliku CSV
# Load data from CSV file
df = pd.read_csv("horeca_pomorskie_with_powiat.csv")
documents = df['name']

# 📌 2. Tokenizacja i lematyzacja tekstu w języku polskim
# Tokenize and lemmatize Polish text
nlp = stanza.Pipeline('pl', processors='tokenize,lemma')
spacy_nlp = spacy.load("pl_core_news_lg")
def stanza_lemmatize(text):
    doc = nlp(text)
    lemmas = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.lemma.isalpha() and word.lemma.lower() not in spacy_nlp.Defaults.stop_words:
                lemmas.append(word.lemma.lower())
    return lemmas

tokenized_docs = [stanza_lemmatize(doc) for doc in documents]

# 📌 3. Wczytanie wcześniej wytrenowanych wektorów fastText (Polish)
# Load pre-trained fastText vectors (Polish)
model = gensim.models.fasttext.load_facebook_model("cc.pl.300.bin")


# 📌 4. Tworzenie listy unikalnych słów zawartych w dokumentach (tylko te, które są w modelu)
# Create list of unique words in documents (only if in model)
unique_words = list(set(
    word for doc in tokenized_docs for word in doc if word in model.wv.key_to_index
))
print(unique_words)

# 📌 5. Dodanie listy zakazanych słów (forbidden words) i lematyzacja ich
# Add list of forbidden words and lemmatize them
forbidden_words = ["krajowego", "planu", "odbudowy", "gospodarka", "dywersyfikacja",
                 "projekt", "województwo", "Gdańsk", "pomorze", "innowacyjności", "uodpornienie", "wzmocnienie",
                 "rozszerzenia", "zdywersyfikowanie", "wprowadzenie", "spółka", "odpowiedzialnością", "ograniczoną"]
forbidden_words = [nlp(word).sentences[0].words[0].lemma.lower() for word in forbidden_words]
print(forbidden_words)
unique_words += forbidden_words

# 📌 6. Konwersja słów do wektorów osadzeń (embeddings)
# Convert words to embeddings
embeddings = np.array([model.wv[word] for word in unique_words])

# 📌 7. Grupowanie słów w klastry metodą aglomeracyjną (cosine distance)
# Cluster words using AgglomerativeClustering (cosine distance)
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.15, metric='cosine', linkage='average')
clusters = clustering.fit_predict(embeddings)

# 📌 8. Mapowanie słów na ID klastrów
# Map words to cluster IDs
word_to_cluster = {word: cluster_id for word, cluster_id in zip(unique_words, clusters)}
forbidden_clusters = [word_to_cluster[forbidden_word] for forbidden_word in forbidden_words]
print(forbidden_clusters)

# 📌 9. Zamiana słów w dokumentach na ID klastrów
# Replace words in documents with cluster IDs
cluster_docs = []
for doc in tokenized_docs:
    cluster_docs.append([str(word_to_cluster[word]) for word in doc if word in word_to_cluster])

# 📌 10. Obliczanie TF-IDF dla ID klastrów (jako tokenów)
# Run TF-IDF on cluster IDs (as tokens)
tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
tfidf_matrix = tfidf.fit_transform(cluster_docs)

# 📌 11. Tworzenie odwrotnej mapy: ID klastra → przykładowe słowo
# Inverse mapping: cluster_id → example word
cluster_to_word = {}
for word, cluster_id in word_to_cluster.items():
    if cluster_id not in cluster_to_word:
        cluster_to_word[cluster_id] = word

feature_names = tfidf.get_feature_names_out()  # cluster IDs as strings

# 📌 12. Wyznaczenie 5 najważniejszych słów (klastrów) dla każdego dokumentu
# Get top 5 important clusters (words) for each document
top1_list, top2_list, top3_list, top4_list, top5_list = [], [], [], [], []
for doc_idx, vector in enumerate(tfidf_matrix):
    print(f"\nDocument {doc_idx + 1}:")
    scores = vector.toarray().flatten()

    # Sortowanie malejąco po wyniku tf-idf
    # Sort indices by descending tf-idf score
    sorted_indices = scores.argsort()[::-1]

    count = 0
    doc_top_words = ["", "", "", "", ""]  # placeholders for top1..top5
    for idx in sorted_indices:
        cluster_id_str = feature_names[idx]
        cluster_id = int(cluster_id_str)

        if cluster_id in forbidden_clusters:
            continue  # skip forbidden clusters

        example_word = cluster_to_word.get(cluster_id, "N/A")
        doc_top_words[count] = example_word
        score = scores[idx]
        count += 1
        print(f"  Top {count} cluster: ID={cluster_id}, example word='{example_word}', tf-idf={score:.4f}")

        if count >= 5:  # print top 5 allowed clusters
            break

    top1_list.append(doc_top_words[0])
    top2_list.append(doc_top_words[1])
    top3_list.append(doc_top_words[2])
    top4_list.append(doc_top_words[3])
    top5_list.append(doc_top_words[4])

# 📌 13. Dodanie kolumn z wynikami do DataFrame
# Add results to DataFrame
df["tfidf1"] = top1_list
df["tfidf2"] = top2_list
df["tfidf3"] = top3_list
df["tfidf4"] = top4_list
df["tfidf5"] = top5_list

# 📌 14. Zapis danych do CSV
# Save results to CSV
df.to_csv("horeca_pomorskie_with_powiat_with_tfidf.csv", index=False)