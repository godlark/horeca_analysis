# Opis / Description

Ten skrypt analizuje opisy dofinansowania obiektów gastronomicznych/turystycznych,
lematyzuje je, grupuje podobne słowa w klastry przy użyciu wektorów fastText,
a następnie oblicza top 5 najważniejszych słów (po klastrach) dla każdego wpisu.

This script analyzes Polish descriptions of fundings for horeca objects, lemmatizes them,
clusters similar words using fastText embeddings, and calculates the top 5
most important cluster-based words for each entry.

---

# Wymagania / Requirements

- Python 3.11+
- Zainstalowane biblioteki z pliku requirements.txt
- Plik `horeca_pomorskie_with_powiat.csv` z kolumną `name`
- Plik `cc.pl.300.bin` (Polish fastText model from Facebook)

---

# Instalacja / Installation

1. Utwórz i aktywuj wirtualne środowisko:
   python -m venv venv
   source venv/bin/activate  (Linux/Mac)
   venv\Scripts\activate     (Windows)

2. Zainstaluj wymagane biblioteki:
   pip install -r requirements.txt

3. Pobierz polski model fastText:
   https://fasttext.cc/docs/en/crawl-vectors.html
   (plik `cc.pl.300.bin` umieść w katalogu skryptu)

4. Pobierz polskie modele spaCy i Stanza:
   python -m spacy download pl_core_news_lg
   import stanza; stanza.download('pl')

---

# Uruchomienie / Run

python add_semantif_tfdif.py

Wynik zostanie zapisany w pliku:
horeca_pomorskie_with_powiat_with_tfidf.csv
