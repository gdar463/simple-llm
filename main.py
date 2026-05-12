import json
import re
from argparse import ArgumentParser

import numpy as np
from tensorflow.keras.models import load_model

parser = ArgumentParser()
parser.add_argument("-m", "--model", help="file di output del modello keras",
                    default="model/model.keras")
parser.add_argument("-i", "--indices", help="nome del file di testo da cui iniziare",
                    default="model/indices.json")
parser.add_argument("-l", "--len", help="lunghezza sequenza", default=5)
args = parser.parse_args()

# =========================================================
# 1. CARICAMENTO MODELLO E INDICI
# =========================================================

modello = load_model(args.model)

with open(args.indices, "rt") as f:
    indici = json.load(f)
indice_parola = {int(indice): parola for indice, parola in indici.items()}
parola_indice = {parola: indice for indice, parola in indice_parola.items()}

# =========================================================
# 2. FUNZIONE PER TRASFORMARE TESTO IN SEUQUENZA
# =========================================================

def text_to_sequence(frase):
    sequenza = []
    for parola in frase.split(" "):
        sequenza.append(parola_indice.get(parola))
    return sequenza

# =========================================================
# 3. FUNZIONE PER PREDIRE UNA PAROLA
# =========================================================

def predici_prossima_parola(frase, modello, lunghezza_sequenza):
    frase = frase.lower()
    frase = re.sub(r"[^\w\sàèéìòù]", "", frase)
    frase = re.sub(r"\s+", " ", frase).strip()

    sequenza = text_to_sequence(frase)

    if len(sequenza) < lunghezza_sequenza:
        raise ValueError(f"La frase deve contenere almeno {lunghezza_sequenza} parole.")

    sequenza = sequenza[-lunghezza_sequenza:]
    sequenza = np.array([sequenza])

    predizione = modello.predict(sequenza, verbose=0)
    indice_predetto = np.argmax(predizione)

    parola_predetta = indice_parola.get(indice_predetto, None)
    return parola_predetta

# =========================================================
# 4. FUNZIONE PER GENERARE TESTO
# =========================================================

def genera_testo(seed_text, n_parole, modello, lunghezza_sequenza):
    testo_generato = seed_text

    for _ in range(n_parole):
        try:
            prossima_parola = predici_prossima_parola(
                testo_generato,
                modello,
                lunghezza_sequenza
            )
        except ValueError:
            break

        if prossima_parola is None:
            break

        testo_generato += " " + prossima_parola

    return testo_generato

# =========================================================
# 5. TEST
# =========================================================

seed = "nel mezzo del cammin di"
testo_generato = genera_testo(seed, 10, modello, args.len)

print("\nSeed iniziale:")
print(seed)

print("\nTesto generato:")
print(testo_generato)