import tensorflow as tf
import numpy as np
import re

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# =========================================================
# 1. TESTO DI PARTENZA
# =========================================================

testo = """
Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura
che la diritta via era smarrita
Ahi quanto a dir qual era è cosa dura
esta selva selvaggia e aspra e forte
che nel pensier rinova la paura
"""

# =========================================================
# 2. PULIZIA TESTO
# =========================================================

testo = testo.lower()
testo = re.sub(r"[^\w\sàèéìòù]", "", testo)
testo = re.sub(r"\s+", " ", testo).strip()

print("Testo pulito:")
print(testo)
print()

# =========================================================
# 3. TOKENIZZAZIONE
# =========================================================

tokenizer = Tokenizer()
tokenizer.fit_on_texts([testo])

sequenza_tokenizzata = tokenizer.texts_to_sequences([testo])[0]
vocabolario = len(tokenizer.word_index) + 1

print("Vocabolario:")
print(tokenizer.word_index)
print()

print("Sequenza tokenizzata:")
print(sequenza_tokenizzata)
print()

# =========================================================
# 4. CREAZIONE DATASET
# =========================================================

lunghezza_sequenza = 5

X = []
y = []

for i in range(len(sequenza_tokenizzata) - lunghezza_sequenza):
    seq_input = sequenza_tokenizzata[i:i + lunghezza_sequenza]
    parola_output = sequenza_tokenizzata[i + lunghezza_sequenza]

    X.append(seq_input)
    y.append(parola_output)

X = np.array(X)
y = np.array(y)

print("Shape X:", X.shape)
print("Shape y:", y.shape)

# one-hot encoding di y
y = to_categorical(y, num_classes=vocabolario)

print("Shape y dopo one-hot:", y.shape)
print()

# =========================================================
# 5. MODELLO
# =========================================================

modello = Sequential([
    Embedding(input_dim=vocabolario, output_dim=32),
    LSTM(64),
    Dense(vocabolario, activation="softmax")
])

modello.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

modello.summary()

# =========================================================
# 6. TRAINING
# =========================================================

modello.fit(X, y, epochs=300, verbose=1)

# =========================================================
# 7. MAPPA INVERSA indice -> parola
# =========================================================

indice_parola = {indice: parola for parola, indice in tokenizer.word_index.items()}

# =========================================================
# 8. FUNZIONE PER PREDIRE UNA PAROLA
# =========================================================

def predici_prossima_parola(frase, modello, tokenizer, lunghezza_sequenza):
    frase = frase.lower()
    frase = re.sub(r"[^\w\sàèéìòù]", "", frase)
    frase = re.sub(r"\s+", " ", frase).strip()

    sequenza = tokenizer.texts_to_sequences([frase])[0]

    if len(sequenza) < lunghezza_sequenza:
        raise ValueError(f"La frase deve contenere almeno {lunghezza_sequenza} parole.")

    sequenza = sequenza[-lunghezza_sequenza:]
    sequenza = np.array([sequenza])

    predizione = modello.predict(sequenza, verbose=0)
    indice_predetto = np.argmax(predizione)

    parola_predetta = indice_parola.get(indice_predetto, None)
    return parola_predetta

# =========================================================
# 9. FUNZIONE PER GENERARE TESTO
# =========================================================

def genera_testo(seed_text, n_parole, modello, tokenizer, lunghezza_sequenza):
    testo_generato = seed_text

    for _ in range(n_parole):
        try:
            prossima_parola = predici_prossima_parola(
                testo_generato,
                modello,
                tokenizer,
                lunghezza_sequenza
            )
        except ValueError:
            break

        if prossima_parola is None:
            break

        testo_generato += " " + prossima_parola

    return testo_generato

# =========================================================
# 10. TEST
# =========================================================

seed = "nel mezzo del cammin di"
testo_generato = genera_testo(seed, 10, modello, tokenizer, lunghezza_sequenza)

print("\nSeed iniziale:")
print(seed)

print("\nTesto generato:")
print(testo_generato)