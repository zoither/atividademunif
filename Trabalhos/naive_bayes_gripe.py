"""
NAIVE BAYES
Dataset: Gripe
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─────────────────────────────────────────────
# 1. LEITURA DOS DADOS
# ─────────────────────────────────────────────
url = "https://docs.google.com/spreadsheets/d/1g1aQ61vijh6uHJuc8sijeBEMsoIQ2a5yLwUK04Wptlg/export?format=csv"
df = pd.read_csv(url)

mapping = {
    "Carimbo de data/hora": "timestamp",
    "Você ficou gripado no ano passado ?": "gripe_ano_passado",
    "Você tomou vacina da gripe no ano passado?": "vacina",
    "  Você frequentou no ano passado,  semanalmente ambientes com muitas pessoas? (salas cheias, ônibus, eventos, etc.)  ": "ambientes_cheios",
    "  Você viajou no ano passado mais de 100 km de distância?  ": "viajou",
    "  Você tem alergia nas vias aéreas (rinite, sinusite, etc.)?  ": "alergia",
    "Quantas horas você dormiu em média por noite no ano passado?": "horas_sono",
    "Você praticou atividade física no ano passado?": "exercicio",
    "Você se alimentou de forma balanceada no ano passado?": "alimentacao",
    "Em média, quantas vezes você lavou as mãos por dia no ano passado?": "lavagem_maos",
    "Na sua percepção, o seu nível de estresse no ano passado foi:": "estresse"
}

df = df.rename(columns=mapping).dropna().drop(columns=["timestamp"])

print("=== NAIVE BAYES ===")
print(f"Total de registros: {len(df)}")
print(f"Distribuição da classe alvo:\n{df['gripe_ano_passado'].value_counts()}\n")

# ─────────────────────────────────────────────
# 2. PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────
le_dict = {}
df_encoded = df.copy()
for col in df_encoded.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    le_dict[col] = le

X = df_encoded.drop(columns=["gripe_ano_passado"])
y = df_encoded["gripe_ano_passado"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ─────────────────────────────────────────────
# 3. MODELO GAUSSIAN NAIVE BAYES
# ─────────────────────────────────────────────
modelo = GaussianNB()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print(f"Acurácia: {accuracy_score(y_test, y_pred):.2%}\n")
print("--- Relatório de Classificação ---")
print(classification_report(y_test, y_pred, target_names=["Não Gripado", "Gripado"]))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# ─────────────────────────────────────────────
# 4. PROBABILIDADES MANUAIS (cálculo didático)
# ─────────────────────────────────────────────
print("\n--- Probabilidades da Classe Alvo (P(gripe)) ---")
total = len(df)
for classe, count in df["gripe_ano_passado"].value_counts().items():
    print(f"  P({classe}) = {count}/{total} = {count/total:.4f}")

# ─────────────────────────────────────────────
# 5. EXEMPLO DE PREDIÇÃO COM PROBABILIDADES
# ─────────────────────────────────────────────
print("\n--- Exemplo de Predição ---")
exemplo = X_test.iloc[[0]]
probs = modelo.predict_proba(exemplo)[0]
classes = le_dict["gripe_ano_passado"].classes_
for c, p in zip(classes, probs):
    print(f"  P({c}) = {p:.4f} ({p:.2%})")
pred = modelo.predict(exemplo)[0]
print(f"  → Classe prevista: {le_dict['gripe_ano_passado'].inverse_transform([pred])[0]}")
