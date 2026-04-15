"""
KNN - K-Nearest Neighbors
Dataset: Gripe
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
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

print("=== KNN - K-Nearest Neighbors ===")
print(f"Total de registros: {len(df)}")
print(f"Colunas: {list(df.columns)}\n")

# ─────────────────────────────────────────────
# 2. PRÉ-PROCESSAMENTO
# ─────────────────────────────────────────────
le = LabelEncoder()
df_encoded = df.copy()
for col in df_encoded.columns:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

X = df_encoded.drop(columns=["gripe_ano_passado"])
y = df_encoded["gripe_ano_passado"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ─────────────────────────────────────────────
# 3. TREINAMENTO E AVALIAÇÃO (K=3, 5, 7)
# ─────────────────────────────────────────────
for k in [3, 5, 7]:
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"K={k} → Acurácia: {acc:.2%}")

# Melhor K = 5 para relatório final
print("\n--- Relatório com K=5 ---")
modelo_final = KNeighborsClassifier(n_neighbors=5)
modelo_final.fit(X_train, y_train)
y_pred = modelo_final.predict(X_test)

print(classification_report(y_test, y_pred, target_names=["Não Gripado", "Gripado"]))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# ─────────────────────────────────────────────
# 4. EXEMPLO DE PREDIÇÃO
# ─────────────────────────────────────────────
print("\n--- Exemplo de Predição ---")
exemplo = X_test.iloc[[0]]
resultado = modelo_final.predict(exemplo)
print(f"Entrada: {exemplo.values}")
print(f"Predição: {'Gripado' if resultado[0] == 1 else 'Não Gripado'}")
