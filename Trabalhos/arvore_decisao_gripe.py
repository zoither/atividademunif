"""
ÁRVORE DE DECISÃO
Dataset: Gripe
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

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

print("=== ÁRVORE DE DECISÃO ===")
print(f"Total de registros: {len(df)}\n")

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
feature_names = list(X.columns)
class_names = le_dict["gripe_ano_passado"].classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ─────────────────────────────────────────────
# 3. TREINAMENTO
# ─────────────────────────────────────────────
modelo = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print(f"Acurácia: {accuracy_score(y_test, y_pred):.2%}\n")
print("--- Relatório de Classificação ---")
print(classification_report(y_test, y_pred, target_names=class_names))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# ─────────────────────────────────────────────
# 4. VISUALIZAÇÃO DA ÁRVORE (TEXTO)
# ─────────────────────────────────────────────
print("\n--- Estrutura da Árvore (texto) ---")
regras_texto = export_text(modelo, feature_names=feature_names)
print(regras_texto)

# ─────────────────────────────────────────────
# 5. VISUALIZAÇÃO GRÁFICA
# ─────────────────────────────────────────────
plt.figure(figsize=(20, 8))
plot_tree(
    modelo,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Árvore de Decisão - Gripe", fontsize=14)
plt.tight_layout()
plt.savefig("arvore_decisao_gripe.png", dpi=150, bbox_inches="tight")
print("\nÁrvore salva em: arvore_decisao_gripe.png")

# ─────────────────────────────────────────────
# 6. IMPORTÂNCIA DOS ATRIBUTOS
# ─────────────────────────────────────────────
print("\n--- Importância dos Atributos ---")
importancias = sorted(
    zip(feature_names, modelo.feature_importances_),
    key=lambda x: x[1], reverse=True
)
for feat, imp in importancias:
    bar = "█" * int(imp * 40)
    print(f"  {feat:<20} {imp:.4f}  {bar}")
