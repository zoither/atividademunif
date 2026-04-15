"""
REGRAS DE ASSOCIAÇÃO (Association Rules)
Dataset: Gripe
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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

print("=== REGRAS DE ASSOCIAÇÃO ===")
print(f"Total de registros: {len(df)}\n")

# ─────────────────────────────────────────────
# 2. INSTALA E IMPORTA mlxtend
# ─────────────────────────────────────────────
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mlxtend", "-q"])
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

# ─────────────────────────────────────────────
# 3. TRANSFORMAÇÃO PARA ONE-HOT (binário por item)
# ─────────────────────────────────────────────
# Cada célula vira "coluna=valor" para fazer one-hot
transactions = []
for _, row in df.iterrows():
    items = [f"{col}={val}" for col, val in row.items()]
    transactions.append(items)

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_onehot = pd.DataFrame(te_array, columns=te.columns_)

print(f"Itens únicos gerados: {len(te.columns_)}")

# ─────────────────────────────────────────────
# 4. FREQUENT ITEMSETS (Apriori)
# ─────────────────────────────────────────────
MIN_SUPPORT = 0.3

frequent_itemsets = apriori(
    df_onehot,
    min_support=MIN_SUPPORT,
    use_colnames=True
)
frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)

print(f"\nItemsets frequentes (suporte >= {MIN_SUPPORT}): {len(frequent_itemsets)}")
print(frequent_itemsets.sort_values("support", ascending=False).head(10).to_string(index=False))

# ─────────────────────────────────────────────
# 5. GERAÇÃO DAS REGRAS
# ─────────────────────────────────────────────
MIN_CONFIDENCE = 0.6

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=MIN_CONFIDENCE
)

# Filtrar somente regras que concluem sobre gripe
rules_gripe = rules[
    rules["consequents"].apply(
        lambda c: any("gripe_ano_passado" in item for item in c)
    )
].copy()

rules_gripe = rules_gripe.sort_values("confidence", ascending=False)

print(f"\nRegras sobre 'gripe_ano_passado' (conf >= {MIN_CONFIDENCE}): {len(rules_gripe)}")
print("\n--- TOP 10 REGRAS ---")

for i, row in rules_gripe.head(10).iterrows():
    ant = ", ".join(sorted(row["antecedents"]))
    con = ", ".join(sorted(row["consequents"]))
    print(f"\nRegra {i+1}:")
    print(f"  SE   {ant}")
    print(f"  ENTÃO {con}")
    print(f"  Suporte:    {row['support']:.4f} ({row['support']:.2%})")
    print(f"  Confiança:  {row['confidence']:.4f} ({row['confidence']:.2%})")
    print(f"  Lift:       {row['lift']:.4f}")

# ─────────────────────────────────────────────
# 6. PRIMEIRA REGRA (para entrega do exercício)
# ─────────────────────────────────────────────
if len(rules_gripe) > 0:
    primeira = rules_gripe.iloc[0]
    print("\n=== PRIMEIRA REGRA (classe soft ou none) ===")
    ant = ", ".join(sorted(primeira["antecedents"]))
    con = ", ".join(sorted(primeira["consequents"]))
    print(f"SE   {ant}")
    print(f"ENTÃO {con}")
    print(f"Confiança: {primeira['confidence']:.2%} | Suporte: {primeira['support']:.2%} | Lift: {primeira['lift']:.4f}")
