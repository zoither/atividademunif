# Trabalho - Machine Learning / Gripe

Alunos:

Augusto Ogawa - Ra: 23175970-2

Vinicius Nunes - Ra: 23000808-2


Implementações dos algoritmos de classificação usando o dataset da pesquisa de gripe.

## Arquivos

| Arquivo | Algoritmo |
|---|---|
| `knn_gripe.py` | KNN - K-Nearest Neighbors |
| `naive_bayes_gripe.py` | Naive Bayes (Gaussian) |
| `arvore_decisao_gripe.py` | Árvore de Decisão (entropy, max_depth=4) |
| `regras_gripe.py` | Regras de Associação (Apriori via mlxtend) |

## Dataset

Pesquisa de gripe disponível em:
https://docs.google.com/spreadsheets/d/1g1aQ61vijh6uHJuc8sijeBEMsoIQ2a5yLwUK04Wptlg/export?format=csv

## Instalação

```bash
pip install pandas numpy scikit-learn matplotlib mlxtend
```

## Como executar

```bash
python knn_gripe.py
python naive_bayes_gripe.py
python arvore_decisao_gripe.py
python regras_gripe.py
```

## Atributos do Dataset

| Coluna | Descrição |
|---|---|
| `gripe_ano_passado` | **Classe alvo** — ficou gripado? |
| `vacina` | Tomou vacina? |
| `ambientes_cheios` | Frequentou ambientes cheios? |
| `viajou` | Viajou mais de 100km? |
| `alergia` | Tem alergia nas vias aéreas? |
| `horas_sono` | Horas de sono por noite |
| `exercicio` | Praticou atividade física? |
| `alimentacao` | Alimentação balanceada? |
| `lavagem_maos` | Vezes que lavou as mãos/dia |
| `estresse` | Nível de estresse |
