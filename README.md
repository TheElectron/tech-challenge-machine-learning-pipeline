# ✈️ Atrasos e Cancelamentos de Voos | Pipeline de ML

Pipeline de machine learning end-to-end sobre atrasos em voos domésticos dos EUA (2015), cobrindo análise exploratória, engenharia de features, classificação supervisionada e clustering não supervisionado.

---

## Objetivo

1. Aprendizado Supervisionado: **Classificar** se um voo chegará atrasado (atraso ≥ 15 min).
2. Aprendizado Não Supervisionado: Uso de clustering para **Perfilar as principais causas de atraso**.

---

## Dataset

Voos domésticos dos EUA do [dataset Kaggle Flight Delays 2015](https://www.kaggle.com/datasets/usdot/flight-delays).

| Arquivo | Registros | Descrição |
|---|---|---|
| `flights.csv` | 5.874.020 | Registros com horários, clima e outras condições,   |
| `airlines.csv` | 16 | Relaciona o código IATA com o nome da companhia aérea|
| `airports.csv` | ~300 | Código IATA, cidade, estado, coordenadas |

**Coluna alvo:** `ARRIVAL_DELAY`, categórica binária (0 = pontual, 1 = atrasado)
**Distribuição das classes:** ≈ 81% pontual / 19% atrasado  
**Escopo:** apenas voos concluídos (`CANCELLED=0`, `DIVERTED=0`) — 5.714.008 registros

---

## Arquitetura

```
data/raw/*.csv
      │
      ▼
eda_flight_delays_and_cancellations.ipynb   ← Fase 1: EDA
      │
      ▼
feature_eng.ipynb                           ← Fase 2: Engenharia de Features
      │
      ▼  data/processed/flights_features.parquet
      │
      ├──────────────────────────┐
      ▼                          ▼
classifier.ipynb            clustering.ipynb
(Modelos Supervisionados)   (Não Supervisionado — K-Means)
      │                          │
      ▼                          ▼
models/logistic_regression  models/kmeans
models/random_forest        models/kmeans_preprocessor
models/gbt
```

---

## Pipeline

### Fase 1 — EDA (`eda_flight_delays_and_cancellations.ipynb`)

Principais tópicos:
- Forte desbalanceamento de classes 81% pontual e 19% atrasado.  
**Escopo:** apenas voos concluídos (CANCELLED=0, DIVERTED=0), 5.714.008 registros
- Colunas de causa de atraso possuem **nulos informativos**, pois o DOT só as preenche quando `ARRIVAL_DELAY ≥ 15`. Nulos informativos serão preenchidos com 0 para voos pontuais.
- `DEPARTURE_DELAY` é altamente correlacionado com `ARRIVAL_DELAY`, informação indisponível no momento de predição.
- Efeito cascata: atrasos se acumulam ao longo do dia, e sazonais anualmente.

### Fase 2 — Engenharia de Features (`feature_eng.ipynb`)

**Entrada:** `data/raw/*.csv`
**Saída:** `data/processed/flights_features.parquet`

**Split temporal:** treino = meses 1–10, teste = meses 11–12 (aplicado antes de qualquer agregado)

| Grupo de features | Features criadas |
|---|---|
| Temporais | `HORA_PARTIDA`, `HORA_CHEGADA_PROG`, `TURNO`, `IS_WEEKEND`, `IS_PEAK_MONTH` |
| Rota | `ROTA`, `FREQ_ROTA`*, `LOG_DISTANCE` |
| Históricas | `HIST_DELAY_AIRLINE`, `HIST_DELAY_ORIGIN`, `HIST_DELAY_DEST`, `HIST_DELAY_ROTA`, `HIST_DELAY_AIRLINE_DOW`, `HIST_DELAY_AIRLINE_TURNO` |
| Encoded | `AIRLINE_IDX`, `TURNO_IDX` (StringIndexer, fit apenas no treino) |
*`FREQ_ROTA` calculada apenas nos meses de treino; rotas inéditas no teste recebem a média do treino.

**Prevenção de leakage:** todos os agregados derivados do alvo (atrasos históricos) calculados exclusivamente nos dados de treino e depois unidos ao teste. `StringIndexer` ajustado apenas no treino.

### Fase 3 — Modelos Supervisionados (`classifier.ipynb`)

**Split:** treino = meses 1–10 (83,7%), teste = meses 11–12 (16,3%)

**Estratégia de desbalanceamento:**
- Regressão Logística e Random Forest: `weightCol` (frequência inversa das classes)
- GBT: oversampling da classe minoritária (não suporta `weightCol`)

**Baseline Dummy** Voo sempre pontual, Acurácia ≈ 81,39% Recall(Atrasado) = 0.

| Modelo | AUC-ROC | PR-AUC | F1-macro | F1 (cls 1) | Recall (1) | Acurácia |
|---|---|---|---|---|---|---|
| Regressão Logística | 0,6178 | 0,2471 | 0,6887 | 0,3235 | 0,4591 | 0,6552 |
| Random Forest | 0,6157 | 0,2416 | 0,7133 | 0,3052 | 0,3761 | 0,6925 |
| Gradient Boosted Trees | 0,6032 | 0,2360 | 0,6918 | 0,2996 | 0,4022 | 0,6624 |

Melhor modelo: Regressão Logística, pois detecta quase 46% dos casos positivos, superando os modelos baseados em árvore de decisão, neste recorte específico dos dados.
Evolução: Avaliar hiperparâmetros e alterar o *threshold* das árvores de decisão. Neste contexto, o que tem um custo maior, um Falso Positivo ou um Falso Negativo?

### Fase 4 — Clustering (`clustering.ipynb`)

**Escopo:** apenas voos atrasados 1.063.439 voos.

**Features:** 4 colunas de causa de atraso (minutos por causa):
`AIR_SYSTEM_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY`

> Nota: `SECURITY_DELAY` foi excluída, média < 0,1 min em todos os voos. Variância próxima a zero.

**Seleção de K** (WSSSE + silhueta em 10% da amostra):

| k | WSSSE | Silhouette |
|---|---|---|
| 2 | 462.072 | 0,913 |
| 4 | 335.268 | 0,598 |
| **5** | **318.455** | **0,799** |
| 6 | 258.797 | 0,780 |

K=5 possui o melhor equilíbrio entre redução de WSSSE e coesão dos clusters. Alem disso, corresponde aos 4 perfis de atraso esperados cascata, operacional, clima, rotina e outliers.

**Perfis dos clusters:**

- **Cluster 0 (Rotina):** É o grupo majoritário (~80% dos voos). Representa atrasos "normais" e mais curtos (média 38 min), pulverizados entre várias causas, mas dominados pelo Sistema Aéreo (*Air System*).
- **Cluster 1 (Operacional):** Atrasos significativos (quase 2 horas), causados por problemas operacionais diretos da companhia aérea (*Airline Delay*).
- **Cluster 2 (Clima):** Causa isolada e específica, atrasos longos (mais de 3h) causados quase exclusivamente por fatores meteorológicos (*Weather Delay*).
- **Cluster 3 (Cascata):** Atrasos de mais de 2 horas causados unicamente pelo avião anterior que atrasou (*Late Aircraft Delay*).
- **Cluster 4 (Outliers):** É o menor grupo, contém menos de 1% da base, mas representa os **atrasos extremos**, média de quase 7,5 horas.

---

## Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.13 |
| Processamento distribuído | PySpark 4.x |
| Machine Learning | Spark MLlib |
| Análise / Visualização | Pandas, NumPy, Matplotlib, Seaborn |
| Formato de dados | Parquet (colunar) |
| Ambiente | Jupyter Notebook |

---

## Configuração

```bash
git clone <repo-url>
cd tech-challenge-machine-learning-pipeline

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

**Requisito:** Java 17+

```bash
sudo apt install openjdk-17-jdk   # Ubuntu/Debian
```

Baixe o dataset do Kaggle e coloque os arquivos em `data/raw/`, depois execute os notebooks na ordem:

```
1. eda_flight_delays_and_cancellations.ipynb
2. feature_eng.ipynb
3. classifier.ipynb   (independente do 4)
4. clustering.ipynb   (independente do 3)
```

---

## Roadmap

- [x] Fase 1 — Análise Exploratória de Dados
- [x] Fase 2 — Engenharia de Features (sem leakage, split temporal)
- [x] Fase 3 — Modelos Supervisionados (LR, Random Forest, GBT + otimização de limiar)
- [x] Fase 4 — Clustering Não Supervisionado (K-Means, k=5)
- [ ] Fase 5 — Deploy & MLOps (Model Registry, REST API, Docker, monitoramento)

---
## 🚀 Entregaveis

- [Vídeo](https://drive.google.com/file/d/1hCOfiq_-BBuEqhHhyaceQZEAz1ZxYyku/view?usp=sharing)

## Licença

MIT — veja [LICENSE](LICENSE).
