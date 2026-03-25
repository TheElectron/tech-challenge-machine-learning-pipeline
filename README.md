# ✈️ Flight Delays and Cancellations | ML Pipeline

Pipeline completo de ciência de dados sobre atrasos de voos nos EUA (2015), cobrindo análise exploratória, engenharia de features, modelos supervisionados e não supervisionados.

---

## Objetivo

Construir um pipeline end-to-end capaz de:
1. **Prever** se um voo chegará atrasado (≥ 15 min) — classificação supervisionada
2. **Identificar perfis de atraso** — agrupamento não supervisionado por causa

---

## Dataset

Dados públicos de voos domésticos nos EUA, disponíveis no [Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays).

| Arquivo | Tipo | Registros | Descrição |
|---|---|---|---|
| `flights.csv` | Tabela Fato | 5.874.020 | Um registro por voo: horários, atrasos, causas, rota |
| `airlines.csv` | Dimensão | 16 | Código IATA → nome da companhia |
| `airports.csv` | Dimensão | ~300 | Código IATA, cidade, estado, coordenadas |

**Variável-alvo:** `ARRIVAL_DELAY` binarizada → `LABEL` (0 = pontual, 1 = atrasado ≥ 15 min)
**Distribuição:** ≈ 81% pontual / 19% atrasado (dataset desbalanceado)

---

## Estrutura do Projeto

```
.
├── data/
│   ├── raw/                        # Dados originais (não versionados)
│   │   ├── flights.csv
│   │   ├── airlines.csv
│   │   └── airports.csv
│   └── processed/
│       └── flights_features.parquet   # Saída do feature_eng.ipynb
│
├── notebooks/
│   ├── eda_flight_delays_and_cancellations.ipynb   # Fase 1 — EDA
│   ├── feature_eng.ipynb                           # Fase 2 — Limpeza e Features
│   ├── classifier.ipynb                            # Fase 3 — Modelos Supervisionados
│   └── clustering.ipynb                            # Fase 4 — Clustering
│
├── models/                         # Modelos treinados (Spark ML PipelineModel)
│   ├── logistic_regression/
│   ├── random_forest/
│   ├── gbt/
│   ├── kmeans/
│   └── kmeans_preprocessor/
│
├── docs/                           # Documentação e dicionário de dados
├── src/                            # Módulos Python reutilizáveis (em desenvolvimento)
├── deploy/                         # Configurações de deploy (em desenvolvimento)
├── conf/                           # Configurações do projeto
└── requirements.txt
```

---

## Pipeline

Os notebooks devem ser executados na seguinte ordem:

```
eda_flight_delays_and_cancellations.ipynb
        ↓
feature_eng.ipynb  →  data/processed/flights_features.parquet
        ↓                         ↓
classifier.ipynb            clustering.ipynb
        ↓
    models/
```

### Fase 1 — EDA (`eda_flight_delays_and_cancellations.ipynb`)
Análise exploratória completa: distribuições, padrões temporais, análise por companhia/aeroporto, correlações e identificação de riscos de leakage.

**Principais achados:**
- Forte desbalanceamento de classes (81/19)
- Colunas de causa de atraso têm nulos informativos (DOT não preenche para voos pontuais)
- `DEPARTURE_DELAY` tem alta correlação com `ARRIVAL_DELAY` → risco de leakage
- Padrão "bola de neve": atrasos se acumulam ao longo do dia

### Fase 2 — Feature Engineering (`feature_eng.ipynb`)
**Input:** `data/raw/*.csv` → **Output:** `data/processed/flights_features.parquet`

- Filtragem do escopo: voos concluídos com `ARRIVAL_DELAY` informado (5.714.008 registros)
- Tratamento de nulos nas colunas de causa: confirmado que nulos = voo pontual → `fillna(0)`
- Criação de 16 features preditivas: temporais, de rota e históricas
- **Correção de leakage**: features históricas calculadas exclusivamente no treino (meses 1-10) e aplicadas no teste via join — grupos ausentes recebem média global do treino
- Encoding com `StringIndexer` ajustado apenas no treino

### Fase 3 — Modelos Supervisionados (`classifier.ipynb`)
**Input:** `flights_features.parquet` → **Output:** modelos em `models/`

Split temporal: treino = meses 1-10 (83,7%), teste = meses 11-12 (16,3%)

| Modelo | AUC-ROC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.6279 | 0.692 | 0.7502 | 0.6587 | 0.6587 |
| Random Forest | 0.6213 | 0.720 | 0.7425 | 0.7029 | 0.7029 |
| Gradient Boosted Trees | — | — | — | — | — |

Estratégia anti-desbalanceamento:
- LR e RF: `weightCol` com pesos inversamente proporcionais à frequência da classe
- GBT: oversampling da classe minoritária (~4,35×)

### Fase 4 — Clustering (`clustering.ipynb`)
**Input:** `flights_features.parquet` → **Output:** modelo em `models/kmeans`

Agrupa os **voos atrasados** por perfil de causa usando K-Means.

Features: `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY`

Perfis esperados:

| Cluster | Causa dominante |
|---|---|
| Efeito cascata | `LATE_AIRCRAFT_DELAY` |
| Operacional | `AIRLINE_DELAY` |
| Climático | `WEATHER_DELAY` |
| Sistêmico | `AIR_SYSTEM_DELAY` |

K ótimo determinado por curva de cotovelo (WSSSE) + coeficiente de silhueta em amostra de 10%.

---

## Tech Stack

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.13 |
| Processamento distribuído | PySpark 4.x |
| Machine Learning | Spark MLlib |
| Análise / Visualização | Pandas, NumPy, Matplotlib, Seaborn |
| Formato de dados | Parquet (compressão columnar) |
| Ambiente | Jupyter Notebook |

---

## Setup

```bash
# 1. Clonar o repositório
git clone <repo-url>
cd tech-challenge-machine-learning-pipeline

# 2. Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Baixar os dados
# Faça o download em: https://www.kaggle.com/datasets/usdot/flight-delays
# Coloque os arquivos em data/raw/

# 5. Executar os notebooks na ordem indicada no pipeline acima
jupyter notebook notebooks/
```

**Requisito:** Java 17+ instalado (necessário para o PySpark)

```bash
sudo apt install openjdk-17-jdk   # Ubuntu/Debian
```

---

## Roadmap

- [x] **Fase 1** — Análise Exploratória dos Dados (EDA)
- [x] **Fase 2** — Limpeza, Tratamento de Nulos e Feature Engineering
- [x] **Fase 3** — Modelos Supervisionados (LR, Random Forest, GBT)
- [x] **Fase 4** — Modelo Não Supervisionado (K-Means clustering)
- [ ] **Fase 5** — Deploy e MLOps (Model Registry, API REST, Docker, monitoramento)

---

## Licença

Distribuído sob a licença MIT. Veja [LICENSE](LICENSE) para mais detalhes.
