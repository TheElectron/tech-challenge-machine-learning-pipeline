# ✈️ Atrasos e Cancelamentos de Voos | Pipeline de ML

Pipeline de machine learning end-to-end sobre atrasos em voos domésticos dos EUA (2015), cobrindo análise exploratória, engenharia de features, classificação supervisionada e clustering não supervisionado.

---

## Objetivo

1. **Classificar** se um voo chegará atrasado (≥ 15 min) — aprendizado supervisionado
2. **Perfilar causas de atraso** — clustering não supervisionado

**Momento de predição:** no horário de partida programado, usando apenas informações pré-voo (sem atraso de partida nem colunas operacionais preenchidas após a decolagem).

---

## Dataset

Voos domésticos dos EUA do [dataset Kaggle Flight Delays 2015](https://www.kaggle.com/datasets/usdot/flight-delays).

| Arquivo | Registros | Descrição |
|---|---|---|
| `flights.csv` | 5.874.020 | Um registro por voo: horários, atrasos, rota |
| `airlines.csv` | 16 | Código IATA → nome da companhia |
| `airports.csv` | ~300 | Código IATA, cidade, estado, coordenadas |

**Alvo:** `ARRIVAL_DELAY` binarizado → `LABEL` (0 = pontual, 1 = atrasado ≥ 15 min)  
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

Principais achados:
- Forte desbalanceamento de classes (81/19)
- Colunas de causa de atraso possuem **nulos informativos**: o DOT só as preenche quando `ARRIVAL_DELAY ≥ 15` → preenchidos com 0 para voos pontuais
- `DEPARTURE_DELAY` é altamente correlacionado com `ARRIVAL_DELAY` → excluído (informação pós-partida, indisponível no momento de predição)
- Efeito cascata: atrasos se acumulam ao longo do dia

### Fase 2 — Engenharia de Features (`feature_eng.ipynb`)

**Entrada:** `data/raw/*.csv` → **Saída:** `data/processed/flights_features.parquet`

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
- GBT: oversampling da classe minoritária (~4,35×, GBT não suporta `weightCol`)

**Baseline ingênuo** (sempre prediz pontual): acurácia ≈ 0,8139, recall(1) = 0.

| Modelo | AUC-ROC | PR-AUC | F1-macro | F1 (cls 1) | Recall (1) | Acurácia |
|---|---|---|---|---|---|---|
| Regressão Logística | 0,6178 | 0,2471 | 0,6887 | 0,3235 | 0,4591 | 0,6552 |
| Random Forest | 0,6157 | 0,2416 | 0,7133 | 0,3052 | 0,3761 | 0,6925 |
| Gradient Boosted Trees | 0,6032 | 0,2360 | 0,6918 | 0,2996 | 0,4022 | 0,6624 |

🏆 Melhor modelo (AUC-ROC): Regressão Logística

A otimização de limiar é aplicada no melhor modelo após o treinamento para maximizar o F1 na classe de atraso (o padrão 0,5 é subótimo dado o desbalanceamento).

### Fase 4 — Clustering (`clustering.ipynb`)

**Escopo:** apenas voos atrasados (LABEL=1, 1.063.439 voos)

**Features:** 4 colunas de causa de atraso (minutos por causa):
`AIR_SYSTEM_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY`

> `SECURITY_DELAY` excluída: média < 0,1 min em todos os voos — variância próxima de zero.

**Seleção de K** (WSSSE + silhueta em 10% da amostra):

| k | WSSSE | Silhouette |
|---|---|---|
| 2 | 462.072 | 0,913 — apenas 2 grupos grosseiros |
| 4 | 335.268 | 0,598 |
| **5** | **318.455** | **0,799** ← selecionado |
| 6 | 258.797 | 0,780 |

K=5 selecionado: melhor equilíbrio entre redução de WSSSE e coesão dos clusters, e corresponde aos 4 perfis de atraso esperados (cascata, operacional, clima, espaço aéreo) mais um grupo misto/severo.

**Perfis esperados dos clusters:**

| Perfil | Causa dominante |
|---|---|
| Cascata | `LATE_AIRCRAFT_DELAY` |
| Operacional | `AIRLINE_DELAY` |
| Clima | `WEATHER_DELAY` |
| Espaço aéreo | `AIR_SYSTEM_DELAY` |
| Misto / Severo | múltiplas causas |

---

## Limitações Conhecidas

| Limitação | Impacto |
|---|---|
| Features excluem `DEPARTURE_DELAY` | Teto de AUC-ROC ~0,63; modelo ainda é acionável pré-partida |
| Dados de 2015 apenas | O modelo requer retreinamento para mudanças estruturais (novas companhias, padrões pós-COVID) |
| Features históricas são estáticas | `HIST_DELAY_*` ficam desatualizadas conforme o desempenho das companhias muda; precisam de recomputo periódico |
| K-Means em distribuições de atraso desbalanceadas | Clusters dominantes são possíveis; k=5 foi escolhido para mitigar, mas os tamanhos por cluster devem ser monitorados |
| Sem intervalos de confiança nas métricas | Apenas estimativas pontuais; variância é baixa dado ~900k registros de teste |

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

- [Vídeo](https://drive.google.com/drive/folders/12yRYJUiJujpUMznBTROywDhfxgMcZA0G?usp=sharing)

## Licença

MIT — veja [LICENSE](LICENSE).
