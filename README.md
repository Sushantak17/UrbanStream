# 🌆 UrbanStream

> **Real-Time Pollution-Aware Routing for Gig Workers**  
> Apache Spark · Redpanda (Kafka) · Redis · MinIO · PyTorch · Multi-Agent AI

A real-time urban air quality monitoring and worker safety platform for New York City. Ingests live traffic, pollution, and weather data through a streaming pipeline, applies 4 ML models for intelligent decision-making, and orchestrates worker relocations via a multi-agent system.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                                   │
│   NYC Traffic (13M rows)  ·  OpenAQ Pollution (4M rows)  ·  Weather      │
└──────────┬───────────────────────────┬──────────────────────┬────────────┘
           │                           │                      │
           ▼                           ▼                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    KAFKA PRODUCERS (4 threads)                            │
│   traffic_stream · pollution_stream · weather_stream · worker_stream     │
│   Confluent Kafka · CSV replay · 50 simulated gig workers                │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  REDPANDA (Kafka-compatible broker)                       │
│                  4 topics · 3 partitions each                             │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│             SPARK STRUCTURED STREAMING (Master + Worker)                  │
│   3 streaming queries · foreachBatch · 30s tumbling windows               │
│   Zone scoring · Exposure tracking · 2-min watermark                      │
│   Checkpoints → MinIO (S3)  ·  Results → Redis                           │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
             ┌────────────┐         ┌─────────────┐
             │   Redis    │         │  MinIO (S3)  │
             │ State Store│         │ Checkpoints  │
             └──────┬─────┘         │ Data Lake    │
                    │               └──────────────┘
          ┌─────────┼──────────┐
          ▼         ▼          ▼
┌──────────────────────────────────────────────────────────────────────┐
│              MULTI-AGENT SYSTEM (4 Agents + Coordinator)             │
│                                                                      │
│   Monitor (15s)  →  Forecaster (30s)  →  Router (30s)  →  Alert (30s)│
│   IsolationForest    LSTM                 LinUCB Bandit    Briefings  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                                │
│   Real-time monitoring · Worker exposure feed · Relocate dispatch     │
│   Zone clustering map · Agent status · ML model metrics               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ML Models

| Model | Algorithm | Purpose | Key Details |
|---|---|---|---|
| **Zone Clustering** | K-Means (K=4) | Classify 30 zones into risk profiles | 7 features, silhouette-validated, 4 clusters: Safe Corridor, Weather Sensitive, Peak Hour Hazardous, Permanently Hazardous |
| **AQI Forecaster** | 2-layer LSTM (PyTorch) | Predict next 3 AQI time steps | Input: 24 historical readings, 32.5% better than AR(3) baseline |
| **Anomaly Detector** | Isolation Forest | Detect unusual zone states | 5 features, 5% contamination, generates human-readable explanations |
| **Worker Router** | LinUCB Contextual Bandit | Optimal worker-to-zone routing | 13-dim context, 30 arms, exploration-exploitation with cold-start fallback |

---

## Multi-Agent System

| Agent | Interval | ML Model | Role |
|---|---|---|---|
| **Monitor** | 15s | Isolation Forest | Scans zones for anomalies |
| **Forecaster** | 30s | LSTM | Predicts AQI trends |
| **Router** | 30s | LinUCB Bandit | Assigns workers to safe zones |
| **Alert** | 30s | Rule-based | Generates safety briefings |
| **Coordinator** | 15s | — | Orchestrates lifecycle, restarts dead agents |

---

## Quick Start

### Prerequisites
- Docker Desktop (≥ 8 GB RAM allocated)
- Python 3.10+

### One-Command Launch

```bash
# Download data + start everything
chmod +x run.sh
./run.sh
```

This will:
1. Start Docker services (Redpanda, Spark, Redis, MinIO)
2. Install Python dependencies
3. Train ML models (KMeans, LSTM, Isolation Forest)
4. Start Kafka producers (4 data streams)
5. Submit Spark streaming job
6. Launch recommender + multi-agent system
7. Open the Streamlit dashboard at **http://localhost:8501**

### Manual Step-by-Step

<details>
<summary>Click to expand manual setup</summary>

#### 1. Download Datasets

```bash
python3 download_data.py
```

Or manually download:
- **NYC Traffic**: [NYC OpenData Traffic Speed](https://data.cityofnewyork.us/Transportation/Traffic-Speed/4h9m-uh3q) → `data/nyc_traffic.csv`
- **OpenAQ Pollution**: [OpenAQ](https://openaq.org/data/) (NYC, pm25 + no2) → `data/openaq_nyc.csv`
- **Weather**: [Open-Meteo](https://open-meteo.com/) (NYC, 2023) → `data/weather_nyc.csv`

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Start Docker Services

```bash
docker compose up -d
# Wait ~60s for health checks to pass
docker compose ps
```

#### 4. Train ML Models

```bash
python3 ml/clustering.py           # KMeans zone clustering
python3 ml/aqi_forecaster.py       # LSTM AQI forecaster
python3 ml/anomaly_detector.py     # Isolation Forest
```

#### 5. Start Producers

```bash
KAFKA_BROKER=localhost:9092 python3 producers/run_all_producers.py
```

#### 6. Submit Spark Job

```bash
docker exec urbanstream-spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.367 \
  --py-files /opt/spark/jobs/streaming_kmeans.py \
  --conf spark.hadoop.fs.s3a.endpoint=http://minio:9000 \
  --conf spark.hadoop.fs.s3a.access.key=minioadmin \
  --conf spark.hadoop.fs.s3a.secret.key=minioadmin \
  --conf spark.hadoop.fs.s3a.path.style.access=true \
  --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
  --conf spark.hadoop.fs.s3a.connection.ssl.enabled=false \
  /opt/spark/jobs/stream_processor.py
```

#### 7. Start ML + Agents

```bash
REDIS_HOST=localhost python3 ml/recommender.py &
REDIS_HOST=localhost python3 agents/run_agents.py &
```

#### 8. Launch Dashboard

```bash
REDIS_HOST=localhost streamlit run dashboard/dashboard.py --server.port 8501
```

</details>

---

## Service URLs

| Service | URL |
|---|---|
| **Dashboard** | http://localhost:8501 |
| Redpanda Console | http://localhost:8080 |
| Spark Master UI | http://localhost:8888 |
| MinIO Console | http://localhost:9001 (`minioadmin` / `minioadmin`) |
| Redis | `localhost:6379` |

---

## Project Structure

```
urbanstream/
├── run.sh                          ← One-command launcher
├── docker-compose.yml              ← 8 Docker services
├── requirements.txt                ← Python dependencies
├── download_data.py                ← Dataset downloader
│
├── producers/
│   └── run_all_producers.py        ← 4-thread Kafka producer
│
├── spark/
│   ├── stream_processor.py         ← Spark Structured Streaming (3 queries)
│   └── streaming_kmeans.py         ← Real-time KMeans inside Spark
│
├── ml/
│   ├── clustering.py               ← Offline KMeans (K=4) zone clustering
│   ├── aqi_forecaster.py           ← PyTorch 2-layer LSTM
│   ├── anomaly_detector.py         ← Isolation Forest + explanation gen
│   ├── bandit_router.py            ← LinUCB contextual bandit (30 arms)
│   ├── recommender.py              ← Unified recommendation engine
│   └── models/                     ← Pre-trained model artifacts
│       ├── kmeans_zones.joblib
│       ├── scaler_zones.joblib
│       ├── cluster_labels.json
│       ├── aqi_lstm.pt
│       ├── aqi_lstm_metrics.json
│       ├── anomaly_iforest.joblib
│       ├── anomaly_scaler.joblib
│       └── anomaly_metrics.json
│
├── agents/
│   ├── base_agent.py               ← Abstract base (perceive → decide → act)
│   ├── monitor_agent.py            ← Anomaly detection agent (15s)
│   ├── forecaster_agent.py         ← LSTM prediction agent (30s)
│   ├── router_agent.py             ← LinUCB routing agent (30s)
│   ├── alert_agent.py              ← Safety briefing agent (30s)
│   ├── coordinator.py              ← Supervisor — lifecycle management
│   └── run_agents.py               ← Agent launcher
│
├── dashboard/
│   └── dashboard.py                ← Streamlit dashboard (3 tabs)
│
├── tests/
│   └── test_urbanstream.py         ← 33 tests (pytest)
│
├── clustering_evaluation.ipynb     ← Elbow, silhouette, PCA analysis
└── data/                           ← (gitignored) Downloaded CSVs
    ├── nyc_traffic.csv             ← 13.4 MB
    ├── openaq_nyc.csv              ← 3.9 MB
    └── weather_nyc.csv             ← 256 KB
```

---

## Configuration

### Producer Speed

| Variable | Default | Description |
|---|---|---|
| `TRAFFIC_SPEED` | 500 | Traffic records/sec |
| `POLLUTION_SPEED` | 200 | Pollution records/sec |
| `WEATHER_SPEED` | 10 | Weather records/sec |
| `NUM_WORKERS` | 50 | Simulated gig workers |
| `MOVE_INTERVAL` | 30 | Seconds between worker zone changes |

### Exposure Thresholds

| Variable | Default | Description |
|---|---|---|
| `HIGH_AQI` | 100 | WHO threshold — unhealthy for sensitive groups |
| `WARN_HRS` | 1.5 | Hours in high-AQI zone → WARNING |
| `CRIT_HRS` | 3.0 | Hours in high-AQI zone → CRITICAL |

---

## Testing

```bash
pytest tests/ -v
# 33 tests covering: KMeans, LSTM, IForest, LinUCB, agents, pipeline, integration
```

---

## Troubleshooting

**Redpanda not starting?**
```bash
docker compose logs redpanda
# Increase Docker memory to ≥8GB in Docker Desktop settings
```

**Spark job fails with S3A errors?**
```bash
curl http://localhost:9000/minio/health/live
docker logs urbanstream-minio-init
```

**Dashboard shows no data?**
```bash
redis-cli -h localhost ping
# Dashboard shows synthetic fallback data even without live pipeline
# Real data flows once Spark + producers are running
```

---

## Stopping

```bash
docker compose down          # Stop services (keep data)
docker compose down -v       # Stop + delete all volumes
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Message Broker | Redpanda (Kafka-compatible) |
| Stream Processing | Apache Spark 3.5.3 Structured Streaming |
| State Store | Redis 7.2 |
| Object Storage | MinIO (S3-compatible) |
| ML Framework | PyTorch, scikit-learn |
| Agent System | Custom Python multi-agent (perceive-decide-act) |
| Dashboard | Streamlit + Plotly + PyDeck |
| Infrastructure | Docker Compose (8 services) |
