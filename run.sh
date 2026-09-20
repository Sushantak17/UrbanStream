#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  UrbanStream — One-command launcher
#  Usage:  ./run.sh              (full pipeline: Docker + producers + Spark + ML + dashboard)
#          ./run.sh demo         (dashboard only — no Docker needed, synthetic data)
#          ./run.sh stop         (stop everything)
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
R='\033[0;31m' G='\033[0;32m' Y='\033[0;33m' B='\033[0;34m' P='\033[0;35m'
C='\033[0;36m' W='\033[1;37m' DIM='\033[2m' RESET='\033[0m'

LOG_DIR="$SCRIPT_DIR/.logs"
mkdir -p "$LOG_DIR"

# ── Helpers ───────────────────────────────────────────────────────────

banner() {
  echo ""
  echo -e "${C}  ╔═══════════════════════════════════════════════╗${RESET}"
  echo -e "${C}  ║${W}   ⬡  UrbanStream v5.0                        ${C}║${RESET}"
  echo -e "${C}  ║${DIM}   Multi-Agent AI · LSTM · LinUCB · IForest    ${C}║${RESET}"
  echo -e "${C}  ╚═══════════════════════════════════════════════╝${RESET}"
  echo ""
}

step() { echo -e "\n${G}━━━ [$1/${TOTAL_STEPS}] $2${RESET}"; }
info() { echo -e "    ${DIM}$1${RESET}"; }
ok()   { echo -e "    ${G}✓ $1${RESET}"; }
warn() { echo -e "    ${Y}⚠ $1${RESET}"; }
fail() { echo -e "    ${R}✗ $1${RESET}"; }

wait_for_healthy() {
  local service="$1" max_wait="${2:-120}" elapsed=0
  while [ $elapsed -lt $max_wait ]; do
    if docker compose ps "$service" 2>/dev/null | grep -q "healthy"; then
      return 0
    fi
    sleep 3
    elapsed=$((elapsed + 3))
    printf "    ${DIM}waiting... %ds / %ds${RESET}\r" "$elapsed" "$max_wait"
  done
  echo ""
  return 1
}

cleanup_pids() {
  # Kill background processes on exit
  if [ -f "$LOG_DIR/pids" ]; then
    while IFS= read -r pid; do
      kill "$pid" 2>/dev/null || true
    done < "$LOG_DIR/pids"
    rm -f "$LOG_DIR/pids"
  fi
}

# ── STOP ──────────────────────────────────────────────────────────────

do_stop() {
  banner
  echo -e "${Y}Stopping UrbanStream...${RESET}"

  # Kill Python background processes
  if [ -f "$LOG_DIR/pids" ]; then
    while IFS= read -r pid; do
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null && echo -e "  ${DIM}killed PID $pid${RESET}" || true
      fi
    done < "$LOG_DIR/pids"
    rm -f "$LOG_DIR/pids"
  fi

  # Also kill by name as fallback
  pkill -f "run_all_producers.py" 2>/dev/null || true
  pkill -f "ml/recommender.py" 2>/dev/null || true
  pkill -f "agents/run_agents.py" 2>/dev/null || true
  pkill -f "streamlit run dashboard" 2>/dev/null || true

  # Stop Docker
  if command -v docker &>/dev/null; then
    docker compose down 2>/dev/null || true
  fi

  echo -e "\n${G}✓ Everything stopped.${RESET}"
  echo -e "${DIM}  To also remove data volumes: docker compose down -v${RESET}\n"
  exit 0
}

# ── DEMO MODE ─────────────────────────────────────────────────────────

do_demo() {
  banner
  TOTAL_STEPS=2
  step 1 "Installing Python dependencies"
  pip3 install -q -r requirements.txt 2>"$LOG_DIR/pip.log" && ok "Dependencies installed" || warn "Some deps may have failed (check .logs/pip.log)"

  step 2 "Launching dashboard (synthetic data mode)"
  info "No Docker required — dashboard uses built-in synthetic data"
  info "Open → http://localhost:8501"
  echo ""
  exec streamlit run dashboard/dashboard.py --server.port 8501 --server.headless true
}

# ── FULL PIPELINE ─────────────────────────────────────────────────────

do_full() {
  banner
  TOTAL_STEPS=8
  trap cleanup_pids EXIT
  > "$LOG_DIR/pids"  # reset PID file

  # ── Step 1: Check prerequisites ──────────────────────────────────
  step 1 "Checking prerequisites"

  if ! command -v docker &>/dev/null; then
    fail "Docker not found. Install Docker Desktop first."
    echo -e "    ${DIM}https://www.docker.com/products/docker-desktop/${RESET}"
    exit 1
  fi
  ok "Docker found"

  if ! docker info &>/dev/null; then
    fail "Docker daemon is not running. Start Docker Desktop first."
    exit 1
  fi
  ok "Docker daemon running"

  if ! command -v python3 &>/dev/null; then
    fail "Python 3 not found."
    exit 1
  fi
  ok "Python 3 found ($(python3 --version 2>&1 | awk '{print $2}'))"

  # Check datasets
  MISSING_DATA=0
  for f in data/nyc_traffic.csv data/openaq_nyc.csv data/weather_nyc.csv; do
    if [ ! -f "$f" ]; then
      warn "Missing: $f"
      MISSING_DATA=1
    fi
  done
  if [ $MISSING_DATA -eq 1 ]; then
    warn "Some datasets are missing — see README.md Step 1 for download instructions"
    warn "The dashboard will still work with synthetic data"
  else
    ok "All datasets present"
  fi

  # ── Step 2: Install Python dependencies ──────────────────────────
  step 2 "Installing Python dependencies"
  pip3 install -q -r requirements.txt 2>"$LOG_DIR/pip.log" && ok "Dependencies installed" || warn "Some deps may have failed (check .logs/pip.log)"

  # ── Step 3: Start Docker services ────────────────────────────────
  step 3 "Starting Docker services (Redpanda, Spark, MinIO, Redis)"
  docker compose up -d 2>&1 | tail -5
  info "Waiting for services to become healthy..."

  for svc in redis minio redpanda spark-master spark-worker; do
    if wait_for_healthy "$svc" 120; then
      ok "$svc is healthy"
    else
      warn "$svc may not be fully ready yet (continuing anyway)"
    fi
  done

  echo ""
  info "Service URLs:"
  info "  Redpanda Console  → http://localhost:8080"
  info "  Spark Master UI   → http://localhost:8888"
  info "  MinIO Console     → http://localhost:9001  (minioadmin/minioadmin)"
  info "  Redis             → localhost:6379"

  # ── Step 4: Train ML models (if not already trained) ─────────────
  step 4 "Training ML models (if needed)"

  if [ -f "ml/models/aqi_lstm.pt" ]; then
    ok "LSTM model already trained"
  else
    info "Training LSTM AQI forecaster..."
    python3 ml/aqi_forecaster.py > "$LOG_DIR/lstm_train.log" 2>&1 && ok "LSTM trained" || warn "LSTM training failed (check .logs/lstm_train.log)"
  fi

  if [ -f "ml/models/anomaly_iforest.joblib" ]; then
    ok "Anomaly detector already trained"
  else
    info "Training anomaly detector..."
    python3 ml/anomaly_detector.py > "$LOG_DIR/anomaly_train.log" 2>&1 && ok "Anomaly detector trained" || warn "Training failed (check .logs/anomaly_train.log)"
  fi

  if [ -f "ml/models/cluster_labels.json" ]; then
    ok "Offline KMeans already trained"
  else
    info "Training offline KMeans..."
    python3 ml/clustering.py > "$LOG_DIR/clustering.log" 2>&1 && ok "KMeans trained" || warn "Training failed (check .logs/clustering.log)"
  fi

  # ── Step 5: Start Kafka producers ────────────────────────────────
  step 5 "Starting Kafka producers (4 topics)"
  KAFKA_BROKER=localhost:9092 python3 producers/run_all_producers.py \
    > "$LOG_DIR/producers.log" 2>&1 &
  echo $! >> "$LOG_DIR/pids"
  ok "Producers running in background (PID $(tail -1 "$LOG_DIR/pids"))"
  info "Log: .logs/producers.log"

  # ── Step 6: Install Python deps in Spark + submit streaming job ──
  step 6 "Preparing Spark container & submitting streaming job"
  info "Installing Python packages inside Spark containers..."
  docker exec urbanstream-spark-master pip install -q redis scikit-learn joblib numpy 2>"$LOG_DIR/spark_pip_master.log" \
    && ok "Spark master: redis + sklearn installed" || warn "pip install failed on master (check .logs/spark_pip_master.log)"
  docker exec urbanstream-spark-worker pip install -q redis scikit-learn joblib numpy 2>"$LOG_DIR/spark_pip_worker.log" \
    && ok "Spark worker: redis + sklearn installed" || warn "pip install failed on worker (check .logs/spark_pip_worker.log)"

  info "Submitting Spark streaming job..."
  docker exec urbanstream-spark-master /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --executor-cores 2 \
    --executor-memory 1g \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.367 \
    --py-files /opt/spark/jobs/streaming_kmeans.py \
    --conf spark.sql.shuffle.partitions=8 \
    --conf spark.executor.memory=1g \
    --conf spark.hadoop.fs.s3a.endpoint=http://minio:9000 \
    --conf spark.hadoop.fs.s3a.access.key=minioadmin \
    --conf spark.hadoop.fs.s3a.secret.key=minioadmin \
    --conf spark.hadoop.fs.s3a.path.style.access=true \
    --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem \
    --conf spark.hadoop.fs.s3a.connection.ssl.enabled=false \
    /opt/spark/jobs/stream_processor.py \
    > "$LOG_DIR/spark.log" 2>&1 &
  SPARK_PID=$!
  echo $SPARK_PID >> "$LOG_DIR/pids"
  ok "Spark job submitted (PID $SPARK_PID)"
  info "Log: .logs/spark.log"
  info "Spark UI: http://localhost:8888"

  # ── Step 7: Start ML recommender + agents ────────────────────────
  step 7 "Starting ML recommender & multi-agent system"

  REDIS_HOST=localhost python3 ml/recommender.py \
    > "$LOG_DIR/recommender.log" 2>&1 &
  echo $! >> "$LOG_DIR/pids"
  ok "Recommender running (PID $(tail -1 "$LOG_DIR/pids"))"

  python3 agents/run_agents.py \
    > "$LOG_DIR/agents.log" 2>&1 &
  echo $! >> "$LOG_DIR/pids"
  ok "Multi-agent system running (PID $(tail -1 "$LOG_DIR/pids"))"
  info "Logs: .logs/recommender.log, .logs/agents.log"

  # ── Step 8: Launch dashboard ─────────────────────────────────────
  step 8 "Launching dashboard"
  info "Opening → http://localhost:8501"
  echo ""
  echo -e "${G}═══════════════════════════════════════════════════${RESET}"
  echo -e "${W}  UrbanStream is running!${RESET}"
  echo -e "${G}═══════════════════════════════════════════════════${RESET}"
  echo ""
  echo -e "  ${C}Dashboard${RESET}        → ${W}http://localhost:8501${RESET}"
  echo -e "  ${C}Spark UI${RESET}         → ${W}http://localhost:8888${RESET}"
  echo -e "  ${C}Redpanda Console${RESET} → ${W}http://localhost:8080${RESET}"
  echo -e "  ${C}MinIO Console${RESET}    → ${W}http://localhost:9001${RESET}"
  echo ""
  echo -e "  ${DIM}Stop everything:  ./run.sh stop${RESET}"
  echo -e "  ${DIM}Background logs:  ls .logs/${RESET}"
  echo ""

  # Run dashboard in foreground (Ctrl+C stops it + triggers cleanup)
  exec streamlit run dashboard/dashboard.py --server.port 8501 --server.headless true
}

# ── MAIN ──────────────────────────────────────────────────────────────

case "${1:-}" in
  stop)  do_stop ;;
  demo)  do_demo ;;
  *)     do_full ;;
esac
