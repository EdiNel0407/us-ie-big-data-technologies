#!/usr/bin/env bash
set -euo pipefail
: "${PUBSUB_EMULATOR_HOST:?Set PUBSUB_EMULATOR_HOST (e.g., localhost:8085) before running}"

PROJECT_ID="demo-project"
TOPIC="usage-topic"
SUB="usage-sub"

gcloud config set project "$PROJECT_ID" >NUL 2>&1 || true
gcloud pubsub topics create "$TOPIC"
gcloud pubsub subscriptions create "$SUB" --topic="$TOPIC" --ack-deadline=60
echo "Created: topic=$TOPIC  subscription=$SUB  project=$PROJECT_ID"
