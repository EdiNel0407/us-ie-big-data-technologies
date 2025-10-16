#!/usr/bin/env bash
set -euo pipefail
echo "Starting Pub/Sub emulator on localhost:8085 ..."
gcloud beta emulators pubsub start --host-port=localhost:8085