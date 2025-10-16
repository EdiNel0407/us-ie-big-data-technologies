# test_late_publisher.py
import json, os, uuid, random, time
from datetime import datetime, timedelta, timezone
from google.cloud import pubsub_v1

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "demo-project")
TOPIC = "usage-topic"

def mk(user, url, event_ts):
    return {
        "user_id": user,
        "fullname": {"01f4f1c2":"Bob the Builder","ab12cd34":"Alice Wonder","9f00ee77":"Charlie Brown"}[user],
        "url": url,
        "timestamp": event_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "bytes": random.randint(10_000, 50_000),
    }

if __name__ == "__main__":
    emu = os.environ.get("PUBSUB_EMULATOR_HOST")
    if not emu: raise SystemExit("Set PUBSUB_EMULATOR_HOST=localhost:8085")
    pub = pubsub_v1.PublisherClient()
    topic = pub.topic_path(PROJECT, TOPIC)

    now = datetime.now(timezone.utc)
    ontime_ts = now - timedelta(seconds=10)
    late_ts   = now - timedelta(seconds=90)  # <-- LATE for a 60s window

    msgs = []
    msgs += [mk("01f4f1c2","www.google.com/colab", ontime_ts) for _ in range(3)]
    msgs += [mk("01f4f1c2","www.google.com/colab", late_ts)   for _ in range(3)]

    print(f"Publishing {len(msgs)} messages (3 on-time, 3 late) to emulator={emu}")
    for m in msgs:
        mid = pub.publish(topic, json.dumps(m).encode(), event_id=str(uuid.uuid4())).result()
        print("Published", mid, m)
