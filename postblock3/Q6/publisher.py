import argparse, json, os, uuid, random
from datetime import datetime, timedelta, timezone
from google.cloud import pubsub_v1

PROJECT_ID = "demo-project"
TOPIC_ID = "usage-topic"

def make_record():
    users = [
        ("01f4f1c2", "Bob the Builder"),
        ("ab12cd34", "Alice Wonder"),
        ("9f00ee77", "Charlie Brown"),
    ]
    urls = [
        "www.google.com/colab",
        "news.example.com/story/42",
        "stream.example.net/live",
        "docs.example.org/help",
    ]
    user_id, fullname = random.choice(users)
    url = random.choice(urls)
    ts = datetime.now(timezone.utc) - timedelta(seconds=random.randint(0, 120))
    bytes_used = random.randint(100, 50000)
    return {
        "user_id": user_id,
        "fullname": fullname,
        "url": url,
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "bytes": bytes_used,
    }

def main(n):
    if not os.environ.get("PUBSUB_EMULATOR_HOST"):
        raise SystemExit("PUBSUB_EMULATOR_HOST not set. Start emulator and export it first.")
    pub = pubsub_v1.PublisherClient()
    topic_path = pub.topic_path(PROJECT_ID, TOPIC_ID)
    print(f"Publishing {n} messages to {topic_path} (emulator={os.environ['PUBSUB_EMULATOR_HOST']}) ...")
    for _ in range(n):
        rec = make_record()
        future = pub.publish(topic_path, json.dumps(rec).encode("utf-8"), event_id=str(uuid.uuid4()))
        msg_id = future.result()
        print(f"Published message_id={msg_id}  payload={rec}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5)
    args = ap.parse_args()
    main(args.count)
