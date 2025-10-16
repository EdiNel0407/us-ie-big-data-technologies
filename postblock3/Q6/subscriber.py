# subscriber.py
import os
import json
import time
from typing import List
from google.cloud import pubsub_v1

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "demo-project")
SUBSCRIPTION_ID = "usage-sub"

def pull_n_messages(n: int = 5, timeout: float = 20.0) -> List[str]:
    """Pull up to n messages, print them, ack, then return their payloads."""
    emulator = os.getenv("PUBSUB_EMULATOR_HOST", "(not set)")
    print(f"Connecting to Pub/Sub emulator at {emulator}")
    subscriber = pubsub_v1.SubscriberClient()
    sub_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)
    print(f"Subscription: {sub_path}\n")

    received = []
    start = time.time()

    def callback(message: pubsub_v1.subscriber.message.Message) -> None:
        payload = message.data.decode("utf-8", errors="replace")
        print(f"Received message_id={message.message_id} data={payload}")
        message.ack()
        received.append(payload)
        if len(received) >= n:
            streaming_pull_future.cancel()

    streaming_pull_future = subscriber.subscribe(sub_path, callback=callback)
    print(f"Listening (will stop after {n} messages or {timeout}s)...\n")
    try:
        streaming_pull_future.result(timeout=timeout)
    except Exception:
        streaming_pull_future.cancel()

    print(f"\nDone. Received {len(received)} message(s).")
    return received

if __name__ == "__main__":
    pull_n_messages(n=5, timeout=20.0)
