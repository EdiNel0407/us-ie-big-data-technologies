# assignment.py — pure-Python streaming windowing for Q6.3 / Q6.4
# Works with Pub/Sub emulator on Windows where Apache Beam wheels aren't available.

import os
import json
import time
import queue
import signal
import argparse
from datetime import datetime, timezone
from collections import defaultdict
from google.cloud import pubsub_v1

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "demo-project")
SUBSCRIPTION_ID = "usage-sub"

def parse_event_ts(s: str) -> int:
    # "YYYY-MM-DD HH:MM:SS" -> epoch seconds (UTC)
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())

def bucket_start(ts: int, width: int = 60) -> int:
    return (ts // width) * width

def fmt_hms(ts: int) -> str:
    return datetime.utcfromtimestamp(ts).strftime("%H:%M:%S")

def run(allowed_lateness: int = 0, window_sec: int = 60, flush_every: float = 1.0, max_runtime: int = 0):
    """
    allowed_lateness = 0   -> Q6.3 behavior (late events are dropped)
    allowed_lateness > 0   -> Q6.4 behavior (accept late events up to N seconds; print UPDATE)
    """
    emulator = os.getenv("PUBSUB_EMULATOR_HOST", "(not set)")
    print(f"Connecting to Pub/Sub emulator at {emulator}")
    print(f"Project={PROJECT}  Subscription={SUBSCRIPTION_ID}")
    print(f"Window={window_sec}s  Allowed lateness={allowed_lateness}s\n")

    subscriber = pubsub_v1.SubscriberClient()
    sub_path = subscriber.subscription_path(PROJECT, SUBSCRIPTION_ID)

    q: "queue.Queue[dict]" = queue.Queue()
    max_event_ts = 0  # tracks watermark basis (latest event time seen)

    # windows[(win_start)][(user_id,url)] = bytes_sum
    windows: dict[int, defaultdict] = defaultdict(lambda: defaultdict(int))
    # to know if we already printed a window result (so later we print UPDATE)
    printed: set[int] = set()

    stop = False
    def _sig(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    def callback(message: pubsub_v1.subscriber.message.Message):
        nonlocal max_event_ts
        try:
            data = json.loads(message.data.decode("utf-8"))
            evt_ts = parse_event_ts(data["timestamp"])
            max_event_ts = max(max_event_ts, evt_ts)
            data["_evt_ts"] = evt_ts
            q.put(data)
        except Exception as e:
            print("PARSE-ERROR:", e)
        finally:
            message.ack()

    streaming = subscriber.subscribe(sub_path, callback=callback)
    print(f"Listening... Press Ctrl+C to stop.\n")

    start_wall = time.time()
    try:
        while not stop:
            # drain quickly
            drained = False
            while True:
                try:
                    rec = q.get_nowait()
                    drained = True
                except queue.Empty:
                    break
                evt_ts = rec["_evt_ts"]
                win = bucket_start(evt_ts, window_sec)
                key = (rec["user_id"], rec["url"])
                # Late handling
                # Watermark approximation = latest event seen (max_event_ts)
                # A window [win, win+W) is 'on-time' if win+W <= watermark
                # If allowed_lateness == 0 and this is strictly older than current max window -> drop
                newest_win = bucket_start(max_event_ts, window_sec)
                if allowed_lateness == 0 and win < newest_win:
                    print(f"LATE-DROP [{fmt_hms(win)}–{fmt_hms(win+window_sec)}) "
                          f"user={key[0]} url={key[1]} bytes={rec['bytes']}")
                    continue
                # accept record
                windows[win][key] += int(rec["bytes"])

            # periodic flushing:
            # watermark := max_event_ts - allowed_lateness
            watermark = max(0, max_event_ts - allowed_lateness)
            # flush windows whose end <= watermark
            to_flush = [w for w in list(windows.keys()) if (w + window_sec) <= watermark]
            for w in sorted(to_flush):
                lines = []
                for (user_id, url), total in sorted(windows[w].items()):
                    prefix = "UPDATE" if w in printed else "RESULT"
                    lines.append(f"{prefix} [{fmt_hms(w)}–{fmt_hms(w+window_sec)}) "
                                 f"user={user_id} url={url} bytes={total}")
                if lines:
                    print("\n".join(lines))
                printed.add(w)
                # keep window in memory a bit longer if lateness>0 (to allow more updates)
                # but once it's older than watermark, we can drop
                del windows[w]

            if max_runtime and (time.time() - start_wall) > max_runtime:
                break

            if not drained:
                time.sleep(flush_every)

    finally:
        streaming.cancel()
        print("Stopped.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--allowed_lateness", type=int, default=0, help="seconds; 0 for Q6.3, >0 for Q6.4")
    ap.add_argument("--window", type=int, default=60, help="window size in seconds")
    ap.add_argument("--max_runtime", type=int, default=0, help="optional seconds to auto-stop (0 = run until Ctrl+C)")
    args = ap.parse_args()
    run(allowed_lateness=args.allowed_lateness, window_sec=args.window, max_runtime=args.max_runtime)
