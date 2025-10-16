# assignment.py — Q6.3 (Fixed 60s window aggregation)
import os, json, logging
from datetime import datetime, timezone
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

# Pub/Sub emulator project + subscription
PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "demo-project")
SUB_PATH = f"projects/{PROJECT}/subscriptions/usage-sub"

def to_event_ts(s: str) -> float:
    """Convert the event 'timestamp' string to epoch seconds (UTC)."""
    # Input format from your publisher: 'YYYY-MM-DD HH:MM:SS'
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()

class FormatWithWindow(beam.DoFn):
    """Attach window boundaries to the aggregated output for readability."""
    def process(self, kv, window=beam.DoFn.WindowParam):
        (user_id, url), total_bytes = kv
        ws = datetime.utcfromtimestamp(int(window.start)).strftime("%H:%M:%S")
        we = datetime.utcfromtimestamp(int(window.end)).strftime("%H:%M:%S")
        yield f"[{ws}–{we}) user={user_id} url={url} bytes={total_bytes}"

def run():
    # Streaming pipeline with DirectRunner
    opts = PipelineOptions(streaming=True, save_main_session=True)
    p = beam.Pipeline(options=opts)

    (
        p
        # 1) Read messages from Pub/Sub emulator subscription (bytes)
        | "Read" >> beam.io.ReadFromPubSub(subscription=SUB_PATH)

        # 2) Decode & parse JSON
        | "Decode" >> beam.Map(lambda b: b.decode("utf-8"))
        | "ParseJSON" >> beam.Map(json.loads)

        # 3) Use the event timestamp carried in the payload (event-time)
        | "Stamp" >> beam.Map(lambda r: beam.window.TimestampedValue(r, to_event_ts(r["timestamp"])))

        # 4) Key by (user_id, url) and take the numeric bytes
        | "KV" >> beam.Map(lambda r: ((r["user_id"], r["url"]), int(r["bytes"])))

        # 5) Fixed (tumbling) windows of 60 seconds
        | "Win60" >> beam.WindowInto(beam.window.FixedWindows(60))

        # 6) Aggregate: sum the bytes in each (user_id, url, window)
        | "Sum" >> beam.CombinePerKey(sum)

        # 7) Format output with window bounds and print to console
        | "Fmt" >> beam.ParDo(FormatWithWindow())
        | "Print" >> beam.Map(print)
    )

    result = p.run()
    result.wait_until_finish()

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
