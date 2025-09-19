# student.py  —  run with:  python student.py --host 127.0.0.1 --keyspace wind_turbine_data ...
# Requires: cassandra-driver  (pip install cassandra-driver)

from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
from datetime import datetime
import argparse
import uuid

def connect(host: str, keyspace: str):
    cluster = Cluster([host])
    session = cluster.connect()
    # ensure the keyspace exists before use (safe if it already exists)
    session.execute(f"""
        CREATE KEYSPACE IF NOT EXISTS {keyspace}
        WITH replication = {{ 'class': 'SimpleStrategy', 'replication_factor': 1 }};
    """)
    session.set_keyspace(keyspace)

    # OPTIONAL: ensure table exists (schema used in the assignment)
    session.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            turbine_id   uuid,
            day          text,
            timestamp    timestamp,
            wind_speed   float,
            temperature  float,
            power_output float,
            status       text,
            PRIMARY KEY ((turbine_id, day), timestamp)
        ) WITH CLUSTERING ORDER BY (timestamp DESC);
    """)

    # OPTIONAL: index to count zeros without ALLOW FILTERING (safer for prod)
    session.execute("""
        CREATE INDEX IF NOT EXISTS sensor_readings_power_output_idx
        ON sensor_readings (power_output);
    """)
    return session

# ---------- Queries (parameterised) ----------

def get_last_reading(session, turbine_id: uuid.UUID, day: str):
    q = session.prepare("""
        SELECT timestamp, power_output, temperature, wind_speed, status
        FROM sensor_readings
        WHERE turbine_id = ? AND day = ?
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    return session.execute(q, (turbine_id, day)).one()

def get_two_readings(session, turbine_id: uuid.UUID, day: str):
    q = session.prepare("""
        SELECT timestamp, power_output, temperature, wind_speed, status
        FROM sensor_readings
        WHERE turbine_id = ? AND day = ?
        ORDER BY timestamp DESC
        LIMIT 2
    """)
    return list(session.execute(q, (turbine_id, day)))

def count_readings_between(session, turbine_id: uuid.UUID, day: str,
                           ts_from: datetime, ts_to: datetime) -> int:
    q = session.prepare("""
        SELECT count(*) AS c
        FROM sensor_readings
        WHERE turbine_id = ? AND day = ? AND timestamp >= ? AND timestamp <= ?
    """)
    row = session.execute(q, (turbine_id, day, ts_from, ts_to)).one()
    return int(row.c) if row and row.c is not None else 0

def turbine_health(session, turbine_id: uuid.UUID, day: str):
    # Average wind speed
    q_avg = session.prepare("""
        SELECT avg(wind_speed) AS avg_ws
        FROM sensor_readings
        WHERE turbine_id = ? AND day = ?
    """)
    avg_ws_row = session.execute(q_avg, (turbine_id, day)).one()
    avg_ws = float(avg_ws_row.avg_ws) if avg_ws_row and avg_ws_row.avg_ws is not None else 0.0

    # Count power_output == 0 (secondary index created in connect())
    q_zero = session.prepare("""
        SELECT count(*) AS zero_events
        FROM sensor_readings
        WHERE turbine_id = ? AND day = ? AND power_output = 0
    """)
    zero_row = session.execute(q_zero, (turbine_id, day)).one()
    zero_events = int(zero_row.zero_events) if zero_row and zero_row.zero_events is not None else 0

    is_healthy = (15.0 <= avg_ws <= 25.0) and (zero_events <= 5)
    return {
        "avg_wind_speed": avg_ws,
        "zero_power_events": zero_events,
        "healthy": is_healthy
    }

# ---------- CLI for quick testing ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1", help="Cassandra contact point (default 127.0.0.1)")
    p.add_argument("--keyspace", default="wind_turbine_data", help="Keyspace name")
    p.add_argument("--turbine_id", required=True, help="Turbine UUID (e.g., d6a82a95-1b92-4e0a-843c-def36840ff07)")
    p.add_argument("--day", required=True, help="Day partition (YYYY-MM-DD)")
    p.add_argument("--from_ts", help="From timestamp (ISO 8601, e.g. 2025-09-07T09:00:00Z)")
    p.add_argument("--to_ts", help="To timestamp   (ISO 8601, e.g. 2025-09-07T10:00:00Z)")
    args = p.parse_args()

    session = connect(args.host, args.keyspace)
    tid = uuid.UUID(args.turbine_id)

    print("Last reading:")
    print(get_last_reading(session, tid, args.day))

    print("\nTwo readings:")
    for r in get_two_readings(session, tid, args.day):
        print(r)

    if args.from_ts and args.to_ts:
        # naive ISO parse; add stricter parsing if needed
        ts_from = datetime.fromisoformat(args.from_ts.replace("Z", "+00:00"))
        ts_to   = datetime.fromisoformat(args.to_ts.replace("Z", "+00:00"))
        print("\nCount in window:", count_readings_between(session, tid, args.day, ts_from, ts_to))

    print("\nHealth:")
    print(turbine_health(session, tid, args.day))
