import asyncio
import csv
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import filelock
from filelock import FileLock

DATA_DIR = Path("data") / "losses"
HOURS_DIR = DATA_DIR / "hours"
HOURS_DIR.mkdir(parents=True, exist_ok=True)

CSV_FIELDS = ["YYYY", "MM", "DD", "HH", "MM_min", "ROOM", "PACKETS", "REACHES", "LOSSES", "PERCENTS"]


def parse_timestamp(ts: str) -> datetime:
    """
    Parses a timestamp string into a datetime object.
    :param ts: Timestamp string in format "YYYY-MM-DD HH:MM".
    :return: Parsed datetime object.
    """
    return datetime.strptime(ts, "%Y-%m-%d %H:%M")


def create_row_from_values(dt: datetime, room: str, packets: int, reached: int, losses: int) -> Dict[str, str]:
    """
    Creates a row dictionary from given values, calculating percentage.
    :param dt: Datetime object.
    :param room: Room number.
    :param packets: Number of packets.
    :param reached: Number of reached packets.
    :param losses: Number of losses.
    :return: Dictionary representing the row.
    """
    percent = 0.0
    if packets:
        percent = losses / packets * 100.0
    return {
        "YYYY": f"{dt.year:04d}",
        "MM": f"{dt.month:02d}",
        "DD": f"{dt.day:02d}",
        "HH": f"{dt.hour:02d}",
        "MM_min": f"{dt.minute:02d}",
        "ROOM": room,
        "PACKETS": str(packets),
        "REACHES": str(reached),
        "LOSSES": str(losses),
        "PERCENTS": f"{percent:.3f}",
    }


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """
    Reads rows from a CSV file into a list of dictionaries.
    :param path: Path to the CSV file.
    :return: List of row dictionaries.
    """
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh, delimiter=";")
        next(reader, None)  # Skip the header row
        for r in reader:
            if not r:
                continue
            if len(r) < 9:
                continue
            if len(r) == 9:
                rows.append({
                    "YYYY": r[0], "MM": r[1], "DD": r[2], "HH": r[3], "MM_min": r[4],
                    "ROOM": "", "PACKETS": r[5], "REACHES": r[6], "LOSSES": r[7], "PERCENTS": r[8],
                })
            else:
                rows.append({
                    "YYYY": r[0], "MM": r[1], "DD": r[2], "HH": r[3], "MM_min": r[4],
                    "ROOM": r[5], "PACKETS": r[6], "REACHES": r[7], "LOSSES": r[8], "PERCENTS": r[9],
                })
    return rows


def write_csv_rows_atomic(path: Path, rows: List[List[str]]) -> None:
    """
    Writes rows to a CSV file atomically using a temporary file.
    :param path: Path to write the CSV.
    :param rows: List of lists representing rows.
    :return: None
    """
    dirpath = path.parent
    dirpath.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=str(dirpath))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", newline="") as fh:
            writer = csv.writer(fh, delimiter=";")
            writer.writerow(CSV_FIELDS)
            for r in rows:
                writer.writerow(r)
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def rows_to_sorted_matrix(rows: List[Dict[str, str]]) -> List[List[str]]:
    """
    Converts row dictionaries to a sorted list of lists.
    :param rows: List of row dictionaries.
    :return: Sorted list of lists.
    """

    def keyfn(r: Dict[str, str]):
        dt = datetime(int(r["YYYY"]), int(r["MM"]), int(r["DD"]), int(r["HH"]), int(r["MM_min"]))
        room_sort = r["ROOM"] if r["ROOM"] != "" else ""
        return dt, room_sort

    rows_sorted = sorted(rows, key=keyfn)
    matrix = []
    for r in rows_sorted:
        matrix.append([r["YYYY"], r["MM"], r["DD"], r["HH"], r["MM_min"], r["ROOM"],
                       r["PACKETS"], r["REACHES"], r["LOSSES"], r["PERCENTS"]])
    return matrix


def get_hour_key(dt: datetime) -> str:
    """
    Generates hour key from datetime.
    :param dt: Datetime object.
    :return: Hour key string "YYYY-MM-DD_HH".
    """
    return dt.strftime("%Y-%m-%d_%H")


def get_hour_file_paths(dt_hour: datetime) -> Tuple[Path, Path]:
    """
    Returns paths for hour files and lock.
    :param dt_hour: Datetime for the hour.
    :return: Tuple (per_hour_path, lock_path).
    """
    key = get_hour_key(dt_hour)
    per_hour = HOURS_DIR / f"losses_{key}.csv"
    lock = HOURS_DIR / f"losses_{key}.lock"
    return per_hour, lock


def upsert_per_hour_rows(
        existing: List[Dict[str, str]], additions: List[Dict[str, str]], room: str
) -> List[Dict[str, str]]:
    """
    Upserts additions into existing rows for a specific room.
    :param existing: Existing row dictionaries.
    :param additions: New row dictionaries to add or update.
    :param room: Room number.
    :return: Updated list of row dictionaries.
    """
    index = {}
    for i, r in enumerate(existing):
        ts = f"{r['YYYY']}-{r['MM']}-{r['DD']} {r['HH']}:{r['MM_min']}"
        index[(ts, r["ROOM"])] = i
    for add in additions:
        ts_add = f"{add['YYYY']}-{add['MM']}-{add['DD']} {add['HH']}:{add['MM_min']}"
        key = (ts_add, str(room))
        if key in index:
            existing[index[key]] = add
        else:
            existing.append(add)
    return existing


def normalize_counts(packets_raw, reached_raw, losses_raw) -> Tuple[int, int, int]:
    """
    Normalizes and computes packet counts, handling anomalies.
    :param packets_raw: Raw packets value.
    :param reached_raw: Raw reached value.
    :param losses_raw: Raw losses value (optional).
    :return: Tuple (packets, reached, losses).
    """
    try:
        packets = int(packets_raw)
    except Exception:
        packets = 0
    try:
        reached = int(reached_raw)
    except Exception:
        reached = 0
    if losses_raw is not None:
        try:
            losses = int(losses_raw)
        except Exception:
            losses = max(0, packets - reached)
    else:
        losses = packets - reached
    if losses < 0:
        logging.warning("Negative losses: packets=%r, reached=%r, losses=%r. Clamping.", packets, reached, losses)
        reached = min(reached, packets)
        losses = max(0, packets - reached)
    if reached < 0:
        logging.warning("Negative reached: %r -> 0", reached)
        reached = 0
    return packets, reached, losses


def group_data_by_hour(data: Dict[str, Dict[str, int]]) -> Dict[datetime, List[Tuple[datetime, int, int, int]]]:
    """
    Groups input data by hour.
    :param data: Input data dictionary.
    :return: Dictionary of hour to list of (dt, packets, reached, losses).
    """
    by_hour: Dict[datetime, List[Tuple[datetime, int, int, int]]] = {}
    for ts_str, vals in data.items():
        dt = parse_timestamp(ts_str)
        dt_hour = dt.replace(minute=0, second=0, microsecond=0)
        packets_raw = vals.get("packets", 0)
        reached_raw = vals.get("reached", 0)
        losses_raw = vals.get("losses", None)
        packets, reached, losses = normalize_counts(packets_raw, reached_raw, losses_raw)
        by_hour.setdefault(dt_hour, []).append((dt, packets, reached, losses))
    return by_hour


def update_hourly_files(dt_hour: datetime, entries: List[Tuple[datetime, int, int, int]], room: str) -> None:
    """
    Updates hourly files for a given hour and entries.
    :param dt_hour: Hour datetime.
    :param entries: List of (dt, packets, reached, losses).
    :param room: Room number.
    :return: None
    """
    per_hour_path, lock_path = get_hour_file_paths(dt_hour)
    lock = FileLock(str(lock_path), timeout=10)
    logging.info(f"Acquiring lock for {lock_path} (hour: {dt_hour})")
    try:
        with lock:
            logging.info(f"Acquired lock for {lock_path}")
            existing_rows = read_csv_rows(per_hour_path)
            additions = [create_row_from_values(dt, room, packets, reached, losses) for dt, packets, reached, losses in
                         entries]
            merged = upsert_per_hour_rows(existing_rows, additions, room)
            matrix = rows_to_sorted_matrix(merged)
            write_csv_rows_atomic(per_hour_path, matrix)
        logging.info(f"Released lock for {lock_path}")
        try:
            lock_path.unlink()
        except Exception as e:
            print(e)
    except filelock.Timeout:
        logging.error(f"Timeout acquiring lock for {lock_path}")
        raise


def process_losses_sync(room: str, file: Path) -> None:
    """
    Synchronous processing of losses file, updating hourly and daily files.
    :param room: Room number.
    :param file: Path to input JSON file.
    :return: None
    """
    if not os.path.exists(file) or os.path.getsize(file) == 0:
        print(f"File {file} is empty or missing")
        return None
    with file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    by_hour = group_data_by_hour(data)
    for dt_hour, entries in by_hour.items():
        update_hourly_files(dt_hour, entries, room)
    try:
        file.unlink()
    except Exception as e:
        print(e)


async def process_losses(room: str, file: Path) -> None:
    """
    Asynchronous wrapper to process losses in a separate thread.
    :param room: Room number.
    :param file: Path to input JSON file.
    :return: None
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, process_losses_sync, room, file)
