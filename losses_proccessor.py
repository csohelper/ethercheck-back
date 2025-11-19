import asyncio
import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, TypedDict

# Удобная блокировка для конкурентного доступа (pip install filelock)
from filelock import FileLock

DATA_DIR = Path("data") / "losses"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_FIELDS = ["YYYY", "MM", "DD", "HH", "MM_min", "ROOM", "PACKETS", "REACHES", "LOSSES", "PERCENTS"]


# note: MM_min — название поля для минуты (т.к. MM уже занят месяц). В CSV будет записано как минутa.

def _parse_timestamp(ts: str) -> datetime:
    # ожидаемый формат "YYYY-MM-DD HH:MM"
    return datetime.strptime(ts, "%Y-%m-%d %H:%M")


def _row_from_values(dt: datetime, room: int, packets: int, reached: int, losses: int) -> Dict[str, str]:
    percent = 0.0
    if packets:
        percent = losses / packets * 100.0
    return {
        "YYYY": f"{dt.year:04d}",
        "MM": f"{dt.month:02d}",
        "DD": f"{dt.day:02d}",
        "HH": f"{dt.hour:02d}",
        "MM_min": f"{dt.minute:02d}",
        "ROOM": str(room),
        "PACKETS": str(packets),
        "REACHES": str(reached),
        "LOSSES": str(losses),
        "PERCENTS": f"{percent:.3f}",
    }


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh, delimiter=";")
        next(reader, None)  # Skip the header row
        for r in reader:
            # skip empty lines
            if not r:
                continue
            # Expect either 10 columns (our format) or older versions; be defensive
            if len(r) < 9:
                continue
            # if there are 9 columns (no room), adapt:
            if len(r) == 9:
                # YYYY;MM;DD;HH;MM;PACKETS;REACHES;LOSSES;PERCENTS  (total file)
                rows.append({
                    "YYYY": r[0],
                    "MM": r[1],
                    "DD": r[2],
                    "HH": r[3],
                    "MM_min": r[4],
                    "ROOM": "",  # empty for totals
                    "PACKETS": r[5],
                    "REACHES": r[6],
                    "LOSSES": r[7],
                    "PERCENTS": r[8],
                })
            else:
                # 10+ columns -> use first 10
                rows.append({
                    "YYYY": r[0],
                    "MM": r[1],
                    "DD": r[2],
                    "HH": r[3],
                    "MM_min": r[4],
                    "ROOM": r[5],
                    "PACKETS": r[6],
                    "REACHES": r[7],
                    "LOSSES": r[8],
                    "PERCENTS": r[9],
                })
    return rows


def _write_csv_rows_atomic(path: Path, rows: List[List[str]]):
    # rows: list of lists (already stringified)
    dirpath = path.parent
    dirpath.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=str(dirpath))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", newline="") as fh:
            writer = csv.writer(fh, delimiter=";")

            if path.name.endswith("_total.csv"):
                writer.writerow([
                    "YYYY", "MM", "DD", "HH", "MM", "PACKETS", "REACHES", "LOSSES", "PERCENTS"
                ])
            else:
                writer.writerow([
                    "YYYY", "MM", "DD", "HH", "MM", "ROOM",
                    "PACKETS", "REACHES", "LOSSES", "PERCENTS"
                ])

            for r in rows:
                writer.writerow(r)
        # atomic replace
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _rows_to_sorted_matrix(rows: List[Dict[str, str]]) -> List[List[str]]:
    # convert dict rows to list-of-lists and sort by datetime and room (room numeric if possible)
    def keyfn(r: Dict[str, str]):
        dt = datetime(
            int(r["YYYY"]), int(r["MM"]), int(r["DD"]), int(r["HH"]), int(r["MM_min"])
        )
        room_sort = int(r["ROOM"]) if r["ROOM"] != "" else -1
        return dt, room_sort

    rows_sorted = sorted(rows, key=keyfn)
    matrix = []
    for r in rows_sorted:
        matrix.append([
            r["YYYY"],
            r["MM"],
            r["DD"],
            r["HH"],
            r["MM_min"],
            r["ROOM"],
            r["PACKETS"],
            r["REACHES"],
            r["LOSSES"],
            r["PERCENTS"],
        ])
    return matrix


def _hour_key(dt: datetime) -> str:
    # returns "YYYY-MM-DD_HH"
    return dt.strftime("%Y-%m-%d_%H")


def _hour_file_paths_for_dt_hour(dt_hour: datetime) -> Tuple[Path, Path, Path]:
    """
    Returns (per_hour_path, total_path, lock_path)
    """
    key = _hour_key(dt_hour)
    per_hour = DATA_DIR / f"losses_{key}.csv"
    total = DATA_DIR / f"losses_{key}_total.csv"
    lock = DATA_DIR / f"losses_{key}.lock"
    return per_hour, total, lock


class AggVal(TypedDict):
    packets: int
    reaches: int
    losses: int
    dt: datetime


def _aggregate_totals_from_rows(rows: List[Dict[str, str]]) -> List[List[str]]:
    agg: Dict[str, AggVal] = {}
    for r in rows:
        ts = f"{r['YYYY']}-{r['MM']}-{r['DD']} {r['HH']}:{r['MM_min']}"
        packets = int(r["PACKETS"])
        reaches = int(r["REACHES"])
        losses = int(r["LOSSES"])

        if ts not in agg:
            agg[ts] = AggVal(
                packets=0,
                reaches=0,
                losses=0,
                dt=_parse_timestamp(ts)
            )

        agg[ts]["packets"] += packets
        agg[ts]["reaches"] += reaches
        agg[ts]["losses"] += losses

    items = sorted(agg.items(), key=lambda kv: kv[1]["dt"])
    out = []
    for ts, vals in items:
        dt = vals["dt"]
        pk = vals["packets"]
        rc = vals["reaches"]
        lo = vals["losses"]
        pct = (lo / pk * 100.0) if pk else 0.0
        out.append([
            f"{dt.year:04d}",
            f"{dt.month:02d}",
            f"{dt.day:02d}",
            f"{dt.hour:02d}",
            f"{dt.minute:02d}",
            str(pk),
            str(rc),
            str(lo),
            f"{pct:.3f}",
        ])
    return out


def _upsert_per_hour_rows(
        existing: List[Dict[str, str]], additions: List[Dict[str, str]], room: int
) -> List[Dict[str, str]]:
    # existing rows: list of dicts
    # additions: rows to add/replace (all for current room)
    # criteria for replacement: same timestamp (YYYY-MM-DD HH:MM) and same ROOM
    # create mapping (ts,room) -> index
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


def _process_losses_sync(room: int, file: Path):
    # 1) read input json
    with file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    # data: dict[str timestamp -> dict with packets/reached/losses]

    # group incoming records by their hour (dt.replace(minute=0))
    by_hour: Dict[datetime, List[Tuple[datetime, int, int, int]]] = {}
    for ts_str, vals in data.items():
        dt = _parse_timestamp(ts_str)
        dt_hour = dt.replace(minute=0, second=0, microsecond=0)
        packets = int(vals.get("packets", 0))
        reached = int(vals.get("reached", 0))
        losses = int(vals.get("losses", 0))
        by_hour.setdefault(dt_hour, []).append((dt, packets, reached, losses))

    # For each hour touched, update per-hour and total files
    for dt_hour, entries in by_hour.items():
        per_hour_path, total_path, lock_path = _hour_file_paths_for_dt_hour(dt_hour)
        # use file lock to prevent concurrent modifications
        lock = FileLock(str(lock_path))
        with lock:
            # read existing per-hour rows
            existing_rows = _read_csv_rows(per_hour_path)

            # build additions rows (for this room)
            additions = []
            for dt, packets, reached, losses in entries:
                additions.append(_row_from_values(dt, room, packets, reached, losses))

            # upsert
            merged = _upsert_per_hour_rows(existing_rows, additions, room)

            # sort and write per-hour file
            matrix = _rows_to_sorted_matrix(merged)
            _write_csv_rows_atomic(per_hour_path, matrix)

            # recompute totals from merged per-hour rows
            totals_matrix = _aggregate_totals_from_rows(merged)
            _write_csv_rows_atomic(total_path, totals_matrix)

    # optionally delete the input file
    try:
        file.unlink()
    except Exception:
        pass


async def process_losses(room: int, file: Path) -> None:
    """
    Асинхронная оболочка, которая выполнит синхронную работу в отдельном потоке.
    После успешной обработки входной json-файл удаляется.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _process_losses_sync, room, file)
