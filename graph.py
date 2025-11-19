# graph.py
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from quart import Blueprint, render_template, request, jsonify

BASE_DIR = Path(__file__).resolve().parent

graph_bp = Blueprint(
    'graph',
    __name__,
    template_folder=BASE_DIR / "templates",
    static_folder=BASE_DIR / "static"
)

HOURS_DIR = Path("data/losses/hours")

# Кэшируем список комнат (обновляется раз в 30 сек)
_rooms_cache = None
_cache_time: datetime | None = None


async def get_all_rooms() -> list[int]:
    global _rooms_cache, _cache_time
    now = datetime.now()
    if _rooms_cache is None or (now - _cache_time).seconds > 30:
        rooms = set()
        for p in HOURS_DIR.glob("losses_*.csv"):
            if p.name.endswith("_total.csv"):
                continue
            try:
                df = pd.read_csv(p, delimiter=";", usecols=["ROOM"])
                rooms.update(df["ROOM"].astype(int).dropna().unique())
            except Exception as e:
                print(e)
                continue
        _rooms_cache = sorted(rooms)
        _cache_time = now
    return _rooms_cache


@graph_bp.route("/")
async def index():
    rooms = await get_all_rooms()
    return await render_template("index.html", rooms=rooms)


@graph_bp.route("/api/rooms")
async def api_rooms():
    return jsonify(await get_all_rooms())


@graph_bp.route("/api/data")
async def api_data():
    start_str = request.args.get("start")  # "2025-11-19 15"
    end_str = request.args.get("end")  # "2025-11-19 23"
    rooms_param = request.args.get("rooms", "")

    if not start_str or not end_str:
        return jsonify({"error": "start и end обязательны"}), 400

    try:
        start_dt = datetime.strptime(start_str, "%Y-%m-%d %H")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d %H")
    except Exception as e:
        return jsonify({"error": "Формат: YYYY-MM-DD HH", "message": str(e)}), 400

    selected_rooms = [int(x) for x in rooms_param.split(",") if x.isdigit()]
    aggregate = "total" in rooms_param

    # Включительно оба конца: от start:00 до end:59
    current = start_dt.replace(minute=0, second=0)
    end_inclusive = end_dt.replace(hour=end_dt.hour, minute=59, second=59)

    timeline = pd.date_range(current, end_inclusive, freq="min")

    if aggregate or not selected_rooms:
        # Суммарно по всем — только _total.csv
        losses_dict = {}
        hour = current.replace(minute=0)
        while hour <= end_dt:
            key = hour.strftime("%Y-%m-%d_%H")
            path = HOURS_DIR / f"losses_{key}_total.csv"
            if path.exists():
                try:
                    df = pd.read_csv(path, delimiter=";")
                    df['dt'] = pd.to_datetime(df['YYYY'] + '-' + df['MM'] + '-' + df['DD'] + ' ' +
                                              df['HH'] + ':' + df['MM_min'])
                    for _, row in df.iterrows():
                        if current <= row['dt'] <= end_inclusive:
                            losses_dict[row['dt']] = int(row['LOSSES'])
                except Exception as e:
                    print(e)
            hour += timedelta(hours=1)

        data = [{"x": dt.isoformat(), "y": losses_dict.get(dt, 0)} for dt in timeline]

        return jsonify({
            "datasets": [{
                "label": "Суммарно по всем комнатам",
                "data": data,
                "borderColor": "#ff5555",
                "backgroundColor": "#ff555550",
                "fill": True,
                "tension": 0.3
            }]
        })

    else:
        # По комнатам
        room_data = {room: {} for room in selected_rooms}
        hour = current.replace(minute=0)
        while hour <= end_dt:
            key = hour.strftime("%Y-%m-%d_%H")
            path = HOURS_DIR / f"losses_{key}.csv"
            if path.exists():
                try:
                    df = pd.read_csv(path, delimiter=";", dtype=str)
                    df = df[df['ROOM'].isin(map(str, selected_rooms))]
                    df['dt'] = pd.to_datetime(
                        df['YYYY'] + '-' + df['MM'] + '-' + df['DD'] + ' ' + df['HH'] + ':' + df['MM_min'])
                    df = df[(df['dt'] >= current) & (df['dt'] <= end_inclusive)]
                    for _, r in df.iterrows():
                        room = int(r['ROOM'])
                        dt = r['dt']
                        room_data[room][dt] = room_data[room].get(dt, 0) + int(r['LOSSES'])
                except Exception as e:
                    print(e)
            hour += timedelta(hours=1)

        colors = ["#ff5555", "#50fa7b", "#ffb86c", "#8be9fd", "#ff79c6", "#bd93f9", "#f1fa8c", "#ff6e96"]
        datasets = []
        for i, room in enumerate(sorted(selected_rooms)):
            data = [{"x": dt.isoformat(), "y": room_data[room].get(dt, 0)} for dt in timeline]
            datasets.append({
                "label": f"Комната {room}",
                "data": data,
                "borderColor": colors[i % len(colors)],
                "backgroundColor": colors[i % len(colors)] + "50",
                "fill": True,
                "tension": 0.3
            })

        return jsonify({"datasets": datasets})
