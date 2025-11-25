import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from quart import Blueprint, render_template, request, jsonify
import logging

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


def optimize_stepped_data(data: list[dict]) -> list[dict]:
    """
    Оптимизация для stepped графиков.
    Удаляет промежуточные точки в длинных сериях одинаковых значений,
    но оставляет первую и последнюю точку каждой серии для правильного отображения временных интервалов.
    """
    if len(data) <= 2:
        return data

    optimized = []
    i = 0

    while i < len(data):
        current_y = data[i]['y']

        # Добавляем первую точку серии
        optimized.append(data[i])

        # Ищем где заканчивается серия одинаковых значений
        j = i + 1
        while j < len(data) and data[j]['y'] == current_y:
            j += 1

        # Если серия длиннее 1 точки, добавляем последнюю точку серии
        if j - i > 1:  # <- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: проверяем длину серии
            optimized.append(data[j - 1])

        # Переходим к следующей серии
        i = j

    # Всегда добавляем последнюю точку, если её ещё нет
    if not optimized or optimized[-1]['x'] != data[-1]['x']:
        optimized.append(data[-1])

    return optimized


async def get_all_rooms() -> list[str]:
    global _rooms_cache, _cache_time
    now = datetime.now()
    if _rooms_cache is None or (now - _cache_time).seconds > 30:
        rooms = set()
        for p in HOURS_DIR.glob("losses_*.csv"):
            if p.name.endswith("_total.csv"):
                continue
            try:
                df = pd.read_csv(p, delimiter=";", usecols=["ROOM"])
                rooms.update(df["ROOM"].astype(str).dropna().unique())
            except Exception as e:
                logging.error(e)
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

    selected_rooms = rooms_param.split(",")
    if "total" in rooms_param:
        selected_rooms = await get_all_rooms()

    # Включительно оба конца: от start:00 до end:59
    current = start_dt.replace(minute=0, second=0)
    end_inclusive = end_dt.replace(hour=end_dt.hour, minute=59, second=59)

    timeline = pd.date_range(current, end_inclusive, freq="min")

    # По комнатам
    room_data = {room: {} for room in selected_rooms}
    hour = current.replace(minute=0)
    while hour <= end_dt:
        key = hour.strftime("%Y-%m-%d_%H")
        path = HOURS_DIR / f"losses_{key}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path, delimiter=";", header=0)
                df.columns = ["YYYY", "MM", "DD", "HH", "MM_min", "ROOM", "PACKETS", "REACHES", "LOSSES",
                              "PERCENTS"]
                df = df[df['ROOM'].isin(selected_rooms)]
                df['dt'] = pd.to_datetime(
                    df['YYYY'].astype(str) + '-' +
                    df['MM'].astype(str).str.zfill(2) + '-' +
                    df['DD'].astype(str).str.zfill(2) + ' ' +
                    df['HH'].astype(str).str.zfill(2) + ':' +
                    df['MM_min'].astype(str).str.zfill(2)
                )
                df = df[(df['dt'] >= current) & (df['dt'] <= end_inclusive)]
                for _, r in df.iterrows():
                    room = r['ROOM']
                    dt = r['dt']
                    room_data[room][dt] = float(r['PERCENTS'])
            except Exception as e:
                logging.error(e)
        hour += timedelta(hours=1)

    colors = ["#ff5555", "#50fa7b", "#ffb86c", "#8be9fd", "#ff79c6", "#bd93f9", "#f1fa8c", "#ff6e96"]
    datasets = []
    for i, room in enumerate(sorted(selected_rooms)):
        data = [{"x": dt.isoformat(), "y": room_data[room].get(dt, 0.0)} for dt in timeline]

        # Оптимизация для stepped графиков
        data = optimize_stepped_data(data)

        datasets.append({
            "label": f"Процент потерь (комната {room})",
            "data": data,
            "borderColor": colors[i % len(colors)],
            "backgroundColor": colors[i % len(colors)] + "50",
            "fill": True,
            "stepped": "before"
        })

    return jsonify({"datasets": datasets})
