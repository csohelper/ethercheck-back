import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
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

# Равномерно распределённых по цветовому кругу с шагом 5.625° (360° / 64).
colors = [
    "#e6194b", "#e9273f", "#ec3434", "#ee4129", "#f04e1e", "#f25a14", "#f3670b", "#f47404",
    "#f88100", "#fb8d00", "#fd9a00", "#ffa629", "#ffb34d", "#ffbf70", "#ffcc94", "#ffd9b8",
    "#fee08b", "#fee7a2", "#ffedb8", "#fff4ce", "#e6f598", "#d9f08b", "#cceb7f", "#bfe573",
    "#b3e067", "#a6da5c", "#99d451", "#8cce46", "#80c83c", "#74c234", "#68bb2c", "#5cb524",
    "#50af1d", "#44a916", "#3aa310", "#2f9d0a", "#259700", "#1e9100", "#188b00", "#128500",
    "#0c7f00", "#067900", "#007304", "#006d0f", "#00671a", "#006125", "#005b30", "#00553b",
    "#004f46", "#004951", "#00435c", "#003d67", "#003772", "#00317d", "#002b88", "#002593",
    "#001f9e", "#0019a9", "#0013b4", "#000dbf", "#0000ca", "#0d00d4", "#1a00df", "#2600ea",
    "#3300f5", "#4000ff", "#4c0dff", "#591aff", "#661aff", "#7326ff", "#8033ff", "#8d40ff",
    "#9a4dff", "#a65aff", "#b367ff", "#bf74ff", "#cc81ff", "#d98eff", "#e59bff", "#f2a8ff",
    "#ffa8f2", "#ff9be6", "#ff8eda", "#ff80ce", "#ff73c2", "#ff66b6", "#ff59aa", "#ff4d9e",
    "#ff4092", "#ff3486", "#ff287a", "#ff1c6e", "#ff1062", "#ff0456", "#f4004a", "#e6194b"
]


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
        start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M")
    except Exception as e:
        return jsonify({"error": "Формат: YYYY-MM-DD HH-mm", "message": str(e)}), 400

    selected_rooms = rooms_param.split(",")
    if "total" in rooms_param:
        selected_rooms = await get_all_rooms()

    # Включительно оба конца: от start:00 до end:59

    timeline = pd.date_range(start_dt, end_dt, freq="min")

    # По комнатам
    room_data = {room: {} for room in selected_rooms}
    hour = start_dt.replace(minute=0)
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
                df = df[(df['dt'] >= start_dt) & (df['dt'] <= end_dt)]
                for _, r in df.iterrows():
                    room = r['ROOM']
                    dt = r['dt']
                    room_data[room][dt] = float(r['PERCENTS'])
            except Exception as e:
                logging.error(e)
        hour += timedelta(hours=1)

    datasets = []
    for room in sorted(selected_rooms):
        room_str = str(room)
        hash_object = hashlib.sha256(room_str.encode())
        hash_int = int(hash_object.hexdigest(), 16)

        color_index = hash_int % len(colors)
        color = colors[color_index]

        data = [{"x": dt.isoformat(), "y": room_data[room].get(dt, 0.0)} for dt in timeline]
        data = optimize_stepped_data(data)

        datasets.append({
            "label": str(room),
            "data": data,
            "borderColor": color,
            "backgroundColor": color + "50",
            "fill": True
        })

    return jsonify({"datasets": datasets})
