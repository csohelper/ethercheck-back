import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
from pydantic import BaseModel, Field, field_validator
from quart import Blueprint, jsonify, render_template, send_from_directory
from quart_schema import validate_querystring, validate_response, hide

BASE_DIR = Path(__file__).resolve().parent

graph_bp = Blueprint(
    'graph',
    __name__,
    # template_folder=BASE_DIR / "templates",
    # static_folder=BASE_DIR / "static"
)

HOURS_DIR = Path("data/losses/hours")

# Кэшируем список комнат (обновляется раз в 30 сек)
_rooms_cache = None
_cache_time: datetime | None = None

# Равномерно распределённых по цветовому кругу с шагом 10° (360° / 36).
colors = [
    "#e6194b", "#f15838", "#fb8b2d", "#ffb41f", "#ffdc19", "#e6f50e", "#b8e986", "#86d958",
    "#57c42b", "#2cae0c", "#1a9600", "#008c1a", "#00805c", "#00749f", "#2f67d4", "#5e5af5",
    "#8a4dff", "#b33cff", "#d41aff", "#f500c8", "#ff0091", "#ff3366", "#ff6640", "#ff991a",
    "#ffcc00", "#e6ee00", "#b3f00c", "#80f22d", "#4df44d", "#1af66e", "#00e68a", "#00d4b3",
    "#00c2dc", "#00aaff", "#3377ff", "#5f4dff"
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
        if j - i > 1:
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


class RoomsResponse(BaseModel):
    rooms: List[str] = Field(
        ...,
        title="Rooms",
        description="Список всех доступных комнат"
    )

    model_config = {
        'json_schema_extra': {
            'example': {
                'rooms': ["101", "102", "536"]
            }
        }
    }


# Видимый маршрут с описанием (docstring + @validate_response)
@graph_bp.route("/api/rooms/", methods=["GET"])
@validate_response(RoomsResponse)
async def api_rooms():
    """
    Возвращает список всех доступных комнат.

    Этот эндпоинт возвращает отсортированный список уникальных идентификаторов комнат,
    извлеченных из CSV-файлов в директории HOURS_DIR.
    Кэширует результат на 30 секунд для оптимизации производительности.
    """
    rooms = await get_all_rooms()
    return RoomsResponse(rooms=rooms)


class DataPoint(BaseModel):
    x: str = Field(..., title="X")
    y: float = Field(..., title="Y")

    model_config = {
        'json_schema_extra': {
            'example': {
                'x': '2025-01-01T10:00:00',
                'y': 12.5
            }
        }
    }


class Dataset(BaseModel):
    label: str = Field(..., title="Label")
    data: List[DataPoint] = Field(..., title="Data")
    borderColor: str = Field(..., title="BorderColor")
    backgroundColor: str = Field(..., title="BackgroundColor")
    fill: bool = Field(..., title="Fill")

    model_config = {
        'json_schema_extra': {
            'example': {
                'label': '123',
                'data': [
                    {"x": "2025-11-02T15:00:00", "y": 0},
                    {"x": "2025-11-25T19:55:00", "y": 0},
                    {"x": "2025-11-25T19:56:00", "y": 5.556},
                    {"x": "2025-11-25T19:57:00", "y": 0},
                    {"x": "2025-11-25T19:58:00", "y": 5},
                    {"x": "2025-11-25T19:59:00", "y": 0}
                ],
                'borderColor': '#ff0000',
                'backgroundColor': '#ff000050',
                'fill': True
            }
        }
    }


class ApiResponse(BaseModel):
    datasets: List[Dataset] = Field(..., title="Datasets")

    model_config = {
        'json_schema_extra': {
            'example': {
                'datasets': [
                    {
                        'label': '123',
                        'data': [
                            {"x": "2025-11-02T15:00:00", "y": 0},
                            {"x": "2025-11-25T19:55:00", "y": 0},
                            {"x": "2025-11-25T19:56:00", "y": 5.556},
                            {"x": "2025-11-25T19:57:00", "y": 0},
                            {"x": "2025-11-25T19:58:00", "y": 5},
                            {"x": "2025-11-25T19:59:00", "y": 0}
                        ],
                        'borderColor': '#ff0000',
                        'backgroundColor': '#ff000050',
                        'fill': True
                    }
                ]
            }
        }
    }


class ApiFilters(BaseModel):
    start: str = Field(
        ...,
        description="Начало периода (формат: YYYY-MM-DD HH:MM)",
        json_schema_extra={"example": "2025-11-19 15:00"}
    )
    end: str = Field(
        ...,
        description="Конец периода (формат: YYYY-MM-DD HH:MM)",
        json_schema_extra={"example": "2025-11-19 23:59"}
    )
    rooms: Optional[str] = Field(
        None,
        description="Список комнат через запятую. Или 'total' для всех по отдельности / 'summary'",
        json_schema_extra={"example": "101,102"}
    )

    @field_validator("rooms", mode="before")
    @classmethod
    def strip_rooms(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return ",".join(part.strip() for part in v.split(",") if part.strip())


@graph_bp.get("/api/graph/")
@validate_querystring(ApiFilters)
@validate_response(ApiResponse)
async def get_graph_points(query_args: ApiFilters):
    """
    Возвращает набор точек для построение графика Losses(Time) для каждой комнаты (или среднее по всем)
    """
    start_str = query_args.start
    end_str = query_args.end
    rooms_param = query_args.rooms or ""

    try:
        start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M")
    except Exception as e:
        return jsonify({"error": "Формат: YYYY-MM-DD HH:MM", "message": str(e)}), 400

    if end_dt < start_dt:
        return jsonify({"error": "end < start"}), 400

    # Список комнат
    selected_rooms = [r.strip() for r in rooms_param.split(",") if r.strip()]
    if "total" in selected_rooms:
        selected_rooms = await get_all_rooms()

    # Основная шкала времени
    timeline = pd.date_range(start_dt, end_dt, freq="T")

    room_data: Dict[str, Dict[datetime, float]] = {room: {} for room in selected_rooms}

    hour = start_dt.replace(minute=0)
    while hour <= end_dt:
        key = hour.strftime("%Y-%m-%d_%H")
        path = HOURS_DIR / f"losses_{key}.csv"

        if path.exists():
            try:
                df = pd.read_csv(path, delimiter=";", header=0)
                df.columns = ["YYYY", "MM", "DD", "HH", "MM_min", "ROOM", "PACKETS", "REACHES", "LOSSES", "PERCENTS"]
                df = df[df["ROOM"].isin(selected_rooms)]

                df["dt"] = pd.to_datetime(
                    df["YYYY"].astype(str) + "-" +
                    df["MM"].astype(str).str.zfill(2) + "-" +
                    df["DD"].astype(str).str.zfill(2) + " " +
                    df["HH"].astype(str).str.zfill(2) + ":" +
                    df["MM_min"].astype(str).str.zfill(2)
                )

                df = df[(df["dt"] >= start_dt) & (df["dt"] <= end_dt)]

                for _, r in df.iterrows():
                    room_data[r["ROOM"]][r["dt"].to_pydatetime()] = float(r["PERCENTS"])

            except Exception as e:
                logging.error(e)

        hour += timedelta(hours=1)

    # Формируем response согласно схеме ApiResponse
    datasets: List[Dataset] = []

    for room in sorted(selected_rooms):
        hash_val = int(hashlib.sha256(str(room).encode()).hexdigest(), 16)
        color = colors[hash_val % len(colors)]

        datapoints_raw: List[dict] = []
        for t in timeline:
            dt = t.to_pydatetime()
            y = room_data[room].get(dt, 0.0)
            datapoints_raw.append({"x": dt.isoformat(), "y": float(y)})

        optimized_datapoints = optimize_stepped_data(datapoints_raw)
        datapoints = [DataPoint(**dp) for dp in optimized_datapoints]

        datasets.append(
            Dataset(
                label=str(room),
                data=datapoints,
                borderColor=color,
                backgroundColor=color + "50",
                fill=True
            )
        )

    return ApiResponse(datasets=datasets)
