import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
from pydantic import BaseModel, Field, field_validator
from quart import Blueprint, jsonify
from quart_schema import validate_querystring, validate_response

BASE_DIR = Path(__file__).resolve().parent

graph_bp = Blueprint(
    'graph',
    __name__
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
            if not p.is_file():
                continue
            try:
                # Читаем ВСЕ колонки, но только нужные строки
                # header=0 — предполагаем, что заголовок есть
                df = pd.read_csv(p, delimiter=";", header=0, usecols=lambda x: x in {"ROOM"}, dtype=str)
                if "ROOM" in df.columns:
                    rooms.update(df["ROOM"].dropna().astype(str).str.strip().unique())
            except Exception as e:
                # Если usecols не сработал — попробуем без него
                try:
                    df = pd.read_csv(p, delimiter=";", header=0, nrows=1000, dtype=str)
                    # Попробуем найти колонку ROOM по позиции (6-я колонка)
                    if df.shape[1] >= 6:
                        rooms.update(df.iloc[:, 5].dropna().astype(str).str.strip().unique())
                except Exception as e2:
                    logging.error(f"Не удалось прочитать комнаты из {p}: {e2}")
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
@graph_bp.route("/api/rooms", methods=["GET"], strict_slashes=False)
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
        description="Список комнат через запятую. "
                    "`total` — все комнаты отдельными линиями, "
                    "`summary` — одна линия с общим % потерь по всем комнатам",
        json_schema_extra={"example": "101,102"}
    )
    excluded_rooms: Optional[str] = Field(
        # "ULK905v4,
        None,
        description="Список исключенных комнат через запятую. "
                    "Эти комнаты не будут обрабатываться при режимах 'total' и 'summary'.",
        json_schema_extra={"example": "103,104"}
    )

    @field_validator("rooms", mode="before")
    @classmethod
    def strip_rooms(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return ",".join(part.strip() for part in v.split(",") if part.strip())

    @field_validator("excluded_rooms", mode="before")
    @classmethod
    def strip_excluded_rooms(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return ",".join(part.strip() for part in v.split(",") if part.strip())


@graph_bp.get("/api/graph", strict_slashes=False)
@validate_querystring(ApiFilters)
@validate_response(ApiResponse)
async def get_graph_points(query_args: ApiFilters):
    """
    Возвращает данные для графика потерь по времени.
    rooms:
      - пусто или конкретные номера → только эти комнаты
      - "total"   → все комнаты отдельными линиями
      - "summary" → одна линия: общий % потерь по всем комнатам
    """
    start_str = query_args.start
    end_str = query_args.end
    rooms_param = query_args.rooms or ""
    excluded_param = query_args.excluded_rooms or ""

    # 1. Парсим даты
    try:
        start_dt, end_dt = _parse_datetime_range(start_str, end_str)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # 2. Определяем исключенные комнаты
    excluded = set(r.strip() for r in excluded_param.split(",") if r.strip())

    # 3. Определяем, какие комнаты нужны
    selected = [r.strip() for r in rooms_param.split(",") if r.strip()]

    if "summary" in selected:
        # Режим summary — игнорируем остальные значения
        df = _load_hourly_data_for_period(start_dt, end_dt, exclude_rooms=excluded)  # все комнаты кроме исключенных
        timeline = pd.date_range(start_dt, end_dt, freq="T")
        dataset = _build_summary_dataset(df, timeline)
        return ApiResponse(datasets=[dataset])

    # Режим total — все комнаты по отдельности кроме исключенных
    if "total" in selected:
        all_rooms = await get_all_rooms()
        selected = [room for room in all_rooms if room not in excluded]

    room_set = set(selected) if selected else None

    # 4. Загружаем данные
    df = _load_hourly_data_for_period(start_dt, end_dt, room_filter=room_set,
                                      exclude_rooms=excluded if "summary" in selected else None)

    if df.empty:
        # Нет данных — возвращаем пустые графики (чтобы фронт не падал)
        return ApiResponse(datasets=[])

    # 5. Формируем временную шкалу
    timeline = pd.date_range(start_dt, end_dt, freq="T")

    # 6. Строим ответ
    if "total" in (rooms_param or "") or selected:
        datasets = _build_room_datasets(df, timeline, list(room_set or []))
    else:
        # Запрос без rooms → по умолчанию summary (или можно оставить пустым)
        dataset = _build_summary_dataset(df, timeline)
        datasets = [dataset]

    return ApiResponse(datasets=datasets)


def _parse_datetime_range(start_str: str, end_str: str):
    """Парсит строки дат и возвращает datetime объекты. Ошибки → 400."""
    try:
        start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M")
    except ValueError as exc:
        raise ValueError("Формат даты: YYYY-MM-DD HH:MM") from exc

    if end_dt < start_dt:
        raise ValueError("end не может быть раньше start")

    return start_dt, end_dt


def _load_hourly_data_for_period(
        start_dt: datetime,
        end_dt: datetime,
        room_filter: set[str] | None = None,
        exclude_rooms: set[str] | None = None,
) -> pd.DataFrame:
    """
    Загружает все hourly CSV за указанный период и возвращает один большой DataFrame.
    room_filter = None → все комнаты.
    exclude_rooms — комнаты для исключения (применяется после room_filter).
    """
    frames: List[pd.DataFrame] = []

    hour = start_dt.replace(minute=0, second=0, microsecond=0)
    while hour <= end_dt:
        csv_path = HOURS_DIR / f"losses_{hour.strftime('%Y-%m-%d_%H')}.csv"
        if not csv_path.exists():
            hour += timedelta(hours=1)
            continue

        try:
            df = pd.read_csv(csv_path, delimiter=";", header=0)
            df.columns = ["YYYY", "MM", "DD", "HH", "MM_min", "ROOM",
                          "PACKETS", "REACHES", "LOSSES", "PERCENTS"]

            # Безопасное приведение к целым числам (NaN → 0)
            df["PACKETS"] = pd.to_numeric(df["PACKETS"], errors="coerce")
            df["PACKETS"] = df["PACKETS"].fillna(0).astype("Int64")
            df["LOSSES"] = pd.to_numeric(df["LOSSES"], errors="coerce")
            df["LOSSES"] = df["LOSSES"].fillna(0).astype("Int64")
            df["PERCENTS"] = pd.to_numeric(df["PERCENTS"], errors="coerce")
            df["PERCENTS"] = df["PERCENTS"].fillna(0.0)

            # Собираем datetime
            df["dt"] = pd.to_datetime(
                df["YYYY"].astype(str) + "-" +
                df["MM"].astype(str).str.zfill(2) + "-" +
                df["DD"].astype(str).str.zfill(2) + " " +
                df["HH"].astype(str).str.zfill(2) + ":" +
                df["MM_min"].astype(str).str.zfill(2)
            )

            # Фильтруем по времени
            df = df[(df["dt"] >= start_dt) & (df["dt"] <= end_dt)]

            # Фильтруем по комнатам (если нужно)
            if room_filter:
                df = df[df["ROOM"].astype(str).isin(room_filter)]

            # Исключаем комнаты (если указано)
            if exclude_rooms:
                df = df[~df["ROOM"].astype(str).isin(exclude_rooms)]

            if not df.empty:
                frames.append(df)

        except Exception as exc:
            logging.error(f"Ошибка чтения {csv_path}: {exc}")

        hour += timedelta(hours=1)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_room_datasets(
        df: pd.DataFrame,
        timeline: pd.DatetimeIndex,
        rooms: List[str],
) -> List[Dataset]:
    """Строит датасеты для отдельных комнат (режим total и обычный запрос)."""
    datasets: List[Dataset] = []

    for room in sorted(rooms):
        room_df = df[df["ROOM"].astype(str) == str(room)]
        # Словарь datetime → процент потерь
        percent_by_time: Dict[datetime, float] = {
            row.dt: float(row.PERCENTS)
            for row in room_df.itertuples(index=False)
        }

        raw_points = [
            {"x": ts.to_pydatetime().isoformat(), "y": percent_by_time.get(ts.to_pydatetime(), 0.0)}
            for ts in timeline
        ]
        optimized = optimize_stepped_data(raw_points)

        color = colors[hash(room) % len(colors)]
        datasets.append(Dataset(
            label=str(room),
            data=[DataPoint(**p) for p in optimized],
            borderColor=color,
            backgroundColor=color + "50",
            fill=True,
        ))

    return datasets


def _build_summary_dataset(
        df: pd.DataFrame,
        timeline: pd.DatetimeIndex,
) -> Dataset:
    """
    Строит один датасет — суммарный процент потерь по всем комнатам.
    Формула: sum(LOSSES) / sum(PACKETS) * 100
    """
    # Группируем по минуте
    grouped = (
        df.groupby("dt")
        .agg(total_packets=("PACKETS", "sum"), total_losses=("LOSSES", "sum"))
        .reindex(timeline.to_pydatetime(), fill_value=0)
    )

    raw_points = []
    for ts in timeline:
        packets = int(grouped.loc[ts.to_pydatetime(), "total_packets"])
        losses = int(grouped.loc[ts.to_pydatetime(), "total_losses"])
        percent = round(losses / packets * 100, 3) if packets > 0 else 0.0
        raw_points.append({"x": ts.to_pydatetime().isoformat(), "y": percent})

    optimized = optimize_stepped_data(raw_points)

    return Dataset(
        label="Суммарные потери",
        data=[DataPoint(**p) for p in optimized],
        borderColor="#ff6b6b",
        backgroundColor="#ff6b6b50",
        fill=True,
    )
