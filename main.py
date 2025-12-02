import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
import asyncio
import aiofiles
from aiohttp.web_request import FileField
from pydantic import BaseModel
from quart import Quart, request, jsonify
from quart_schema import QuartSchema, validate_querystring, validate_response, validate_request, DataSource
from quart_schema.pydantic import File
from werkzeug.utils import secure_filename
from hypercorn.asyncio import serve
from hypercorn.config import Config
import zipfile
from graph import graph_bp

from losses_proccessor import process_losses

app = Quart(__name__)
QuartSchema(app)
app.register_blueprint(graph_bp)
logging.basicConfig(level=logging.INFO)


# TODO
async def process_ping(room: str, file: Path):
    pass


# TODO
async def process_trace(room: str, file: Path):
    pass


async def append_analytics(room: str, file: Path) -> None:
    # Целевая папка = имя файла без .zip
    target_dir = file.with_suffix("")

    # 1. Распаковка
    target_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    # 2. Поиск файлов внутри папки
    ping_file = next(target_dir.glob("ping_*.jsonl"), None)
    trace_file = next(target_dir.glob("trace_*.jsonl"), None)
    losses_file = next(target_dir.glob("losses_*.json"), None)

    # 3. Передать их в обработчики (если файлы есть)
    if ping_file:
        await process_ping(room, ping_file)

    if trace_file:
        await process_trace(room, trace_file)

    if losses_file:
        await process_losses(room, losses_file)

    # 4. Удаление всей временной директории
    shutil.rmtree(target_dir)


class Upload(BaseModel):
    file: File


@dataclass
class Status:
    status: str


@app.route('/api/upload/<room>/', methods=['POST'])
@validate_request(Upload, source=DataSource.FORM_MULTIPART)
@validate_response(Status)
async def upload_data(room: str, data: Upload):
    file = data.file

    if not file:
        return jsonify({"error": "No file part"}), 400
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not file.filename.endswith('.zip'):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    join = Path(f'data/{room}') / filename
    os.makedirs(join.parent, exist_ok=True)

    # Асинхронная запись файла
    async with aiofiles.open(join, 'wb') as f:
        content = file.read()
        await f.write(content)

    await append_analytics(room, join)

    return Status(status="success")


async def main():
    config = Config()
    config.bind = ["0.0.0.0:8080"]
    await serve(app, config)


if __name__ == '__main__':
    asyncio.run(main())
