import hmac
import json
import logging
import math
import os
import queue
import shutil
import signal
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import Config
from .media import Cancelled, MediaError, check, encode_segment, probe, run, segments
from .telegram import Telegram
from .menu import BUTTONS

HELP = '''Пришлите видео — отправлю кружки по 60 секунд, по порядку.
Фото можно отправить как фото или файл. Результат верну PNG-файлом.

/mode safe — бережное увеличение, без нейродорисовки (по умолчанию)
/mode ai — нейроулучшение; мелкие детали могут измениться
/scale 2 или /scale 4 — увеличение
/fit crop — заполнение кружка с обрезкой краёв (по умолчанию)
/fit contain — весь кадр внутри круга с чёрными полями
/status — состояние и лимиты
/cancel — отменить текущую работу и очередь

Рабочие копии удаляются после обработки. Файлы в Telegram остаются.
Размытые детали нельзя гарантированно восстановить без изменений.'''


def purge_jobs(root):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for child in root.glob('job-*'):
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)


class App:
    def __init__(self, cfg, api=None):
        self.cfg, self.api = cfg, api or Telegram(cfg)
        self.jobs = queue.Queue(maxsize=3)
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.active = None
        self.seen = OrderedDict()
        self.settings = {'mode': 'safe', 'scale': 2, 'fit': 'crop'}
        self.status = 'Готов к работе'
        self.ready = False
        self.worker = None
        purge_jobs(cfg.work)

    def say(self, text):
        try:
            self.api.message(text)
        except Exception:
            logging.warning('Telegram notification failed; content omitted')

    def accept(self, update):
        msg = update.get('message') or {}
        if msg.get('from', {}).get('id') != self.cfg.owner or msg.get('chat', {}).get('type') != 'private':
            return True
        if msg.get('chat', {}).get('id') != self.cfg.owner:
            return True
        uid = update.get('update_id')
        if not isinstance(uid, int):
            return True
        with self.lock:
            if uid in self.seen:
                return True
            text = msg.get('text', '')
            text = BUTTONS.get(text, text)
            if text.startswith('/'):
                self.command(text)
            else:
                media = (max(msg['photo'], key=lambda p: p.get('width', 0)*p.get('height', 0))
                         if msg.get('photo') else msg.get('video') or msg.get('animation')
                         or msg.get('video_note') or msg.get('document'))
                if not media or not media.get('file_id'):
                    self.say('Пришлите видео или изображение. /start — помощь.')
                elif self.cfg.max_bytes and media.get('file_size', 0) > self.cfg.max_bytes:
                    self.say('Этот файл больше 20 МБ, а официальный Telegram Bot API не передаёт '
                             'такие входные файлы боту. Сожмите видео на телефоне/компьютере '
                             'или отправьте его как видео в меньшем качестве — сжать файл после '
                             'отправки бот не может, потому что он ещё не получил доступ к файлу.')
                elif self.jobs.full():
                    # Reject with explicit message, not silent HTTP retries and reordered jobs.
                    self.say('Очередь заполнена. Дождитесь результата и отправьте файл ещё раз.')
                else:
                    is_image = bool(msg.get('photo')) or media.get('mime_type', '').startswith('image/')
                    suffix = Path(media.get('file_name', '')).suffix.lower()
                    is_image = is_image or suffix in {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
                    job = {'file_id': media['file_id'], 'image': is_image,
                           'file_size': media.get('file_size', 0),
                           'settings': dict(self.settings), 'cancel': threading.Event()}
                    self.jobs.put_nowait(job)
                    self.say('Принято. Файлы обрабатываются по очереди; /cancel — отмена.')
            self.seen[uid] = None
            if len(self.seen) > 2048:
                self.seen.popitem(last=False)
        return True

    def command(self, text):
        parts = text.split()
        cmd = parts[0].split('@')[0].lower()
        arg = parts[1].lower() if len(parts) == 2 else ''
        if cmd in ('/start', '/help'):
            self.say(HELP)
        elif cmd == '/video':
            self.settings['fit'] = 'crop'
            self.say('Пришлите видео до 20 МБ. Возьму самый большой квадрат строго из центра '
                     'кадра без полей и отправлю кружки по 60 секунд.')
        elif cmd == '/photo':
            self.say('Пришлите фото или изображение файлом. Режим улучшения и увеличение '
                     'можно выбрать кнопками ниже. Результат придёт PNG-файлом.')
        elif cmd == '/status':
            input_limit = (f'{self.cfg.max_bytes // 1024 // 1024} МБ' if self.cfg.max_bytes
                           else 'по свободному диску и ресурсам сервера')
            self.say(f'{self.status}\nВ очереди: {self.jobs.qsize()}\n'
                     f'Лимит входа: {input_limit}; '
                     f'выход фото: до {self.cfg.max_pixels / 1e6:g} Мп.\n'
                     f'Режим: {self.settings["mode"]}, ×{self.settings["scale"]}; '
                     f'кадр: {self.settings["fit"]}.')
        elif cmd == '/cancel':
            if self.active:
                self.active['cancel'].set()
            while True:
                try:
                    self.jobs.get_nowait()['cancel'].set()
                    self.jobs.task_done()
                except queue.Empty:
                    break
            self.say('Отмена запрошена. Уже отправленные результаты останутся в чате.')
        elif cmd == '/mode' and arg in ('safe', 'ai'):
            if arg == 'ai' and not self.cfg.model.is_file():
                self.say('Нейромодель пока не установлена на сервере. Доступен /mode safe.')
                return
            self.settings['mode'] = arg
            self.say('Режим установлен: ' + arg + ('. Нейросеть может изменять мелкие детали.' if arg == 'ai' else '.'))
        elif cmd == '/scale' and arg in ('2', '4'):
            self.settings['scale'] = int(arg)
            self.say(f'Увеличение ×{arg}, в пределах лимита разрешения сервера.')
        elif cmd == '/fit' and arg in ('crop', 'contain'):
            self.settings['fit'] = arg
            self.say('Кадрирование установлено: ' + arg)
        else:
            self.say('Неизвестная команда или параметр. /start — помощь.')

    def process(self, job):
        deadline = time.monotonic() + self.cfg.job_seconds
        cancel, settings = job['cancel'], job['settings']
        if self.cfg.local_root:
            # Reserve space for the downloaded input, a working copy and one output.
            available = min(shutil.disk_usage(self.cfg.work).free,
                            shutil.disk_usage(self.cfg.local_root).free)
            if 2 * job.get('file_size', 0) + 256 * 1024 * 1024 > available:
                raise MediaError('На сервере сейчас недостаточно свободного места для этого файла.')
        with tempfile.TemporaryDirectory(prefix='job-', dir=self.cfg.work) as folder:
            folder = Path(folder)
            source = folder / 'input.bin'
            self.status = 'Скачиваю файл'
            self.api.download(job['file_id'], source, cancel, deadline)
            check(cancel, deadline)
            if job['image']:
                self.status = 'Улучшаю изображение'
                output = folder / 'enhanced.png'
                run([sys.executable, '-m', 'bot.upscale', source, output,
                     '--mode', settings['mode'], '--scale', settings['scale'],
                     '--pixels', self.cfg.max_pixels, '--model', self.cfg.model], cancel, deadline,
                    stage='image')
                if output.stat().st_size > 49 * 1024 * 1024:
                    raise MediaError('Изображение слишком велико для отправки. Уменьшите масштаб.')
                check(cancel, deadline)
                self.api.upload(output, 'image', caption='Улучшенное изображение · ' + settings['mode'])
                return 1
            duration = probe(source, cancel, deadline)
            count = math.ceil(duration / 60)
            self.say(f'Видео: {duration:.1f} с. Кружков: {count}. Отправляю по порядку.')
            for i, start, length in segments(duration):
                self.status = f'Готовлю кружок {i+1}/{count}'
                output = folder / f'circle-{i+1:03}.mp4'
                encode_segment(source, output, start, length, self.cfg, settings['fit'], cancel, deadline)
                check(cancel, deadline)
                self.api.upload(output, 'video', duration=min(60, math.ceil(length)), length=self.cfg.video_size)
                output.unlink()
                if cancel.wait(1.1):
                    raise Cancelled('Задание отменено.')
            return count

    def work(self):
        while not self.stop.is_set():
            with self.lock:
                try:
                    job = self.jobs.get_nowait()
                except queue.Empty:
                    job = None
                if job:
                    self.active = job
            if job is None:
                self.stop.wait(0.2)
                continue
            try:
                self.process(job)
                self.say('Готово. Рабочие файлы бота удалены.')
            except MediaError as exc:
                self.say(str(exc) + '\nРабочие файлы задания очищены. Уже отправленные части остаются в чате.')
            except Exception:
                # Never emit exception repr/traceback containing token URLs or user metadata.
                logging.error('Job failed; details deliberately omitted')
                self.say('Обработка прервана. Проверьте размер/формат и попробуйте ещё раз. '
                         'Для фото можно выбрать /mode safe. Уже отправленные части остаются в чате.')
            finally:
                with self.lock:
                    self.active = None
                    self.status = 'Готов к работе'
                self.jobs.task_done()

    def shutdown(self):
        self.stop.set()
        with self.lock:
            if self.active:
                self.active['cancel'].set()


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, code, body=b''):
            self.send_response(code)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.respond(200 if app.ready else 503, b'ok' if app.ready else b'starting')

        def do_POST(self):
            self.connection.settimeout(10)
            if self.path != '/telegram' or not hmac.compare_digest(
                    self.headers.get('X-Telegram-Bot-Api-Secret-Token', ''), app.cfg.secret):
                self.respond(403)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 256 * 1024:
                    self.respond(413)
                    return
                update = json.loads(self.rfile.read(size))
                if not isinstance(update, dict):
                    raise ValueError()
                app.accept(update)
                self.respond(200)
            except (ValueError, TypeError, KeyError):
                self.respond(400)
            except Exception:
                self.respond(503)
    return Handler


def main():
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
    logging.getLogger('httpx').setLevel(logging.CRITICAL)
    cfg = Config.from_env()
    app = App(cfg)
    signal.signal(signal.SIGTERM, lambda *_: app.shutdown())
    signal.signal(signal.SIGINT, lambda *_: app.shutdown())
    app.worker = threading.Thread(target=app.work, daemon=True)
    app.worker.start()
    server = None
    try:
        if cfg.delivery == 'webhook':
            server = ThreadingHTTPServer(('0.0.0.0', int(os.environ.get('PORT', '10000'))), handler_for(app))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            app.api.call('setWebhook', {'url': cfg.public_url + '/telegram', 'secret_token': cfg.secret,
                                       'allowed_updates': ['message'], 'max_connections': 1})
            app.ready = True
            app.stop.wait()
        else:
            app.api.call('deleteWebhook', {'drop_pending_updates': False})
            app.ready = True
            offset = 0
            while not app.stop.is_set():
                try:
                    updates = app.api.call('getUpdates', {'offset': offset, 'timeout': 25,
                                                          'allowed_updates': ['message']})
                    for update in updates:
                        app.accept(update)
                        offset = max(offset, update['update_id'] + 1)
                except Exception:
                    logging.warning('Polling failed; retrying without logging secrets')
                    app.stop.wait(5)
    finally:
        app.shutdown()
        if server:
            server.shutdown()
        app.worker.join(timeout=130)
        if not app.worker.is_alive():
            purge_jobs(cfg.work)


if __name__ == '__main__':
    main()
