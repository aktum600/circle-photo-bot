import json
import time
from pathlib import Path
from urllib.parse import quote

import httpx

from .media import MediaError, check
from .menu import KEYBOARD


class Telegram:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = httpx.Client(timeout=httpx.Timeout(120, connect=20), follow_redirects=False)

    def call(self, method, data=None, files=None):
        data = {key: json.dumps(value) if isinstance(value, (dict, list, bool)) else str(value)
                for key, value in (data or {}).items()}
        for attempt in range(3):
            # A network timeout after send may mean success. Do not blindly retry uploads.
            response = self.client.post(f'{self.cfg.api_url}/bot{self.cfg.token}/{method}', data=data, files=files)
            try:
                payload = response.json()
            except ValueError:
                raise MediaError('Telegram временно недоступен.') from None
            if payload.get('ok'):
                return payload['result']
            if payload.get('error_code') == 429 and attempt < 2:
                delay = int(payload.get('parameters', {}).get('retry_after', 1))
                if delay > 30:
                    break
                time.sleep(max(1, delay))
                if files:
                    for _, fileobj, *_ in files.values():
                        fileobj.seek(0)
                continue
            break
        raise MediaError('Telegram отклонил запрос. Попробуйте позднее; для больших файлов нужен локальный Bot API.')

    def message(self, text):
        return self.call('sendMessage', {'chat_id': self.cfg.owner, 'text': text,
                                         'reply_markup': KEYBOARD})

    def upload(self, path, kind, **extra):
        method, field = ('sendVideoNote', 'video_note') if kind == 'video' else ('sendDocument', 'document')
        mime = 'video/mp4' if kind == 'video' else 'image/png'
        with path.open('rb') as stream:
            return self.call(method, {'chat_id': self.cfg.owner, **extra},
                             {field: (path.name, stream, mime)})

    def download(self, file_id, dest, cancel, deadline):
        result = self.call('getFile', {'file_id': file_id})
        path = result['file_path']
        if self.cfg.local_root:
            source = Path(path).resolve()
            if not source.is_relative_to(self.cfg.local_root) or source == self.cfg.local_root:
                raise MediaError('Bot API вернул путь вне разрешённого каталога.')
            try:
                if self.cfg.max_bytes and source.stat().st_size > self.cfg.max_bytes:
                    raise MediaError('Файл превышает лимит этого сервера.')
                with source.open('rb') as incoming, dest.open('wb') as outgoing:
                    while chunk := incoming.read(1024 * 1024):
                        check(cancel, deadline)
                        outgoing.write(chunk)
            finally:
                source.unlink(missing_ok=True)
            return
        url = f'{self.cfg.api_url}/file/bot{self.cfg.token}/{quote(path, safe="/")}'
        total = 0
        with self.client.stream('GET', url) as response:
            response.raise_for_status()
            with dest.open('wb') as stream:
                for chunk in response.iter_bytes(1024 * 256):
                    check(cancel, deadline)
                    total += len(chunk)
                    if self.cfg.max_bytes and total > self.cfg.max_bytes:
                        raise MediaError('Файл превышает лимит этого сервера.')
                    stream.write(chunk)
