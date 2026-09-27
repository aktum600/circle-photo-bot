import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from bot.app import App, handler_for
from bot.config import Config
from bot.media import Cancelled, MediaError, check, segments
from bot.upscale import target_size, upscale


@pytest.fixture
def cfg(tmp_path):
    return Config('test-token', 123, 's'*32, 'https://example.test', work=tmp_path / 'jobs')


class FakeAPI:
    def __init__(self):
        self.messages = []
        self.uploads = []

    def message(self, text):
        self.messages.append(text)

    def download(self, file_id, dest, cancel, deadline):
        Image.new('RGB', (16, 12), 'red').save(dest, format='PNG')

    def upload(self, path, kind, **kwargs):
        assert path.exists()
        self.uploads.append((kind, kwargs))


def update(uid=1, owner=123, **content):
    return {'update_id': uid, 'message': {'from': {'id': owner},
            'chat': {'id': owner, 'type': 'private'}, **content}}


@pytest.mark.parametrize('duration,lengths', [(0.1, [0.1]), (60, [60]), (120, [60, 60]),
                                              (180, [60, 60, 60]), (125.5, [60, 60, 5.5])])
def test_split_covers_entire_video(duration, lengths):
    result = list(segments(duration))
    assert [r[2] for r in result] == lengths
    assert sum(r[2] for r in result) == duration
    assert result[-1][1] + result[-1][2] == duration


def test_access_and_duplicate_delivery(cfg):
    api = FakeAPI()
    app = App(cfg, api)
    app.accept(update(owner=999, document={'file_id': 'x'}))
    assert not api.messages and app.jobs.empty()
    item = update(document={'file_id': 'x', 'mime_type': 'image/png'})
    app.accept(item)
    app.accept(item)
    assert app.jobs.qsize() == 1


def test_size_rejected_before_download_and_queue_bounded(cfg):
    app = App(cfg, FakeAPI())
    app.accept(update(document={'file_id': 'x', 'file_size': cfg.max_bytes+1}))
    assert app.jobs.empty()
    for uid in range(2, 10):
        app.accept(update(uid, document={'file_id': 'x'}))
    assert app.jobs.qsize() == 3


def test_cancel_current_and_pending(cfg):
    app = App(cfg, FakeAPI())
    app.active = {'cancel': threading.Event()}
    app.accept(update(document={'file_id': 'x'}))
    app.accept(update(2, text='/cancel'))
    assert app.active['cancel'].is_set() and app.jobs.empty()


def test_failed_download_removes_partial_file(cfg):
    api = FakeAPI()
    def failing(file_id, dest, *_):
        dest.write_bytes(b'private content')
        raise MediaError('interrupted')
    api.download = failing
    app = App(cfg, api)
    with pytest.raises(MediaError):
        app.process({'file_id': 'x', 'image': True, 'cancel': threading.Event(), 'settings': app.settings})
    assert list(cfg.work.iterdir()) == []


def test_success_image_process_cleans_files(cfg):
    api = FakeAPI()
    app = App(cfg, api)
    app.process({'file_id': 'x', 'image': True, 'cancel': threading.Event(), 'settings': app.settings})
    assert len(api.uploads) == 1 and api.uploads[0][0] == 'image'
    assert list(cfg.work.iterdir()) == []


def test_upload_failure_still_cleans_files(cfg):
    api = FakeAPI()
    api.upload = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('network error'))
    app = App(cfg, api)
    with pytest.raises(RuntimeError):
        app.process({'file_id': 'x', 'image': True, 'cancel': threading.Event(), 'settings': app.settings})
    assert list(cfg.work.iterdir()) == []


def test_image_size_alpha_metadata(tmp_path):
    source, dest = tmp_path/'in.png', tmp_path/'out.png'
    Image.new('RGBA', (20, 10), (100, 20, 30, 90)).save(source)
    upscale(source, dest, 'safe', 4, 10_000, tmp_path/'absent.onnx')
    with Image.open(dest) as result:
        assert result.size == (80, 40)
        assert result.mode == 'RGBA'
        assert result.getpixel((0, 0))[3] == 90
        assert not result.getexif()
    with pytest.raises(ValueError):
        target_size((4000, 4000), 2, 8_000_000)


def test_cancel_and_deadline():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        check(cancel, time.monotonic()+100)
    with pytest.raises(MediaError):
        check(threading.Event(), time.monotonic()-1)


def test_config_cloud_limit(monkeypatch):
    with patch.dict('os.environ', {'BOT_TOKEN': 'test', 'OWNER_ID': '123', 'MAX_INPUT_MB': '2048'}, clear=True):
        assert Config.from_env().max_bytes == 20*1024*1024


def test_local_path_restriction(cfg, tmp_path):
    from bot.telegram import Telegram
    api = Telegram(replace(cfg, local_root=tmp_path/'allowed'))
    secret = tmp_path/'outside.txt'
    secret.write_text('never read')
    api.call = lambda *a, **kw: {'file_path': str(secret)}
    with pytest.raises(MediaError):
        api.download('x', tmp_path/'out', threading.Event(), time.monotonic()+20)
    assert secret.exists()
    assert not (tmp_path/'out').exists()


def test_webhook_auth(cfg):
    import httpx
    from http.server import ThreadingHTTPServer
    app = App(cfg, FakeAPI())
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/telegram'
    try:
        with httpx.Client(trust_env=False) as client:
            assert client.post(url, json=update(text='/start')).status_code == 403
            assert not app.api.messages
            assert client.post(url, json=update(text='/start'), headers={
                'X-Telegram-Bot-Api-Secret-Token': cfg.secret}).status_code == 200
            assert len(app.api.messages) == 1
    finally:
        server.shutdown()
        server.server_close()
