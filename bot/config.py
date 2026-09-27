import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Config:
    token: str
    owner: int
    secret: str
    public_url: str
    api_url: str = 'https://api.telegram.org'
    local_root: Path | None = None
    work: Path = Path('/tmp/circle-bot')
    model: Path = Path('/app/models/realesr-general-x4v3.onnx')
    max_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 8_000_000
    job_seconds: int = 660
    video_size: int = 640
    crf: int = 18
    threads: int = 1
    delivery: str = 'polling'

    @classmethod
    def from_env(cls):
        token = os.environ.get('BOT_TOKEN', '').strip()
        secret = os.environ.get('WEBHOOK_SECRET', '').strip()
        owner = int(os.environ.get('OWNER_ID') or 0)
        delivery = os.environ.get('DELIVERY', 'polling')
        url = os.environ.get('PUBLIC_URL', os.environ.get('RENDER_EXTERNAL_URL', '')).rstrip('/')
        if not token or owner <= 0:
            raise ValueError('Set BOT_TOKEN and positive OWNER_ID.')
        if delivery not in ('polling', 'webhook'):
            raise ValueError('DELIVERY must be polling or webhook.')
        if delivery == 'webhook' and (not url.startswith('https://') or len(secret) < 32):
            raise ValueError('PUBLIC_URL must be your public HTTPS address.')
        if delivery == 'webhook' and (not all(c.isalnum() or c in '_-' for c in secret) or not secret.isascii()):
            raise ValueError('WEBHOOK_SECRET must contain only A-Z, a-z, 0-9, _ and -.')
        api = os.environ.get('BOT_API_URL', 'https://api.telegram.org').rstrip('/')
        root = os.environ.get('LOCAL_FILES_ROOT', '')
        limit = int(os.environ.get('MAX_INPUT_MB', '20')) * 1024 * 1024
        if urlparse(api).hostname == 'api.telegram.org':
            limit = min(limit, 20 * 1024 * 1024)
        elif not root:
            raise ValueError('Custom local Bot API requires LOCAL_FILES_ROOT shared with the server.')
        cfg = cls(token, owner, secret, url, api, Path(root).resolve() if root else None,
                  Path(os.environ.get('WORK_DIR', '/tmp/circle-bot')).resolve(),
                  Path(os.environ.get('MODEL_PATH', '/app/models/realesr-general-x4v3.onnx')),
                  limit, int(os.environ.get('MAX_OUTPUT_PIXELS', '8000000')),
                  int(os.environ.get('MAX_JOB_SECONDS', '660')),
                  int(os.environ.get('VIDEO_SIZE', '640')),
                  int(os.environ.get('VIDEO_CRF', '18')),
                  int(os.environ.get('FFMPEG_THREADS', '1')), delivery)
        if not (cfg.max_bytes > 0 and cfg.max_pixels > 0 and cfg.job_seconds > 0
                and 240 <= cfg.video_size <= 1080 and cfg.video_size % 2 == 0
                and 16 <= cfg.crf <= 28 and 1 <= cfg.threads <= 8):
            raise ValueError('Invalid resource or encoding limits.')
        return cfg
