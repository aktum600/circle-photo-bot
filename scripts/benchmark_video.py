"""Local synthetic benchmark; never reads user media."""
import json
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot.config import Config
from bot.media import encode_segment

with tempfile.TemporaryDirectory(prefix='circle-benchmark-') as temp:
    root = Path(temp)
    source = root/'synthetic.mp4'
    subprocess.run([os.environ['FFMPEG'], '-v', 'error', '-y', '-f', 'lavfi',
        '-i', 'testsrc2=size=1280x720:rate=30', '-t', '10', '-c:v', 'libx264',
        '-threads', '1', '-preset', 'ultrafast', str(source)], check=True)
    cfg = Config('unused', 1, '', '')
    results = {}
    for preset in ('fast', 'veryfast'):
        started = time.monotonic()
        output = root/(preset+'.mp4')
        encode_segment(source, output, 0, 10, replace(cfg, video_preset=preset),
                       'crop', threading.Event(), time.monotonic()+120)
        results[preset] = {'seconds': round(time.monotonic()-started, 2),
                           'bytes': output.stat().st_size}
    print(json.dumps(results))
