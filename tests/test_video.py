"""Real FFmpeg integration tests; require ffmpeg and ffprobe (env or PATH)."""
import json
import os
import shutil
import subprocess
import threading
import time

import pytest

from bot.config import Config
from bot.media import encode_segment, probe, segments


@pytest.mark.parametrize('audio', [True, False])
def test_real_video_roundtrip(tmp_path, audio):
    ffmpeg = os.environ.get('FFMPEG') or shutil.which('ffmpeg')
    ffprobe = os.environ.get('FFPROBE') or shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        pytest.skip('Install FFmpeg + ffprobe to run integration test')
    source = tmp_path / 'source.mkv'
    args = [ffmpeg, '-v', 'error', '-nostdin', '-y', '-f', 'lavfi',
            '-i', 'testsrc2=size=320x180:rate=5']
    if audio:
        args += ['-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000']
    args += ['-t', '125', '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p']
    if audio:
        args += ['-c:a', 'aac']
    subprocess.run(args + [str(source)], check=True, capture_output=True)
    cfg = Config('test', 123, 'x'*32, 'https://example.test', work=tmp_path)
    cancel, deadline = threading.Event(), time.monotonic()+240
    duration = probe(source, cancel, deadline)
    outputs = []
    for i, start, length in segments(duration):
        output = tmp_path/f'part-{i}.mp4'
        encode_segment(source, output, start, length, cfg, 'crop' if audio else 'contain', cancel, deadline)
        meta = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-show_streams',
                                                  '-show_format', '-of', 'json', str(output)]))
        video = next(s for s in meta['streams'] if s['codec_type'] == 'video')
        assert (video['width'], video['height'], video['codec_name'], video['pix_fmt']) == (640, 640, 'h264', 'yuv420p')
        assert any(s['codec_type'] == 'audio' for s in meta['streams']) == audio
        outputs.append(float(meta['format']['duration']))
    assert len(outputs) == 3
    assert all(0 < d <= 60.001 for d in outputs)
    assert abs(sum(outputs)-duration) < 0.15
