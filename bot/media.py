import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path


class MediaError(Exception):
    pass


class Cancelled(MediaError):
    pass


def check(cancel: threading.Event, deadline: float):
    if cancel.is_set():
        raise Cancelled('Задание отменено.')
    if time.monotonic() >= deadline:
        raise MediaError('Превышено время обработки на этом сервере. Попробуйте меньший файл.')


def run(args, cancel, deadline, capture=False):
    check(cancel, deadline)
    # Do not record command output: media metadata and private paths can be sensitive.
    process = subprocess.Popen([str(x) for x in args], stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        while True:
            check(cancel, deadline)
            try:
                stdout, _ = process.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise MediaError('Не удалось обработать файл: повреждение, неподдерживаемый формат или нехватка ресурсов.')
        return stdout
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def probe(path, cancel, deadline):
    raw = run([os.environ.get('FFPROBE', 'ffprobe'), '-v', 'error',
               '-protocol_whitelist', 'file,pipe', '-show_entries',
               'format=duration:stream=codec_type,width,height,duration', '-of', 'json', path],
              cancel, deadline, capture=True)
    try:
        info = json.loads(raw)
        video = next(s for s in info['streams'] if s['codec_type'] == 'video')
        raw_duration = video.get('duration')
        if raw_duration in (None, 'N/A', '0', 0):
            raw_duration = info.get('format', {}).get('duration', 0)
        duration = float(raw_duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError()
        if int(video['width']) * int(video['height']) > 40_000_000:
            raise MediaError('Разрешение видео превышает возможности этого сервера.')
        return duration
    except (ValueError, KeyError, StopIteration, TypeError):
        raise MediaError('В файле не найдено видео с определённой длительностью.') from None


def segments(duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid duration')
    # No keyframe copy: every segment is encoded with exact time trimming.
    for i in range(math.ceil(duration / 60)):
        yield i, i * 60, min(60, duration - i * 60)


def video_filter(size, fit):
    # Correct non-square source pixels before fitting. FFmpeg autorotates phone videos.
    prefix = 'scale=trunc(iw*sar/2)*2:ih,setsar=1,'
    if fit == 'crop':
        return prefix + (f'scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size},'
                         'tpad=stop_mode=clone:stop_duration=1,fps=30')
    # Inscribe the complete frame into the visible circular area, including corners.
    inner = int(size / math.sqrt(2)) // 2 * 2
    return prefix + (f'scale={inner}:{inner}:force_original_aspect_ratio=decrease:force_divisible_by=2,'
                     f'pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=black,'
                     'tpad=stop_mode=clone:stop_duration=1,fps=30')


def encode_segment(source, output, start, duration, cfg, fit, cancel, deadline):
    # Reserve container overhead, audio and a two-second VBV burst. CRF keeps
    # simple scenes small while the rate cap budgets complex scenes by duration.
    rate = min(5_000_000, int(cfg.output_max_bytes * 8 * 0.92 / (duration + 2)) - 128_000)
    run([os.environ.get('FFMPEG', 'ffmpeg'), '-nostdin', '-v', 'error', '-y',
         '-threads', str(cfg.threads), '-protocol_whitelist', 'file,pipe',
         '-ss', f'{start:.6f}', '-i', source, '-t', f'{duration:.6f}',
         '-map', '0:v:0', '-map', '0:a:0?', '-map_metadata', '-1', '-map_chapters', '-1',
         '-vf', video_filter(cfg.video_size, fit), '-filter_threads', '1',
         '-c:v', 'libx264', '-preset', 'fast', '-crf', str(cfg.crf),
         '-maxrate', str(rate), '-bufsize', str(rate * 2), '-threads', str(cfg.threads),
         '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', '-ac', '2',
         '-af', 'aresample=async=1:first_pts=0', '-movflags', '+faststart', output],
        cancel, deadline)
    if output.stat().st_size > cfg.output_max_bytes:
        raise MediaError('Кружок превысил заданный размер. Попробуйте меньший фрагмент.')
    actual = probe(output, cancel, deadline)
    if actual > 60.05:
        raise MediaError('Не удалось соблюсти длительность кружка.')
