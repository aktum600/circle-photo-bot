import json
import math
import os
import subprocess
import threading
import time
from collections import deque
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


def process_error(code, stderr, stage):
    if stage == 'image':
        known = {
            20: 'Изображение уже достигает предела разрешения этого сервера; увеличить его нельзя.',
            21: 'Фото слишком велико для нейрообработки на бесплатном сервере. Выберите «Бережное улучшение».',
            22: 'Нейромодель недоступна. Выберите «Бережное улучшение».',
            23: 'Не удалось открыть изображение. Пришлите JPEG, PNG или WebP без анимации.',
        }
        if code in known:
            return known[code]
    detail = stderr.lower()
    if code in (-9, 137) or b'cannot allocate memory' in detail or b'out of memory' in detail:
        return 'Процесс остановлен сервером, возможно из-за нехватки памяти. Попробуйте меньшее разрешение.'
    if b'no space left' in detail:
        return 'На сервере закончилось свободное место. Попробуйте меньший файл.'
    if b'unknown encoder' in detail or b'no such filter' in detail or b'option not found' in detail:
        return 'На сервере недоступна нужная функция обработки. Сообщите разработчику код: MEDIA_FEATURE.'
    if b'invalid data found' in detail or b'moov atom not found' in detail:
        return 'Не удалось прочитать видео. Попробуйте заново отправить его как видео, а не файл.'
    label = {'image': 'улучшение фото', 'probe': 'чтение видео', 'encode': 'создание кружка'}.get(stage, 'обработка')
    return f'Не удалось выполнить этап «{label}». Код: {stage.upper()}_{code}. Попробуйте другой файл.'


def run(args, cancel, deadline, capture=False, stage='media'):
    check(cancel, deadline)
    # Do not record command output: media metadata and private paths can be sensitive.
    process = subprocess.Popen([str(x) for x in args], stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                               stderr=subprocess.PIPE,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    # Drain stderr to bounded RAM; never store or expose paths or media metadata.
    errors = deque(maxlen=8)
    stderr_pipe = process.stderr
    def drain():
        while chunk := stderr_pipe.read(4096):
            errors.append(chunk)
    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    process.stderr = None  # communicate must not race the draining thread.
    try:
        while True:
            check(cancel, deadline)
            try:
                stdout, _ = process.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            reader.join()
            raise MediaError(process_error(process.returncode, b''.join(errors), stage))
        return stdout
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        reader.join()
        stderr_pipe.close()


def probe(path, cancel, deadline):
    raw = run([os.environ.get('FFPROBE', 'ffprobe'), '-v', 'error',
               '-protocol_whitelist', 'file,pipe', '-show_entries',
               'format=duration:stream=codec_type,width,height,duration', '-of', 'json', path],
              cancel, deadline, capture=True, stage='probe')
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
        # Crop before scaling: largest centered square in display pixels,
        # including anamorphic SAR. Avoid a full-resolution intermediate scale.
        return ("crop=w='min(iw,ih/sar)':h='min(ih,iw*sar)':x='(iw-ow)/2':y='(ih-oh)/2',"
                f'scale={size}:{size},setsar=1,tpad=stop_mode=clone:stop_duration=1,fps=30')
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
        cancel, deadline, stage='encode')
    if output.stat().st_size > cfg.output_max_bytes:
        raise MediaError('Кружок превысил заданный размер. Попробуйте меньший фрагмент.')
    actual = probe(output, cancel, deadline)
    if actual > 60.05:
        raise MediaError('Не удалось соблюсти длительность кружка.')
