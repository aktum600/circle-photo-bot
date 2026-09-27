"""Isolated image worker: exits after each job to release the model's memory."""
import argparse
import math
import warnings
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

Image.MAX_IMAGE_PIXELS = 40_000_000
warnings.simplefilter('error', Image.DecompressionBombWarning)


class ImageProblem(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__('Image processing limit')


def target_size(size, scale, max_pixels):
    w, h = size
    factor = min(scale, math.sqrt(max_pixels / (w * h)))
    if factor <= 1:
        raise ImageProblem(20)
    return max(1, int(w * factor)), max(1, int(h * factor))


def neural(image, model, output_size):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
    name = session.get_inputs()[0].name
    # Real-ESRGAN's trained tensors use RGB. Halo covers the compact network's
    # receptive field (34 convolutions), avoiding visible tile seams.
    tile, halo = 64, 36
    width, height = image.size
    result = Image.new('RGB', (width * 4, height * 4))
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            x1, y1 = max(0, x-halo), max(0, y-halo)
            x2, y2 = min(width, x+tile+halo), min(height, y+tile+halo)
            arr = np.asarray(image.crop((x1, y1, x2, y2)), dtype=np.float32) / 255
            tensor = np.transpose(arr, (2, 0, 1))[None]
            pred = session.run(None, {name: tensor})[0][0]
            pred = np.transpose(np.clip(pred, 0, 1), (1, 2, 0))
            patch = Image.fromarray(np.rint(pred * 255).astype(np.uint8))
            left, top = (x-x1)*4, (y-y1)*4
            right, bottom = min(tile, width-x)*4, min(tile, height-y)*4
            result.paste(patch.crop((left, top, left+right, top+bottom)), (x*4, y*4))
    return result.resize(output_size, Image.Resampling.LANCZOS)


def upscale(source: Path, dest: Path, mode, scale, max_pixels, model):
    with Image.open(source) as opened:
        if getattr(opened, 'n_frames', 1) > 1:
            raise ImageProblem(23)
        # Reject oversized headers before loading/copying all decoded pixels.
        target_size(opened.size, scale, max_pixels)
        image = ImageOps.exif_transpose(opened)
        size = target_size(image.size, scale, max_pixels)
        alpha = image.convert('RGBA').getchannel('A') if 'A' in image.getbands() or 'transparency' in image.info else None
        image = image.convert('RGB')
    baseline = image.resize(size, Image.Resampling.LANCZOS)
    if mode == 'ai':
        if not model.is_file():
            raise ImageProblem(22)
        # Bound inference cost independently from source/output resolution.
        # Keep original detail in the baseline; inference is only a gentle blend.
        working = image.copy()
        working.thumbnail((512, 512), Image.Resampling.LANCZOS)
        enhanced = neural(working, model, size)
        result = Image.blend(baseline, enhanced, 0.45)
    else:
        result = baseline.filter(ImageFilter.UnsharpMask(radius=1.2, percent=65, threshold=4))
    if alpha is not None:
        result.putalpha(alpha.resize(size, Image.Resampling.LANCZOS))
    # A new image carries no EXIF, GPS, source ICC or textual metadata.
    result.info.clear()
    result.save(dest, format='PNG', optimize=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('dest', type=Path)
    parser.add_argument('--mode', choices=['safe', 'ai'], required=True)
    parser.add_argument('--scale', type=int, choices=[2, 4], required=True)
    parser.add_argument('--pixels', type=int, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    try:
        upscale(args.source, args.dest, args.mode, args.scale, args.pixels, args.model)
    except ImageProblem as exc:
        sys.exit(exc.code)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning, OSError):
        sys.exit(23)
