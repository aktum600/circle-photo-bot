from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bot.upscale import neural, upscale


MODEL = Path('models/realesr-general-x4v3.onnx')


@pytest.mark.skipif(not MODEL.is_file(), reason='Export official model first')
def test_real_neural_tiles_match_full_inference(tmp_path):
    import onnxruntime as ort
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 256, (75, 83, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    tiled = np.asarray(neural(image, MODEL, (332, 300)), dtype=np.float32)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(MODEL), sess_options=opts, providers=['CPUExecutionProvider'])
    tensor = pixels.astype(np.float32).transpose(2, 0, 1)[None] / 255
    full = session.run(None, {session.get_inputs()[0].name: tensor})[0][0].transpose(1, 2, 0)
    expected = np.rint(np.clip(full, 0, 1) * 255)
    assert np.abs(tiled-expected).max() <= 1
    source, output = tmp_path/'source.png', tmp_path/'result.png'
    image.save(source)
    upscale(source, output, 'ai', 2, 8_000_000, MODEL)
    with Image.open(output) as result:
        assert result.size == (166, 150)
        assert not result.getexif()
