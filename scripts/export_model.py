"""Export the official compact Real-ESRGAN weights; no user media involved.

SRVGG architecture adapted from xinntao/Real-ESRGAN (BSD-3-Clause).
See THIRD_PARTY.md. PyTorch is required only at build time.
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class Compact(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(3, 64, 3, 1, 1), nn.PReLU(64)]
        for _ in range(32):
            layers.extend([nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(64)])
        layers.append(nn.Conv2d(64, 48, 3, 1, 1))
        self.body = nn.ModuleList(layers)
        self.upsampler = nn.PixelShuffle(4)

    def forward(self, x):
        y = x
        for layer in self.body:
            y = layer(y)
        return self.upsampler(y) + F.interpolate(x, scale_factor=4, mode='nearest')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('models/realesr-general-x4v3.onnx'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights = args.output.with_suffix('.pth')
    if not weights.exists():
        urllib.request.urlretrieve(
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth', weights)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != '8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292':
        raise ValueError('Official model checksum mismatch; refusing to load weights')
    print('Official weights SHA256:', digest)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    model = Compact().eval()
    model.load_state_dict(state.get('params_ema', state.get('params', state)), strict=True)
    torch.set_num_threads(1)
    sample = torch.rand(1, 3, 24, 32)
    with torch.inference_mode():
        torch.onnx.export(model, sample, str(args.output), input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {2: 'height', 3: 'width'},
                                        'output': {2: 'height4', 3: 'width4'}},
                          opset_version=17, dynamo=False)
    # Verify numerical parity against the original weights, including another shape.
    import numpy as np
    import onnxruntime as ort
    session = ort.InferenceSession(str(args.output), providers=['CPUExecutionProvider'])
    for height, width in [(24, 32), (31, 19)]:
        tensor = torch.rand(1, 3, height, width)
        with torch.inference_mode():
            expected = model(tensor).numpy()
        actual = session.run(None, {'input': tensor.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=2e-4)
    print('Export and numerical parity check passed.')


if __name__ == '__main__':
    main()
