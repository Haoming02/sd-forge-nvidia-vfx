from functools import wraps

import numpy as np
import nvvfx
import torch
from PIL import Image

from modules import devices, modelloader, shared
from modules.script_callbacks import on_script_unloaded
from modules.upscaler import Upscaler, UpscalerData


class UpscalerNvidia(Upscaler):
    name = "Nvidia VFX"

    def __init__(self):
        super().__init__(False)

        self.scalers = [
            # Standard Upscaling
            UpscalerData("[Nvidia] Speed", "LOW", self, 2),
            UpscalerData("[Nvidia] Balanced", "MEDIUM", self, 2),
            UpscalerData("[Nvidia] Quality", "HIGH", self, 2),
            UpscalerData("[Nvidia] Ultra", "ULTRA", self, 4),
            # Denoise
            UpscalerData("[Nvidia] Denoise Light", "DENOISE_LOW", self, 1),
            UpscalerData("[Nvidia] Denoise Moderate", "DENOISE_MEDIUM", self, 1),
            UpscalerData("[Nvidia] Denoise Aggressive", "DENOISE_HIGH", self, 1),
            UpscalerData("[Nvidia] Denoise Maximum", "DENOISE_ULTRA", self, 1),
            # Deblur
            UpscalerData("[Nvidia] Deblur Light", "DEBLUR_LOW", self, 1),
            UpscalerData("[Nvidia] Deblur Moderate", "DEBLUR_MEDIUM", self, 1),
            UpscalerData("[Nvidia] Deblur Aggressive", "DEBLUR_HIGH", self, 1),
            UpscalerData("[Nvidia] Deblur Maximum", "DEBLUR_ULTRA", self, 1),
            # High-Bitrate
            UpscalerData("[Nvidia] HD Speed", "HIGHBITRATE_LOW", self, 2),
            UpscalerData("[Nvidia] HD Balanced", "HIGHBITRATE_MEDIUM", self, 2),
            UpscalerData("[Nvidia] HD Quality", "HIGHBITRATE_HIGH", self, 2),
            UpscalerData("[Nvidia] HD Ultra", "HIGHBITRATE_ULTRA", self, 4),
        ]

    def upscale(self, img: Image.Image, scale: float, selected_model: str):
        image: Image.Image = img.convert("RGB")
        quality = getattr(nvvfx.effects.QualityLevel, selected_model)

        if selected_model.startswith(("DENOISE", "DEBLUR")):
            if scale != 1.0:
                scale = 1.0
                print("[Warning] Denoise/Deblur requires Scale = 1.0")
        else:
            if scale % 0.25 != 0.0:
                scale = round(scale / 0.25) * 0.25
                print(f"[Warning] Scale must be divisible by 0.25 (changed to {scale})")

        orig_w, orig_h = image.size
        target_w, target_h = int(orig_w * scale), int(orig_h * scale)

        device = devices.device
        assert device.type == "cuda"
        devices.torch_gc()

        ndarray: np.ndarray = np.asarray(image, dtype=np.uint8)
        tensor: torch.Tensor = torch.from_numpy(ndarray)
        tensor = tensor.permute(2, 0, 1).to(device=device, dtype=torch.float32)
        tensor = torch.clamp(tensor.div_(255.0), 0.0, 1.0).contiguous()

        vsr = nvvfx.VideoSuperRes(quality=quality)
        vsr.output_width = target_w
        vsr.output_height = target_h
        vsr.load()

        result = vsr.run(tensor)
        output: torch.Tensor = torch.from_dlpack(result.image).detach().clone()
        vsr.close()

        tensor = output.clamp(0.0, 1.0).permute(1, 2, 0)
        image = (tensor.cpu().numpy() * 255.0).astype("uint8")

        return Image.fromarray(image)

    def do_upscale(self, *args, **kwargs):
        raise SyntaxError

    def load_model(self, *args, **kwargs):
        raise NotImplementedError

    def find_models(self, *args, **kwargs):
        raise NotImplementedError


orig = modelloader.load_upscalers


@wraps(orig)
def extra_upscalers():
    orig()
    shared.sd_upscalers.extend(UpscalerNvidia().scalers)


modelloader.load_upscalers = extra_upscalers


def revert():
    modelloader.load_upscalers = orig


on_script_unloaded(revert)
