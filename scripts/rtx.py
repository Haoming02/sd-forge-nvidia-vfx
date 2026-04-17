from functools import wraps

import numpy as np
import nvvfx
import torch
from PIL import Image

from modules import devices, modelloader, shared
from modules.script_callbacks import on_script_unloaded, on_ui_settings
from modules.shared import OptionInfo, opts
from modules.upscaler import Upscaler, UpscalerData


class UpscalerNvidia(Upscaler):
    name = "Nvidia VFX"

    _cache_config: tuple[int] = None
    _cache_model: nvvfx.VideoSuperRes = None

    def __init__(self):
        super().__init__(False)
        self.scalers = []

        if "Upscale" in getattr(opts, "nvvfx_options", ["Upscale"]):
            self.scalers += [
                UpscalerData("[Nvidia] Speed", "LOW", self, 2),
                UpscalerData("[Nvidia] Balanced", "MEDIUM", self, 2),
                UpscalerData("[Nvidia] Quality", "HIGH", self, 2),
                UpscalerData("[Nvidia] Ultra", "ULTRA", self, 4),
            ]
        if "Deblur" in getattr(opts, "nvvfx_options", ["Upscale"]):
            self.scalers += [
                UpscalerData("[Nvidia] Deblur Light", "DEBLUR_LOW", self, 1),
                UpscalerData("[Nvidia] Deblur Moderate", "DEBLUR_MEDIUM", self, 1),
                UpscalerData("[Nvidia] Deblur Aggressive", "DEBLUR_HIGH", self, 1),
                UpscalerData("[Nvidia] Deblur Maximum", "DEBLUR_ULTRA", self, 1),
            ]
        if "Denoise" in getattr(opts, "nvvfx_options", ["Upscale"]):
            self.scalers += [
                UpscalerData("[Nvidia] Denoise Light", "DENOISE_LOW", self, 1),
                UpscalerData("[Nvidia] Denoise Moderate", "DENOISE_MEDIUM", self, 1),
                UpscalerData("[Nvidia] Denoise Aggressive", "DENOISE_HIGH", self, 1),
                UpscalerData("[Nvidia] Denoise Maximum", "DENOISE_ULTRA", self, 1),
            ]
        if "HD" in getattr(opts, "nvvfx_options", ["Upscale"]):
            self.scalers += [
                UpscalerData("[Nvidia] HD Speed", "HIGHBITRATE_LOW", self, 2),
                UpscalerData("[Nvidia] HD Balanced", "HIGHBITRATE_MEDIUM", self, 2),
                UpscalerData("[Nvidia] HD Quality", "HIGHBITRATE_HIGH", self, 2),
                UpscalerData("[Nvidia] HD Ultra", "HIGHBITRATE_ULTRA", self, 4),
            ]

    @classmethod
    def get_model(cls, quality: int, width: int, height: int):
        config = (quality, width, height)
        if config == cls._cache_config:
            return cls._cache_model

        if cls._cache_model is not None:
            cls._cache_model.close()

        vsr = nvvfx.VideoSuperRes(quality=quality)
        vsr.output_width = width
        vsr.output_height = height
        vsr.load()

        cls._cache_model = vsr
        return vsr

    @torch.inference_mode()
    def upscale(self, img: Image.Image, scale: float, selected_model: str):
        image: Image.Image = img.convert("RGB")
        quality = getattr(nvvfx.effects.QualityLevel, selected_model)

        if selected_model.startswith(("DENOISE", "DEBLUR")):
            if scale != 1.0:
                scale = 1.0
                print("[Warning] Scale must be 1.0 for Denoise/Deblur")
        else:
            if scale % 0.25 != 0.0:
                scale = round(scale / 0.25) * 0.25
                print(f"[Warning] Scale must be divisible by 0.25 (changed to {scale})")

        orig_w, orig_h = image.size

        pad_w: int = (8 - (orig_w % 8)) % 8
        pad_h: int = (8 - (orig_h % 8)) % 8

        if pad_w > 0 or pad_h > 0:
            from PIL import ImageOps

            image = ImageOps.expand(image, (0, 0, pad_w, pad_h), fill=0)

        aligned_w, aligned_h = image.size
        target_w, target_h = int(aligned_w * scale), int(aligned_h * scale)

        device = devices.device
        assert device.type == "cuda"
        devices.torch_gc()

        ndarray: np.ndarray = np.asarray(image, dtype=np.uint8)
        tensor: torch.Tensor = torch.from_numpy(ndarray)
        tensor = tensor.permute(2, 0, 1).to(device=device, dtype=torch.float32)
        tensor = torch.clamp(tensor.div_(255.0), 0.0, 1.0).contiguous()

        vsr = self.get_model(quality, target_w, target_h)
        result = vsr.run(tensor)
        output: torch.Tensor = torch.from_dlpack(result.image)

        tensor = output.detach().clone().permute(1, 2, 0).contiguous()
        image_array = tensor.mul_(255.0).clamp_(0.0, 255.0)
        image_numpy = image_array.cpu().numpy().astype("uint8")

        final_w, final_h = int(orig_w * scale), int(orig_h * scale)
        result_img = Image.fromarray(image_numpy).crop((0, 0, final_w, final_h))

        return result_img

    def do_upscale(self, *args, **kwargs):
        raise SyntaxError

    def load_model(self, *args, **kwargs):
        raise NotImplementedError

    def find_models(self, *args, **kwargs):
        raise NotImplementedError


def on_settings():
    from gradio import CheckboxGroup

    args = {"section": ("nvvfx", "Nvidia VFX"), "category_id": "postprocessing"}

    opts.add_option(
        "nvvfx_options",
        OptionInfo(
            ["Upscale"],
            "Visible Options",
            CheckboxGroup,
            lambda: {"choices": ("Upscale", "Deblur", "Denoise", "HD")},
            **args,
        ).needs_reload_ui(),
    )


orig = modelloader.load_upscalers


@wraps(orig)
def extra_upscalers():
    orig()
    shared.sd_upscalers.extend(UpscalerNvidia().scalers)


modelloader.load_upscalers = extra_upscalers


def revert():
    modelloader.load_upscalers = orig


on_ui_settings(on_settings)
on_script_unloaded(revert)
