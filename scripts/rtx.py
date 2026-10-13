import math
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

    def __init__(self):
        super().__init__(False)
        self.device = devices.device
        assert self.device.type == "cuda"
        self.stream = torch.cuda.Stream(device=self.device)

        self.scalers = []

        if "Upscale" in getattr(opts, "nvvfx_options", ["Upscale"]):
            self.scalers += [
                UpscalerData("[Nvidia] Speed", "LOW", self, 2),
                UpscalerData("[Nvidia] Balanced", "MEDIUM", self, 2),
                UpscalerData("[Nvidia] Quality", "HIGH", self, 2),
                UpscalerData("[Nvidia] Ultra", "ULTRA", self, 2),
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
                UpscalerData("[Nvidia] HD Ultra", "HIGHBITRATE_ULTRA", self, 2),
            ]

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

        padded_w: int = math.ceil(orig_w / 32) * 32
        padded_h: int = math.ceil(orig_h / 32) * 32

        pad_w: int = padded_w - orig_w
        pad_h: int = padded_h - orig_h

        final_w: int = math.floor(orig_w * scale)
        final_h: int = math.floor(orig_h * scale)

        target_w: int = round(padded_w * scale)
        target_h: int = round(padded_h * scale)
        assert target_w % 8 == 0 and target_h % 8 == 0

        ndarray = np.asarray(image, dtype=np.float32)
        if pad_w > 0 or pad_h > 0:
            ndarray = np.pad(ndarray, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

        devices.torch_gc()

        tensor = (
            torch.from_numpy(ndarray)
            .permute(2, 0, 1)
            .div_(255.0)
            .clamp_(0.0, 1.0)
            .contiguous()
            .to(self.device)
        )

        with nvvfx.VideoSuperRes(quality) as sr:
            sr.output_width = target_w
            sr.output_height = target_h
            sr.load()

            dlpack_out = sr.run(tensor, stream_ptr=self.stream.cuda_stream).image
            output = torch.from_dlpack(dlpack_out).detach().clone()

        tensor = output.movedim(0, -1).mul_(255.0).round_().clamp_(0.0, 255.0)
        image_numpy = tensor.cpu().numpy().astype("uint8")
        result_img = Image.fromarray(image_numpy)

        return result_img.crop((0, 0, final_w, final_h))

    def do_upscale(self, *args, **kwargs):
        raise NotImplementedError

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
