import gradio as gr
import numpy as np
import nvvfx
import torch
from PIL import Image

from backend import memory_management
from modules import scripts_postprocessing
from modules.ui_components import InputAccordion

assert memory_management.is_nvidia(), "Only Nvidia GPU is supported..."


class ForgeRTX(scripts_postprocessing.ScriptPostprocessing):
    name = "Nvidia VFX"
    order = 1024

    def ui(self):
        with InputAccordion(value=False, label="Nvidia VFX") as enable:
            with gr.Row():
                quality = gr.Dropdown(
                    choices=("Low", "Medium", "High", "Ultra"),
                    value="High",
                    label="Quality",
                )
                scale = gr.Radio(
                    choices=("x2", "x3", "x4"),
                    value="x2",
                    label="Scale by",
                )

        return {"enable": enable, "quality": quality, "scale": scale}

    def process(self, pp: scripts_postprocessing.PostprocessedImage, **args):
        if not args.pop("enable", False):
            return

        _quality: str = args.get("quality", "High")
        _scale: str = args.get("scale", "x2")

        image: Image.Image = pp.image.convert("RGB")
        quality = getattr(nvvfx.effects.QualityLevel, _quality.upper())
        scale = int(_scale[-1])

        orig_w, orig_h = image.size
        target_w, target_h = orig_w * scale, orig_h * scale

        device = memory_management.get_torch_device()
        memory_management.soft_empty_cache()

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

        memory_management.soft_empty_cache()

        pp.image = Image.fromarray(image)
        pp.info["NvidiaVFX"] = args.copy()
