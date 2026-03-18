# SD Forge Nvidia VFX
This is an Extension for [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo), which implements **VideoSuperRes** from [nvidia-vfx](https://pypi.org/project/nvidia-vfx/) to achieve blazingly-fast high-quality **image enhancement**

> Also supports [Forge Classic](https://github.com/Haoming02/sd-webui-forge-classic/tree/classic), [Automatic1111 Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui), etc.

> [!Important]
> Nvidia GPU is Required

<table>
  <tr>
    <th>256x256</th>
    <th>1024x1024</th>
  </tr>
  <tr>
    <td><img width="512" src="input.jpg" /></td>
    <td><img width="512" src="output.jpg" /></td>
  </tr>
</table>

> **Ultra** option took `0.7s` on a **RTX 3060**

## Features

> [!Tip]
> `Scale by` mode is recommended

- **Upscale:** Super-resolution an image by the desired ratio

> Enable the following options in the **Settings**

- **Deblur:** Sharpen an image
- **Denoise:** Reduce noises from an image
- **HD:** Upscale high quality images with little compression artifacts

> [!Important]
> - For **Upscale**, the `Resize` must be divisible by `0.25`
> - For **Deblur** / **Denoise**, the `Resize` must be `1.0`
