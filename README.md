# SD Forge Nvidia VFX
This is an Extension for [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo), which implements **VideoSuperRes** from [nvidia-vfx](https://pypi.org/project/nvidia-vfx/) to achieve blazingly-fast high-quality **image enhancement**

> Also supports [Forge Classic](https://github.com/Haoming02/sd-webui-forge-classic/tree/classic), [Automatic1111 Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui), etc.

> [!Important]
> Nvidia GPU is Required

<table>
  <caption>
    <b>Ultra</b> took <b>1.2s</b> on a <b>RTX 3060</b>
  </caption>
  <tr>
    <th>256x256</th>
    <th>1024x1024</th>
  </tr>
  <tr>
    <td><img width="512" src="input.jpg" /></td>
    <td><img width="512" src="output.jpg" /></td>
  </tr>
</table>

## Features
> Select the preset from the `Upscaler` dropdown in **Extras**

> [!Tip]
> `Scale by` mode is recommended

#### Upscale

- Super-resolution an image by the desired amount
- `Resize` must be divisible by `0.25`
- The **HD** variants are meant for high quality images with no compression artifacts

#### Deblur

- Sharpen an image
- `Resize` must be `1.0`

#### Denoise

- Remove noises from an image
- `Resize` must be `1.0`

## TODO
- [ ] Support Video
