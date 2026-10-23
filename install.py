import launch

if not launch.is_installed("nvidia-vfx"):
    launch.run_pip(
        "install nvidia-vfx==0.1.0.1 --no-build-isolation --index-url https://pypi.nvidia.com",
        "nvidia-vfx",
    )
