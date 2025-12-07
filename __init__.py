from .nodes import WanPlusExtension

async def comfy_entrypoint() -> WanPlusExtension:
    return WanPlusExtension()