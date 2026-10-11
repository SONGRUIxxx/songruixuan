"""RAW 图像转 JPEG 工具 — rawpy 完整解码 + 内嵌预览回退"""

import os
import tempfile
from pathlib import Path


def raw_to_jpeg(fpath: str, prefix: str = "_srx_raw") -> str | None:
    """把 RAW（DNG/CR2/NEF/ARW）转成 JPEG 临时文件，供视觉模型读取。

    优先 rawpy 完整解码；失败则回退到内嵌预览/缩略图（很多有损压缩 DNG
    无法完整解码，但内嵌了 JPEG 预览图）；再失败尝试 PIL 直接读（DNG 是
    TIFF 容器，部分可直接打开）。返回 JPEG 路径，失败返回 None。
    """
    from PIL import Image

    tmp_path = os.path.join(tempfile.gettempdir(), f"{prefix}_{Path(fpath).stem}.jpg")

    # 1) rawpy 完整解码
    try:
        import rawpy
        with rawpy.imread(fpath) as raw:
            rgb = raw.postprocess()
        Image.fromarray(rgb).save(tmp_path, "JPEG", quality=85)
        return tmp_path
    except Exception:
        pass

    # 2) 内嵌预览/缩略图（有损压缩 DNG 等）
    try:
        import rawpy
        with rawpy.imread(fpath) as raw:
            thumb = raw.extract_thumb()
        if thumb is not None:
            if thumb.format == rawpy.ThumbFormat.JPEG:
                with open(tmp_path, "wb") as f:
                    f.write(thumb.data)
                return tmp_path
            if thumb.format == rawpy.ThumbFormat.BITMAP:
                Image.fromarray(thumb.data).save(tmp_path, "JPEG", quality=85)
                return tmp_path
    except Exception:
        pass

    # 3) PIL 直接读（DNG 是 TIFF 容器，部分可直接打开）
    try:
        im = Image.open(fpath)
        im.load()
        im.convert("RGB").save(tmp_path, "JPEG", quality=85)
        return tmp_path
    except Exception:
        pass

    return None
