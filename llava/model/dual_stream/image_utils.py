"""Image preprocessing -- the same expand2square used at training time.

`CLIPImageProcessor` defaults to `do_center_crop=True`, which crops the long
side down to 336 and throws content away. Training pads the image to a square
first (short side filled with LLaVA's standard background colour), so the
processor's resize is isotropic. Evaluation must do the same, otherwise the
model sees a different image distribution than it was trained on.
"""

from PIL import Image


def expand2square(pil_img: Image.Image, background_color=(122, 116, 104)) -> Image.Image:
    """Pad a PIL image into a square by filling the short side. The colour
    (122, 116, 104) is LLaVA's standard background."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result
