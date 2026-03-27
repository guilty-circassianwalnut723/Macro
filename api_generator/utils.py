from PIL import Image
import io
import os
import base64

def img2base64(img, format="JPEG"):
    # Local image path
    assert isinstance(img, str) or isinstance(img, Image.Image), "img2base64 only supports str or Image format"
    if isinstance(img, str):
        if not os.path.exists(img):
            return Exception(f"File {img} does not exist")
        # Prevent failure on open
        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(img)
    # TODO: currently force-resize so that the longer side does not exceed 1024 pixels
    width, height = img.size
    if width > 1024 or height > 1024:
        if width > height:
            new_width = 1024
            new_height = int(height * 1024 / width)
        else:
            new_height = 1024
            new_width = int(width * 1024 / height)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    img_byte_arr = io.BytesIO()
    if img.mode == "RGBA":
        # Convert image to RGB
        img = img.convert("RGB")
    img.save(img_byte_arr, format=format)
    return base64.b64encode(img_byte_arr.getvalue()).decode()

def base642img(base64_str):
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    return img
