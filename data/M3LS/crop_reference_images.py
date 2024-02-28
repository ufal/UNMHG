import sys
from PIL import Image

img_path = sys.argv[1]

_img = Image.open(img_path).convert('RGB')
_img.crop([0, 0, 1024, 490]).save(img_path.replace("_RAW_", "").replace(".png", ".jpg"))

