# ==========================================================
# ASCII Image Converter — Exact Aspect PNG (Green on Black)
# - Renders PNG, then resizes it to match the source aspect exactly
# - Supports thicker glyphs, brighter green
# - Saves .txt, .html, and .png into ./ascii_output/
# ==========================================================

from pathlib import Path
from datetime import datetime
from typing import Sequence, Tuple, Optional
import re
import numpy as np
import requests
from PIL import Image, ImageOps, ImageFile, ImageFont, ImageDraw
from io import BytesIO
from IPython.display import HTML, display

ImageFile.LOAD_TRUNCATED_IMAGES = True
OUT_DIR = Path("./ascii_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# DENSITY: darkest -> lightest
DENSITY_DEFAULT = "@%#*+=-:. "
DENSITY_HEAVY   = "MW@#%8&$*+=-:. "  # inkier glyphs

def _github_blob_to_raw(url: str) -> str:
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com/", "raw.githubusercontent.com/").replace("/blob/", "/")
    return url

def _load_image(image_path: str) -> Tuple[Image.Image, Path]:
    if image_path.startswith("http"):
        url = _github_blob_to_raw(image_path)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        fname = Path(re.sub(r"[?#].*$", "", url)).name or "image"
        return img, Path(fname)
    else:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        img = Image.open(p).convert("RGB")
        return img, p

def _pick_font(font_size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    for path in [
        "DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "C:\\Windows\\Fonts\\consola.ttf",
    ]:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue
    return ImageFont.load_default()

def _normalize_image_for_ascii(img: Image.Image, cols: int, char_aspect_guess: float = 0.5) -> Image.Image:
    """
    Quick grayscale resize to get an ASCII grid size that’s in the ballpark.
    We'll fix any residual aspect mismatch *after* rendering by resizing the PNG.
    """
    gray = ImageOps.grayscale(img)
    w, h = gray.size
    rows = max(1, int(round((h / w) * cols * char_aspect_guess)))
    return gray.resize((cols, rows), resample=Image.BICUBIC)

def _map_pixels_to_chars(arr: np.ndarray, charset: Sequence[str], invert_map: bool = True, brightness_boost: float = 1.0) -> np.ndarray:
    arr = np.clip(arr * brightness_boost, 0, 255)
    norm = arr.astype(np.float32) / 255.0 if invert_map else 1.0 - (arr.astype(np.float32) / 255.0)
    idx = (norm * (len(charset) - 1)).round().astype(np.int32)
    return np.take(np.array(list(charset)), idx)

def _ascii_to_text_lines(char_grid: np.ndarray) -> str:
    return "\n".join("".join(row.tolist()) for row in char_grid)

def _ascii_to_html(char_grid: np.ndarray, fg_color: str, bg_color: str) -> str:
    text_block = _ascii_to_text_lines(char_grid)
    style = (
        f"background:{bg_color};color:{fg_color};font-family:monospace;"
        f"line-height:1.0;white-space:pre;margin:0;padding:12px;font-size:10px;"
    )
    return f'<div style="background:{bg_color};padding:12px;"><pre style="{style}">{text_block}</pre></div>'

def _render_png_from_ascii(
    char_grid: np.ndarray,
    fg_color: str,
    bg_color: str,
    font: ImageFont.FreeTypeFont,
    stroke_width: int = 1,
    stroke_fill: Optional[str] = None,
    line_spacing_px: int = 0,
):
    """Render ASCII grid to a PIL Image (PNG bitmap), without saving yet."""
    if stroke_fill is None:
        stroke_fill = fg_color

    # Measure cell using 'M' for height and font.getlength for width (advance)
    try:
        char_w = max(1, int(round(font.getlength("A"))))
    except Exception:
        bbox = font.getbbox("A")
        char_w = max(1, bbox[2] - bbox[0])
    bbox = font.getbbox("M")
    char_h = max(1, bbox[3] - bbox[1])
    line_h = char_h + max(0, line_spacing_px)

    rows, cols = char_grid.shape
    img = Image.new("RGB", (cols * char_w, rows * line_h), bg_color)
    draw = ImageDraw.Draw(img)
    y = 0
    for r in range(rows):
        draw.text((0, y), "".join(char_grid[r].tolist()), font=font, fill=fg_color,
                  stroke_width=stroke_width, stroke_fill=stroke_fill)
        y += line_h
    return img

def _aspect_correct_png(png_img: Image.Image, src_w: int, src_h: int, preserve: str = "width") -> Image.Image:
    """
    Resize the rendered PNG so its aspect exactly matches the original image.
    preserve: "width" keeps PNG width, adjusts height; "height" keeps height, adjusts width.
    """
    Wp, Hp = png_img.size
    src_aspect = src_w / src_h

    if preserve == "width":
        target_w = Wp
        target_h = max(1, int(round(target_w / src_aspect)))
    else:
        target_h = Hp
        target_w = max(1, int(round(target_h * src_aspect)))

    # NEAREST keeps ASCII pixels crisp
    if (target_w, target_h) != (Wp, Hp):
        png_img = png_img.resize((target_w, target_h), resample=Image.NEAREST)
    return png_img

def convert_image_to_ascii(
    image_path: str,
    out_width_chars: int = 900,
    charset: str = DENSITY_HEAVY,
    fg_color: str = "#66ff66",
    bg_color: str = "#000000",
    brightness_boost: float = 1.3,
    preview_inline: bool = True,
    out_dir: Path = OUT_DIR,
    png_font_size: int = 12,
    png_line_spacing_px: int = 0,
    use_bold_font: bool = True,
    png_stroke_width: int = 1,
    preserve_aspect_by: str = "width",  # "width" or "height"
    char_aspect_guess: float = 0.5,     # rough guess; final PNG is corrected anyway
):
    """
    Convert to ASCII and export TXT/HTML/PNG.
    Final PNG is post-corrected to match the original image aspect exactly.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load and prepare
    img, fname = _load_image(image_path)
    src_w, src_h = img.size
    font = _pick_font(png_font_size, bold=use_bold_font)

    # 2) Build ASCII grid (initial guess rows); we will correct aspect after rendering
    resized = _normalize_image_for_ascii(img, cols=out_width_chars, char_aspect_guess=char_aspect_guess)
    arr = np.array(resized)
    char_grid = _map_pixels_to_chars(arr, charset, invert_map=True, brightness_boost=brightness_boost)

    # 3) Files base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(fname).stem or "ascii_image"
    base = f"{stem}_ASCII_{out_width_chars}w_{stamp}"
    out_txt, out_html, out_png = out_dir / f"{base}.txt", out_dir / f"{base}.html", out_dir / f"{base}.png"

    # 4) Save text + HTML
    ascii_text = _ascii_to_text_lines(char_grid)
    out_txt.write_text(ascii_text, encoding="utf-8")
    html = _ascii_to_html(char_grid, fg_color, bg_color)
    out_html.write_text(html, encoding="utf-8")

    # 5) Render PNG, then aspect-correct to source ratio
    png_img = _render_png_from_ascii(
        char_grid,
        fg_color=fg_color,
        bg_color=bg_color,
        font=font,
        stroke_width=png_stroke_width,
        stroke_fill=fg_color,
        line_spacing_px=png_line_spacing_px,
    )
    png_img = _aspect_correct_png(png_img, src_w, src_h, preserve=preserve_aspect_by)
    png_img.save(out_png, format="PNG")

    # 6) Optional inline preview (HTML preview aspect may vary by browser font)
    if preview_inline:
        display(HTML(f"<p><b>ASCII preview ({out_width_chars} chars wide)</b></p>"))
        display(HTML(html))

    print(f"Saved TXT  -> {out_txt.resolve()}")
    print(f"Saved HTML -> {out_html.resolve()}")
    print(f"Saved PNG  -> {out_png.resolve()}")
    return {"txt_path": str(out_txt), "html_path": str(out_html), "png_path": str(out_png)}

# ==========================================================
# Example Usage (your image)
# ==========================================================
IMAGE_PATH = "https://github.com/MatteoMel1985/Relational-Dataset-Images/blob/main/Demo%20Logo/With-Snake-Shading.jpg?raw=true"

convert_image_to_ascii(
    IMAGE_PATH,
    out_width_chars=900,
    charset=DENSITY_HEAVY,
    fg_color="#66ff66",
    bg_color="#000000",
    brightness_boost=1.3,
    png_font_size=12,
    png_line_spacing_px=0,
    use_bold_font=True,
    png_stroke_width=1,
    preserve_aspect_by="width",  # keep PNG width; adjust height to match source aspect
)
