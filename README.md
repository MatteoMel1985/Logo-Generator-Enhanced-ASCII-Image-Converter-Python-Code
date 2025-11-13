# Logo-Generator-Enhanced-ASCII-Image-Converter-Python-Code

The following is a thorough explanation of a Python code I wrote, whose function is to convert an input image to ASCII art and exporting three synchronized outputs: plain-text (`.txt`), HTML (`.html`), and a rendered bitmap PNG (`.png`). After rendering the PNG from text, it post-corrects the final image’s aspect ratio to exactly match the original photo/graphic.

# ***Imports: what each module contributes***  

```Python
from pathlib import Path
from datetime import datetime
from typing import Sequence, Tuple, Optional
import re
import numpy as np
import requests
from PIL import Image, ImageOps, ImageFile, ImageFont, ImageDraw
from io import BytesIO
from IPython.display import HTML, display
```

* `pathlib.Path` is a standard Python library to handle file system paths (files and directories) in an object-oriented and cross-platform manner. Instead of manipulating paths as plain strings, such as `"C:\\Users\\..."` or `"/home/user/file.txt"`, `pathlib.Path` gives you a Path object that automatically handles:
    
    * joining paths;
    * checking if files exist;
    * creating directories;
    * extracting file names or extensions;
    * resolving absolute paths;
 
* `datetime.datetime` is used for a timestamp in the output filenames.
* `typing` is a type-hinting library introduced in Python 3.5, used for static type checking and improved code readability.

| Type Hint | Meaning |
| --------- | ------- |
| `Sequence[T]` | Any ordered collection (list, tuple, etc.) of elements of type `T`. | 
| `Tuple[T1, T2, ...]` | A fixed-length tuple of elements of given types. | 
| `Optional[T]` | A value that can be either `T` or `None`. | 

* `re` is Python’s Regular Expression library. In the code is used to remove query strings (`?raw=true`) and hash fragments (`#something`) from URLs to extract a clean filename.

* `numpy` is the fundamental numerical computing library in Python. It provides the `ndarray` (N-dimensional array) structure, allowing fast vectorised math using C-optimised operations.

* `requests` is the de-facto standard library for HTTP communication in Python. It simplifies downloading data from URLs with functions like `requests.get(url)`.

* `PIL`, whic is the modern forked name for `PIL` (Python Imaging Library) is the main image-processing library for Python. It provides tool to:

| Submodule | Role in the Code |
| --------- | ---------------- |
| `Image` | Core image object. Used for opening (`Image.open()`), saving, converting to grayscale (`convert("RGB")`), resizing (`resize()`), and creating new canvases (`Image.new()`). | 
| `ImageOps` | High-level operations: here it’s used for `ImageOps.grayscale(img)` to quickly make an image monochrome. | 
| `ImageFile` | Controls how PIL loads files; setting `ImageFile.LOAD_TRUNCATED_IMAGES = True` tells it not to crash if the image stream is incomplete. | 
| `ImageFont` | Loads font files (`.ttf`, `.ttc`) to draw text with custom style and size. Used to draw ASCII characters on the PNG. | 
| `ImageDraw` | Provides a drawing interface on an `Image` object. Used to draw the ASCII text grid onto a blank canvas. Example: `draw.text((x,y), "ABC", font=font, fill=fg_color)`. | 

* `BytesIO` is a built-in class that creates a memory-resident binary stream, which is an in-RAM “file” made of bytes. When downloading an image via requests, you receive bytes in memory (`r.content`). `BytesIO(r.content)` wraps those bytes in a file-like interface that PIL’s `Image.open()` can understand, without saving to disk.

* `IPython.display` is part of the IPython/Jupyter Notebook ecosystem, and allows you to programmatically display rich media outputs (HTML, images, audio, etc.) inside notebook cells.

# ***Global Switches & Output Directory***  

```Python
ImageFile.LOAD_TRUNCATED_IMAGES = True
OUT_DIR = Path("./ascii_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

* `ImageFile.LOAD_TRUNCATED_IMAGES = True` allows PIL to open slightly incomplete images rather than raising an error. It is useful for flaky network fetches or partially saved files.

* `OUT_DIR` defines the export folder.

* `mkdir(..., exist_ok=True)` creates it (and parent folders) if missing; safe to call repeatedly.

# ***Character Density Scales (Dark and  light)***

```Python
# DENSITY: darkest -> lightest
DENSITY_DEFAULT = "@%#*+=-:. "
DENSITY_HEAVY   = "MW@#%8&$*+=-:. "  # inkier glyphs
```

These strings are ordered from visually densest (covers more pixels) to lightest (space). They become the palette that pixels map to.  

# ***Utility: Convert GitHub “Blob” URLs to Raw***  

```Python
def _github_blob_to_raw(url: str) -> str:
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com/", "raw.githubusercontent.com/").replace("/blob/", "/")
    return url
```

Since my image was stored on GitHub, this function is particularly useful. If the input is a GitHub “blob” page link (the kind you see in the browser), this function rewrites it into a direct raw-file URL so `requests.get(...)` downloads the actual image bytes instead of an HTML page.  

* Function Header (`def _github_blob_to_raw(url: str) -> str:`): it defines a function named _github_blob_to_raw that takes a string argument called url and returns a string value.  

* Conditional Guard (`if "github.com" in url and "/blob/" in url:`): Only runs the rewrite if the URL includes both `github.com` and `/blob/` (for example, it is a GitHub file preview page).

* Rewrite Line (`url = url.replace("github.com/", "raw.githubusercontent.com/").replace("/blob/", "/")`): First, it changes the host from `github.com/` to `raw.githubusercontent.com/` (the CDN that serves file bytes), then, it removes the `/blob/` path segment so the path matches the raw CDN format.

* Return Statement(`return url`): Gives back the modified URL, and if it wasn’t a blob link, it returns the original unchanged.

# ***Load an Image from URL or Local Path***

```Python
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
```

This function takes either a URL or a local file path, loads the image using Pillow (always as RGB), and returns both the image object and a Path representing its filename. 

### ***Function Header***

```Python
def _load_image(image_path: str) -> Tuple[Image.Image, Path]:
```

* `_load_image`: private helper that encapsulates “get me a PIL image from any path/URL”.
* `image_path: str`: input is a string, either a URL or filesystem path.
* `-> Tuple[Image.Image, Path]`: returns a 2-tuple:
    * a `PIL.Image.Image` object (img),
    * a `pathlib.Path` object (`Path`) for the file name/base.

### ***Branch: URL VS Local File***

```Python
if image_path.startswith("http"):
    ...
else:
    ...
```

* If the string starts with `"http"` (so `"http://"` or `"https://"`) it’s treated as a web URL.
* Anything else is treated as a local path (like `"./logo.png"` or `"C:\\images\\photo.jpg"`).

### ***URL Branch – Remote Image Loading***  

### ***Normalise GitHub Blob URLs***

```Python
url = _github_blob_to_raw(image_path)
```

* If the URL is a GitHub “blob” page (`https://github.com/.../blob/...`), this converts it into a raw file URL so we get the actual image bytes.
* If not GitHub (or not a blob), it just returns the original URL unchanged.

### ***Download via HTTP***

```Python
r = requests.get(url, timeout=60)
r.raise_for_status()
```

* `requests.get(url, timeout=60)`:

    *  Contacts the server and fetches the resource.
    *  `timeout=60` means if the server doesn’t respond within 60 seconds, raise a timeout error.
 
*  `r.raise_for_status()`:

    *  If the HTTP status is not 2xx (e.g. 404, 500), this raises an exception (example: `HTTPError`).
    *  This prevents us from mistakenly trying to open an error page as an image.
 
### ***Convert response bytes to a PIL image***  

```Python
img = Image.open(BytesIO(r.content)).convert("RGB")
```

* `r.content`: the raw bytes of the HTTP response (the image file data).
* `BytesIO(r.content)`: wraps those bytes in a file-like object in memory, which PIL understands.
* `Image.open(...)`: asks Pillow to interpret those bytes as an image (JPEG, PNG, etc.).
* `.convert("RGB")`: converts the image to RGB colour space:

    * Ensures a consistent 3-channel format regardless of original mode (L, RGBA, CMYK, etc.).
    * Simplifies downstream handling (no alpha channel or palette surprises).
 
### ***Extract a nice filename from the URL***  

```Python
fname = Path(re.sub(r"[?#].*$", "", url)).name or "image"
```

* `re.sub(r"[?#].*$", "", url)`:

  * Regex pattern: `[?#].*$`
    * `[?#]`: match either `?` or `#`.
    * `.*` everything after it.
    * `$` to the end of the string.
   
* This removes query parameters and hash fragments, like. `?raw=true` or `#something` (for example, `"https://example.com/img/photo.png?raw=true#foo"` becomes `"https://example.com/img/photo.png"`).

* `Path(...).name:`
    
    * `Path("https://example.com/img/photo.png").name` → `"photo.png"`.
    * `.name` gives the last component of the path.

* `or "image"`:  

    * If `.name` somehow ends up empty (very unusual, but safe), fall back to `"image"`.
 
So `fname` becomes something like `"With-Snake-Shading.jpg"`.  

### ***Return Image + Path***  

```Python
return img, Path(fname)
```

* Wraps the filename string in a `Path` object (e.g. `Path("With-Snake-Shading.jpg")`).
* Caller now knows:
    
    * the image pixels (`img`),
    * the base name to use in output filenames (`Path(fname).stem`, etc.).
 
### ***Local Branch - Filesystem Image Loading***  

```Python
else:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    img = Image.open(p).convert("RGB")
    return img, p
```

* `p = Path(image_path)` Convert string to Path
* `if not p.exists():
    raise FileNotFoundError(f"Image not found: {p}")`:

    * `p.exists()` checks that the file is present at that path.
    * If not, raise FileNotFoundError with a clear message.
  
### ***Open with PIL and Normalise to RGB***

```Python
img = Image.open(p).convert("RGB")
```

* `Image.open(p)` reads from disk and builds a PIL image object.
* `.convert("RGB")` same as in the URL branch: standardise to RGB.

### ***Return Image + Path***  

```Python
return img, p
```

The title is self-explanatory.

# ***Choose a Monospaced Font (With Fallbacks)***  

```Python
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
```

The function tries to load a monospaced font (preferably bold) from a small list of common cross-platform font files; it returns the first one that loads successfully, and if none can be loaded, it falls back to Pillow’s built-in default font.  

### ***Function Signature***  

* `def _pick_font(font_size: int, bold: bool = True) -> ImageFont.FreeTypeFont`:

    * `_pick_font`: private helper (leading underscore) to select a font.
    * `font_size: int`: requested point size.
    * `bold: bool = True`: indicates a preference for a bold face.
    * Return type: a Pillow `ImageFont.FreeTypeFont` object.
 
### ***Candidate Font list (Platform-Aware Order)***  

```Python
for path in [
    "DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "C:\\Windows\\Fonts\\consola.ttf",
]:
```

* A small search list with typical locations/names for monospaced fonts:

    * Linux (generic relative): `DejaVuSansMono-Bold.ttf` (current working dir)
    * Linux (system): `/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf`
    * macOS: `/System/Library/Fonts/Menlo.ttc` (Menlo in a font collection)
    * Windows: `C:\Windows\Fonts\consola.ttf `(Consolas; Windows is case-insensitive so consola.ttf is fine)
  
### ***Try to Load each Font; on Success Return Immediately***

```Python
try:
    return ImageFont.truetype(path, font_size)
except Exception:
    continue
```  

* `ImageFont.truetype(path, font_size)` loads a scalable font via FreeType.
* If the file isn’t found, is unreadable, or can’t be parsed, Pillow raises (commonly `OSError` / `IOError`), and the next path is tried.
* On first success, we return (short-circuit).

### ***Fallback to Pillow’s Default Bitmap Font***  

```Python
return ImageFont.load_default()
```

* If no candidate loads, we still return a usable font (small bitmap).
* Ensures the rest of the pipeline (PNG rendering) doesn’t crash.

Monospaced fonts guarantee equal character cell width, crucial for ASCII grids to align perfectly, whereas bold faces produce inkier strokes, improving readability of dense ASCII art (especially at smaller sizes or with stroke outlines).  

# ***First-Pass Resize to an ASCII Grid Size***  

```Python
def _normalize_image_for_ascii(img: Image.Image, cols: int, char_aspect_guess: float = 0.5) -> Image.Image:
    """
    Quick grayscale resize to get an ASCII grid size that’s in the ballpark.
    We'll fix any residual aspect mismatch *after* rendering by resizing the PNG.
    """
    gray = ImageOps.grayscale(img)
    w, h = gray.size
    rows = max(1, int(round((h / w) * cols * char_aspect_guess)))
    return gray.resize((cols, rows), resample=Image.BICUBIC)
```

`_normalize_image_for_ascii()` performs the first stage of the ASCII conversion pipeline: it takes the original image and shrinks it down to an ASCII-friendly resolution, where:  

* the width (in characters) is fixed by the user (`cols`)
* the height (in characters, called `rows`) is estimated using the original aspect ratio and the “stretchiness” of ASCII characters
* the image is converted to grayscale, because ASCII art does not need color—only brightness
* the resized grayscale image is returned as a small grid of pixels, later mapped to characters

* `def _normalize_image_for_ascii(img: Image.Image, cols: int, char_aspect_guess: float = 0.5) -> Image.Image:` defines the function and its inputs:

    * `img: Image.Image` is a PIL `Image` object.
    * `cols: int` defines how many ASCII characters wide you want the output to be. his number comes from the user.
    * `char_aspect_guess: float = 0.5` is a fudge factor that compensates for the fact that ASCII characters are rectangular, not square. Typical monospace characters are ~2× taller than they are wide. A value of `0.5` compresses the computed height accordingly.
    * RETURN TYPE: `Image.Image` The function returns a grayscale, downscaled PIL image whose pixel grid corresponds to ASCII positions.
    
* `gray = ImageOps.grayscale(img)`: since ASCII art only needs brightness information, and colour would be wasted and would make mapping more complicated, this processing lines:

    * Converts the input image to 8-bit grayscale (`mode = "L"`).
    * Every pixel value is now a number between 0 (black) and 255 (white).
    * The ASCII conversion later maps brightness → character.
 
* `w, h = gray.size`: `w` is the width of the grayscale image in pixels, whereas `h` is the height of the grayscale image in pixels. This gives the original aspect ratio `aspect` = `h` / `w`. This ratio is needed to compute how many rows (height) the ASCII grid should have.

* `rows = max(1, int(round((h / w) * cols * char_aspect_guess)))` It calculates how many ASCII text rows the output should have.

  * `(h / w)` is the original aspect ratio, the real-world shape of the image. Example: If `h/w` > 1 = portrait, If < 1 = landscape.
  * `cols`: number of ASCII columns. The number of characters wide was already fixed. Now we need to compute the height that preserves the shape.
  * `(h / w) * cols` is the raw height estimate.
  * `* char_aspect_guess` adjusts for non-square glyphs. ASCII characters are taller than wide. A typical monospace cell might be 12 px high × 6 px wide. Thus:
    
      * If you didn’t compensate → ASCII appears vertically stretched.
      * Multiplying by 0.5 shrinks the rows accordingly.
      * 0.5 is only an approximation; the final PNG correction fixes the remainder
   
  * `round(...) and int(...)` Ensures we never return zero rows. Even a sliver of an image must be represented by at least 1 ASCII line.

* `return gray.resize((cols, rows), resample=Image.BICUBIC)` returns the normalized brightness-grid image.

# ***Map Frayscale Pixels to Characters***  

```Python
def _map_pixels_to_chars(arr: np.ndarray, charset: Sequence[str], invert_map: bool = True, brightness_boost: float = 1.0) -> np.ndarray:
    arr = np.clip(arr * brightness_boost, 0, 255)
    norm = arr.astype(np.float32) / 255.0 if invert_map else 1.0 - (arr.astype(np.float32) / 255.0)
    idx = (norm * (len(charset) - 1)).round().astype(np.int32)
    return np.take(np.array(list(charset)), idx)
```

The function takes a 2D array of grayscale pixels (numbers 0–255) and turns it into a 2D array of characters from your charset, so that darker pixels become “heavier” glyphs (`M`, `W`, `#`, etc.) and lighter pixels become “lighter” glyphs (`.`, space, etc.). It also optionally:  

* boosts brightness (`brightness_boost`)
* inverts the mapping (`invert_map`) so you can choose whether dark pixels give dense chars or the opposite.

### ***Function Signature***  

```Python
def _map_pixels_to_chars(arr: np.ndarray, charset: Sequence[str], invert_map: bool = True, brightness_boost: float = 1.0) -> np.ndarray:
```

* `arr: np.ndarray` is the grayscale image data, typically a 2D array (height × width), where each element is a pixel intensity from 0 (black) to 255 (white).
* `charset: Sequence[str]` is a sequence of characters ordered from “darkest ink” to “lightest ink”. For example, with `DENSITY_HEAVY = "MW@#%8&$*+=-:. "`, index 0 is the densest (`'M'`), and the last index is the lightest (`' '`).
* `invert_map: bool = True` controls how brightness is mapped:

    * `True` → standard mapping: dark pixels → low values → heavy chars.
    * `False` → inverted mapping: dark pixels → heavy chars if you want to flip the light/dark logic (more on this in the next line).
 
* `brightness_boost: float = 1.0` multiplies pixel values by this factor to globally make the image brighter (or darker if < 1).
* `Return type: np.ndarray` returns a NumPy array of the same shape as `arr`, but with each element replaced by a character from `charset`.

### ***Brightness Boosting and Clamping***  

```Python
arr = np.clip(arr * brightness_boost, 0, 255)
```

This line applies global brightness adjustment, then clips the result to a legal grayscal range

* `arr * brightness_boost` every pixel is multiplied by `brightness_boost`. Example:

    * If a pixel is 100 and `brightness_boost = 1.3`, the result is 130.
    * If a pixel is 200 and `brightness_boost` = 1.3, the result is 260.
 
* `np.clip(..., 0, 255)` ensures values stay in the valid 8-bit grayscale range [0, 255]:

    * Anything < 0 becomes 0.
    * Anything > 255 becomes 255.
 
### ***Normalizing To [0, 1] and Optional Inversion***  

```Python
norm = arr.astype(np.float32) / 255.0 if invert_map else 1.0 - (arr.astype(np.float32) / 255.0)
```

