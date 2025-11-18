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

# ***Map Grayscale Pixels to Characters***  

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

It is a conditional operation.

* In both branches, we see `arr.astype(np.float32) / 255.0`:

    * `arr.astype(np.float32)` converts the array to 32-bit floats. This is important because integer division would wreck your normalization (e.g. 100 / 255 as integers would truncate, but as floats it's ≈ 0.392).
    * `/ 255.0` maps pixel values from [0, 255] to [0.0, 1.0]

Then the line is conceptually:

* If `invert_map` is True:

    * 0 (black) → 0.0
    * 255 (white) → 1.0
 
* If `invert_map` is `False`:

    * 0 (black) → 1.0
    * 255 (white) → 0.0
 
### ***Converting Normalized Brightness to Character Indices***  

```Python
idx = (norm * (len(charset) - 1)).round().astype(np.int32)
```

This line maps the continuous brightness [0, 1] to discrete indices into `charset`.  

* `len(charset) - 1` If charset has N characters, valid indices go from 0 to N-1. So multiplying norm (0–1) by (N-1) scales it to the range [0, N-1].
* `norm * (len(charset) - 1)` For example, if `len(charset) = 10` and:

    * `norm = 0.0` → 0.0
    * `norm = 0.5` → 4.5
    * `norm = 1.0` → 9.0
 
 * `.round()` Rounds to the nearest integer index. so:
 
     * 4.4 → 4
     * 4.5 → 4 or 5 depending on NumPy’s rounding rules (NumPy uses “banker’s rounding” – .5 to even).
  
### ***Looking Up the Characters***  

```Python
return np.take(np.array(list(charset)), idx)
```

This is where numeric indices become characters.  
1. `list(charset)`

  * If charset is a string like `"MW@#%8&$*+=-:. "`, `list(charset)` becomes:

```Python
["M", "W", "@", "#", "%", "8", "&", "$", "*", "+", "=", "-", ":", ".", " "]
```

  * If charset is already a sequence of strings, `list()` just ensures we’re working with a plain Python list.

2. `np.array(list(charset))`

  * Converts that list into a 1D NumPy array, e.g. `chars_array.shape == (N,)` where `N = len(charset)`.

3. `np.take(chars_array, idx)`

  * `np.take` takes elements along an axis (here the only axis) using idx as indices.
  * Because `idx` is a 2D array (same shape as the image), NumPy “broadcasts” the indexing across that shape.
  * For each position `(i, j)`, it does:

```Python
chars_array[idx[i, j]]
```

  * The output is a 2D array of the same shape as `idx`, but now each entry is a string character instead of a number.

Result:
A 2D NumPy array of characters, `char_grid`, where each position corresponds to one pixel in the original (resized) grayscale image.  

# ***Turn the Char Grid into Plain Text***  

```Python
def _ascii_to_text_lines(char_grid: np.ndarray) -> str:
    return "\n".join("".join(row.tolist()) for row in char_grid)
```

The function `_ascii_to_text_lines` takes as input a 2-dimensional NumPy array of characters (i.e., the ASCII grid), where:  

* Each row of the array is a sequence of characters (e.g. `["@","M","W","."," "]`).
* The full array represents the entire ASCII image.

The function converts this 2D array into a multiline string, where each row of the array becomes a line of text, and all lines are joined using newline characters (`"\n"`).
In other words, it converts a matrix of characters into a readable ASCII-art text block.

* `"def _ascii_to_text_lines(char_grid: np.ndarray) -> str:"`:

    * Defines a function named` _ascii_to_text_lines`.
    * It expects one argument: `char_grid`, which is a NumPy array.
    * The return type is declared as `str`.
 
* `"return ..."` simply indicates that the function returns the final multiline string. There is no intermediate variable; the expression is computed and immediately returned.

* `"\n".join(...)`: It constructs the final ASCII text by placing one newline per row.

    * `"\n"` is the separator.
    * `.join(...)` merges a sequence of strings into one string, putting a newline between each.
 
 * `"".join(row.tolist())` is the inner join, executed once per row:

     * `row` is a 1D slice of the NumPy array (e.g., `['@','W',' ','.']`).
     * `row.tolist()` converts the NumPy row (which is something like `np.ndarray(['@','W',' ','.'])`) into a Python list of characters.
     * `"".join([...])` concatenates all characters in the list into one single string with no separator.
  
* `"for row in char_grid"` is a generator expression. It iterates over every row of the 2D `char_grid`.

# ***Build an HTML Block for Preview/Export***  

```Python
def _ascii_to_html(char_grid: np.ndarray, fg_color: str, bg_color: str) -> str:
    text_block = _ascii_to_text_lines(char_grid)
    style = (
        f"background:{bg_color};color:{fg_color};font-family:monospace;"
        f"line-height:1.0;white-space:pre;margin:0;padding:12px;font-size:10px;"
    )
    return f'<div style="background:{bg_color};padding:12px;"><pre style="{style}">{text_block}</pre></div>'
```

This function turns the ASCII character grid into an HTLM snippet that can be:

* shown inline in Jupyter (via `display(HTML(...))`), and
* saved as a `.html` file and open in a browser

with green text on black, fixed-width font, and preserved spacing.  

Inputs:

* `char_grid`: a 2D NumPy array, shape (rows, cols), where each element is a single-character string (the ASCII art output).
* `fg_color`: foreground (text) colour, (for example `"#66ff66"`).
* `bg_color`: background colour (for example `"#000000"`).

Process: 

1. Converts the 2D grid into a single multi-line string (`text_block`) where:

    * each row becomes one line of text,
    * rows are separated by `\n`.
  
2. Builds a CSS style string that defines:

    * colours, font, line height, padding, etc.
  
3. Wraps `text_block` in HTML:

    * an outer `<div>` (for outer background + padding),
    * an inner `<pre>` (for monospaced display with preserved spacing).
  
* Output:

    * Returns one big HTML string that, when rendered, shows your ASCII art as a styled block in the browser or notebook.
 
# ***Render a PNG by Drawing Rows of Text***

```Python
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
```

This function is the ASCII to PNG conversion's heart of the pipeline: it takes the previously computed character grid, lays those characters out in a regular another one using a monospaced font, and returns a PIL Image that is a bitmap rendering of that ASCII art.

### ***Function Signature and Parameters***

```Python
def _render_png_from_ascii(
    char_grid: np.ndarray,
    fg_color: str,
    bg_color: str,
    font: ImageFont.FreeTypeFont,
    stroke_width: int = 1,
    stroke_fill: Optional[str] = None,
    line_spacing_px: int = 0,
):
```

* `char_grid: np.ndarray`:

    * A 2D NumPy array: each element is a single character (e.g. `"@"`, `"#"`, `" "`).
    * Shape is `(rows, cols)`, which defines how many text rows and columns the ASCII art has.
    * Example: if you requested `out_width_chars=900`, `cols` should be `900`, while rows is computed earlier during the grayscale resize.
 
* `fg_color: str`:

    * Foreground colour for the characters, e.g. `"#66ff66"` (your bright green).
    * Passed directly to Pillow as the `fill` colour when drawing text.
 
* `bg_color: str`:

    *  Background colour for the image, e.g. `"#000000"` for black.
    * Used when creating the base `Image.new("RGB", size, bg_color)` canvas.
 
* `font: ImageFont.FreeTypeFont`:

    * A pre-chosen font object (monospaced, ideally) from `_pick_font`.
    * It contains metrics (glyph sizes, spacing, etc.) and rendering logic.
 
* `stroke_width: int = 1`:

    *  Thickness of the outline around text.
    *  `1` means a subtle outline that thickens the characters, helping them pop and look “inkier”.
 
* `stroke_fill: Optional[str] = None`:

    * Colour of the outline/stroke.
    * If `None`, the function will default it to the same as `fg_color` (for example, the outline is the same colour as the text, making it look thicker rather than “outlined in another colour”).
 
* `line_spacing_px: int = 0`:

    * Extra pixels to add between lines of text.
    * `0` means lines are packed as tightly as the font allows (just `char_h` tall).
    * A positive value would add vertical breathing room between rows.
 
### ***Set Stroke_Fill Default if Needed***

```Python
    if stroke_fill is None:
        stroke_fill = fg_color
```

* If the caller did not specify a stroke colour:
  * Use the same colour as the text.
 
This makes the characters look thicker/bolder without giving them a contrasting outline. Good for “green terminal” aesthetics: glyphs are still pure green, just “fatter”.

### ***Measuring the Character Cell (Width and Height)***  

This is crucial: we need to know how big each character is in pixels to build a correctly sized image and avoid squashing/stretching.

```Python
    # Measure cell using 'M' for height and font.getlength for width (advance)
    try:
        char_w = max(1, int(round(font.getlength("A"))))
    except Exception:
        bbox = font.getbbox("A")
        char_w = max(1, bbox[2] - bbox[0])
```

* **Goal**: compute `char_w`, the horizontal advance of a character, i.e. how many pixels to move right for each character cell.  

* `font.getlength("A")`:

  * Pillow’s method that returns the advance width of the string `"A"` with this font.
  * More accurate for proportional fonts, but we’re assuming monospaced fonts so each character should have nearly the same width.
  * Using `"A"` is arbitrary but safe; it is a typical letter.
 
* `int(round(...))`:

  * Converts the floating-point width to an integer number of pixels.
 
* `max(1, ...)`:

  * Ensure we don’t get zero or negative values (just a defensive guard).
 
* The `except Exception:` block:

  * Some environments or older Pillow versions may not support `font.getlength`.
    * Returns a bounding box: `(left, top, right, bottom)`.
    * Width ≈ `right - left`.
  * Again, wrapped with max(1, ...) to avoid invalid values.
 
So at the end of this block, `char_w` is our per-character cell width in pixels.  

```Python
    bbox = font.getbbox("M")
    char_h = max(1, bbox[3] - bbox[1])
    line_h = char_h + max(0, line_spacing_px)
```

* Now we measure height using `"M"`:

  * `"M"` is often one of the tallest characters in the alphabet.
  * `bbox = font.getbbox("M")` returns `(left, top, right, bottom)` in font space.
  * Height ≈ `bottom - top`, so:
 
```Python
  char_h = max(1, bbox[3] - bbox[1])
```

* `line_h` is the total vertical step per text row:

  * `line_h = char_h + max(0, line_spacing_px)`
  * If `line_spacing_px` is 0, `line_h` equals `char_h`.
  * If `line_spacing_px` is positive, you get taller lines, i.e. more space between rows.
  * Again `max(0, ...)` avoids accidental negative spacing.
 
Now we have:  

  * `char_w`: width of one character cell in pixels.
  * `line_h`: height of one line including extra spacing.

These measurements ensure that text rows don’t overlap and that the image size matches the grid.

### ***Creating the Output Image Canvas***

```Python
    rows, cols = char_grid.shape
    img = Image.new("RGB", (cols * char_w, rows * line_h), bg_color)
    draw = ImageDraw.Draw(img)
```

* `rows, cols = char_grid.shape`:

  * `char_grid.shape` returns `(number_of_rows, number_of_columns)`.
  * Example: if `char_grid` is 200 rows × 900 columns, then:

    * `rows = 200`
    * `cols = 900`
   
  * `Image.new("RGB", (cols * char_w, rows * line_h), bg_color)`:
 
    * Create a new blank image:
    * Mode `"RGB"`: 3-channel colour.
    * Size: Width = `cols * char_w` and Height = `rows * line_h`
    * Filled entirely with `bg_color` — this is the black background.
   
  * `draw = ImageDraw.Draw(img)`:

    *  Creates a drawing context associated with `img`.
    *  You’ll use `draw.text` to paint text onto this image.
   
### ***Looping Through Each Row and Drawing Text***  

```Python
    y = 0
    for r in range(rows):
        draw.text((0, y), "".join(char_grid[r].tolist()), font=font, fill=fg_color,
                  stroke_width=stroke_width, stroke_fill=stroke_fill)
        y += line_h
    return img
```

### **Initial vertical position** 

* `y = 0`:

  * Start drawing text at the top of the image.
 
### **Row-by-Row Rendering** 

* `for r in range(rows):`:

  * Iterate over each row index `r` in the ASCII grid.
 
* Inside the loop:

```Python
"".join(char_grid[r].tolist())
```

* `char_grid[r]` is the `r`-th row, a 1D array of characters like `["@","%","#","*", ...]`.
* `.tolist()` converts this NumPy array to a Python list.
* `"".join(...)` turns that list into a single string, e.g. `"@%#*+=-:."`.
* So we draw the entire row as one text string, rather than one character at a time.
This is much faster and more consistent.

```Python
draw.text(
    (0, y),
    "".join(char_grid[r].tolist()),
    font=font,
    fill=fg_color,
    stroke_width=stroke_width,
    stroke_fill=stroke_fill
)
```

* `draw.text((0, y), ...)`:

  * Draws the string at pixel coordinates `(x=0, y=y)`:

    * `x = 0`: always left-aligned at the left edge.
    * `y`: vertical position for this row, increasing by line_h each iteration.
   
  * `font=font`: use the font passed into the function.
  * `fill=fg_color`: colour of the glyph interior (your neon green).
  * `stroke_width=stroke_width`: thickness of outline.
  * `troke_fill=stroke_fill`: outline colour (same as fg by default here).
 
* `y += line_h`:

  * After drawing one row, move down by `line_h` pixels to prepare for the next row.
  * This keeps rows evenly spaced with no overlap.
 
### **Return the finished image**  

* `return img`:

  * The function returns the completed PIL Image with all ASCII rows drawn.
  * The caller (`convert_image_to_ascii`) can then:
    * Pass `img` to `_aspect_correct_png` to tweak aspect ratio.
    * Save it to disk as PNG.
   
# ***How this Fits in the Whole Pipeline***

Just to connect the dots with the rest of your code:

1. `convert_image_to_ascii`:

   * Builds `char_grid` using:
  
     * `_normalize_image_for_ascii` (resize & grayscale)
     * `_map_pixels_to_chars` (map brightness ➜ characters)
    
2.  Then `_render_png_from_ascii`:

  * Takes `char_grid` and literally draws those characters onto a bitmap using a properly measured grid based on `font` metrics.

3. Finally `_aspect_correct_png`:

  * Takes that bitmap and resizes it so that its aspect ratio equals the original photo (preserving width or height as requested).

So `_render_png_from_ascii` is the pure rendering stage: no aspect correction yet, no saving yet — just a clean, grid-faithful text drawing.

# ***Post-Render Aspect Correction (Exact Match to Source)***  

```Python
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
```

The functiion takes the already-rendered ASCII PNG and resizes it so its aspect ratio exactly matches the original source image, either keeping the PNG’s current width fixed or its height fixed, and adjusting the other dimension accordingly. It uses nearest-neighbour resampling so the blocky ASCII “pixels” stay crisp instead of getting blurred.  

### ***Function Signature and Purpose***  

```Python
def _aspect_correct_png(png_img: Image.Image, src_w: int, src_h: int, preserve: str = "width") -> Image.Image:
```

`png_img: Image.Image`:  

* A Pillow `Image` object: this is your ASCII art PNG that was already rendered with characters.

`src_w: int, src_h: int`:  

* The original source image’s width and height in pixels (the real photo / image you converted to ASCII).

`preserve: str = "width"`:  

* A flag that tells the function what to keep fixed:

  * `"width"` keeps the PNG’s width as is, adjust height to match the original aspect ratio.
  * `"height"` keeps the PNG’s height, adjust width.
 
`-> Image.Image`

* Returns a new (or possibly the same) Image object that has been resized so it has the same aspect ratio as the original source image.

### ***Getting the Current PNG Size***  

```Python
    Wp, Hp = png_img.size
```

* `png_img.size` is a Pillow property that returns (width, height) in pixels.
* `Wp` = PNG width (P for “PNG”).
* `Hp` = PNG height.

At this point:

* Source image: `(src_w, src_h)`
* Current ASCII PNG: `(Wp, Hp)`

They might have slightly different aspect ratios due to character sizing, font metrics, etc.

### ***Computing the Source Aspect Ratio***  

```Python
    src_aspect = src_w / src_h
```

* Aspect ratio is width / height.
* `src_aspect` = original image’s width divided by height.

Examples

* Landscape image, 1920×1080 → `src_aspect = 1920/1080 ≈ 1.777...` (16:9).
* Square image, 800×800 → `src_aspect = 1.0`.
* Tall/portrait image, 800×1600 → `src_aspect = 0.5`.

This value is the target ratio we want the final PNG to have.

### ***Deciding Which Dimension to Preserve***  

**If we preserve width**

```Python
    if preserve == "width":
        target_w = Wp
        target_h = max(1, int(round(target_w / src_aspect)))
```

Keep the PNG’s current width `Wp` and compute the height that gives the same aspect ratio as the source.

Since we want `target_w` / `target_h` = `src_aspect`, which re-arranged can be intended as `target_h` = `target_w` / `src_aspect`

The conditional statement will process:

`target_w = Wp`

* Use the PNG’s current width as the final width.

`target_h = max(1, int(round(target_w / src_aspect)))`  

* Compute `target_w / src_aspect` (a float).
* `round(...)` → nearest integer, to avoid systematic rounding bias.
* `int(...)` → convert to integer (Pillow needs integer pixel sizes).
* `max(1, ...)` → enforce at least 1 pixel height to avoid illegal size `(width, 0)` or `(width, negative)` in pathological cases.

**If we preserve height**  

```Python
    else:
        target_h = Hp
        target_w = max(1, int(round(target_h * src_aspect)))
```

It will keep the PNG’s current height Hp, adjust the width to match the aspect ratio.

As we still `target_w` / `target_h` = `src_aspect`, and it will be rearranged as `target_w` = `target_h` x `src_aspect`, we will process:

`target_h = Hp`

* Use the PNG’s current height.

`target_w = max(1, int(round(target_h * src_aspect)))`

* Compute the corresponding width so that the aspect ratio matches.
* Round to nearest integer.
* Clamp to at least 1 pixel.

### ***Conditional Resize With NEAREST Resampling***

```Python
    # NEAREST keeps ASCII pixels crisp
    if (target_w, target_h) != (Wp, Hp):
        png_img = png_img.resize((target_w, target_h), resample=Image.NEAREST)
    return png_img
```

**The condition**  

```Python
    if (target_w, target_h) != (Wp, Hp):
```

* If the target size is the same as the current size, there’s no need to do anything.
* This avoids an unnecessary resize operation (which can slightly degrade quality and costs CPU time).

**The Resize Itself**

```Python
        png_img = png_img.resize((target_w, target_h), resample=Image.NEAREST)
```

`png_img.resize((target_w, target_h), ...)`  

* Creates a new image with the given size.

`resample=Image.NEAREST`  

This is the nearest-neighbour resampling algorithm:  

  * It doesn’t blend pixels together like bilinear or bicubic would.
  * Each output pixel simply takes the colour of the nearest input pixel.
  * Result: blocky, crisp pixels — exactly what you want for ASCII art, where each character is like a little tile. If you used bilinear/bicubic, the characters would blur and smear, destroying that “terminal” look.

Then:

```Python
    return png_img
```

The (possibly resized) image is returned.

In short, `_aspect_correct_png`:  

1. Reads the current width/height of the ASCII PNG.
2. Computes the original image’s aspect ratio.
3. Depending on `preserve`:

  * Keeps width and recalculates height, or
  * Keeps height and recalculates width.

4. Ensures the new dimensions are valid integers and at least 1 pixel.
5. Resizes the image only if needed, using nearest-neighbour to keep characters crisp.
6. Returns the corrected PNG.

# ***The Public API: Convert and Export***  

```Python
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
```

The function takes an image (file or URL), turns it into ASCII art, saves three files (plain text, HTML, and PNG), fixes the PNG so it has the same aspect ratio as the original image, and optionally shows you an inline HTML preview. That’s the whole job of `convert_image_to_ascii`.

### ***Function Signature & Parameters***  

```Python
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
```
sdfsdf
