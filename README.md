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

