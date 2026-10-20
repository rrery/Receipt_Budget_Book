"""Optional local cache for unchanged training images; original bytes are retained."""
import hashlib
from io import BytesIO
import os
from pathlib import Path
import uuid

from PIL import Image


def download_image(supabase, image_path: str, bucket: str) -> Image.Image:
    directory = os.getenv("RECEIPT_IMAGE_CACHE")
    if directory:
        project = str(getattr(supabase, "supabase_url", ""))
        key = hashlib.sha256(f"{project}/{bucket}/{image_path}".encode()).hexdigest()
        path = Path(directory) / (key + ".image")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            content = supabase.storage.from_(bucket).download(image_path)
            # Verify before writing; do not cache an error page as an image.
            Image.open(BytesIO(content)).verify()
            temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        return Image.open(path).convert("RGB")
    content = supabase.storage.from_(bucket).download(image_path)
    return Image.open(BytesIO(content)).convert("RGB")
