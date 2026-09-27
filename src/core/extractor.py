"""
Hybrid Document Extractor
Extracts digital text and renders high-resolution page images for scanned PDFs and photo uploads.
Guarantees visual and textual input for multimodal Gemini processing.
"""

import io
from typing import List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image

from src.utils.helpers import get_logger

logger = get_logger(__name__)

# Check library availability
try:
    import pymupdf as fitz
    FITZ_AVAILABLE = True
except ImportError:
    try:
        import fitz
        FITZ_AVAILABLE = True
    except ImportError:
        FITZ_AVAILABLE = False
        logger.warning("PyMuPDF not available. Install PyMuPDF for scanned PDF page rendering.")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


@dataclass
class ExtractedDocument:
    """Container for processed document data."""
    file_name: str
    file_type: str  # 'pdf' or 'image'
    raw_text: str
    page_images: List[Image.Image]
    page_count: int
    is_scanned: bool
    status_message: str


class DocumentExtractor:
    """
    Multimodal Document Extractor for medical lab reports,
    prescriptions, and discharge summaries.
    """

    def __init__(self, max_pages_to_render: int = 5, render_dpi: int = 200):
        self.max_pages_to_render = max_pages_to_render
        self.render_dpi = render_dpi

    def process_file(self, uploaded_file) -> ExtractedDocument:
        """
        Process any uploaded file (PDF or Image).
        Extracts both selectable digital text and high-res page images.
        """
        file_name = uploaded_file.name
        file_ext = file_name.lower().split(".")[-1]
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()

        if file_ext == "pdf":
            return self._process_pdf(file_name, file_bytes)
        elif file_ext in ["png", "jpg", "jpeg", "tiff", "bmp", "webp"]:
            return self._process_image(file_name, file_bytes)
        else:
            raise ValueError(f"Unsupported file format: .{file_ext}. Please upload a PDF or image file.")

    def _process_pdf(self, file_name: str, file_bytes: bytes) -> ExtractedDocument:
        """Extract text and render pages to PIL images from PDF."""
        raw_text = ""
        page_images: List[Image.Image] = []
        page_count = 0

        # Method 1: Try PyMuPDF (fitz) for both text extraction and page rendering
        if FITZ_AVAILABLE:
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                page_count = len(doc)
                pages_to_render = min(page_count, self.max_pages_to_render)

                for page_idx in range(pages_to_render):
                    page = doc[page_idx]
                    
                    # Extract digital text
                    text = page.get_text()
                    if text:
                        raw_text += f"\n--- Page {page_idx + 1} ---\n{text}\n"

                    # Render page as high-res image (for scanned/multimodal vision)
                    pix = page.get_pixmap(dpi=self.render_dpi)
                    img_bytes = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    page_images.append(img)

                doc.close()
                logger.info(f"PyMuPDF processed {pages_to_render}/{page_count} pages with {len(raw_text)} text chars.")
            except Exception as e:
                logger.warning(f"PyMuPDF failed: {e}. Trying fallback methods.")

        # Method 2: Fallback to pdfplumber for text if PyMuPDF extracted minimal text
        if len(raw_text.strip()) < 50 and PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    if page_count == 0:
                        page_count = len(pdf.pages)
                    plumber_text = ""
                    for p_num, page in enumerate(pdf.pages[:self.max_pages_to_render]):
                        t = page.extract_text()
                        if t:
                            plumber_text += f"\n--- Page {p_num + 1} ---\n{t}\n"
                    if len(plumber_text.strip()) > len(raw_text.strip()):
                        raw_text = plumber_text
            except Exception as e:
                logger.warning(f"pdfplumber text extraction failed: {e}")

        # Method 3: Fallback to PyPDF2 if still empty
        if len(raw_text.strip()) < 50 and PYPDF2_AVAILABLE:
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                if page_count == 0:
                    page_count = len(reader.pages)
                pypdf_text = ""
                for p_num, page in enumerate(reader.pages[:self.max_pages_to_render]):
                    t = page.extract_text()
                    if t:
                        pypdf_text += f"\n--- Page {p_num + 1} ---\n{t}\n"
                if len(pypdf_text.strip()) > len(raw_text.strip()):
                    raw_text = pypdf_text
            except Exception as e:
                logger.warning(f"PyPDF2 text extraction failed: {e}")

        # Determine if document is scanned (less than 50 text chars)
        is_scanned = len(raw_text.strip()) < 50

        if is_scanned:
            status_msg = f"Scanned/Image PDF detected ({page_count} pages). Multimodal Vision will visually extract lab data."
        else:
            status_msg = f"Digital PDF detected ({page_count} pages, {len(raw_text)} characters). Multimodal Vision will inspect tables."

        return ExtractedDocument(
            file_name=file_name,
            file_type="pdf",
            raw_text=raw_text.strip(),
            page_images=page_images,
            page_count=max(page_count, 1),
            is_scanned=is_scanned,
            status_message=status_msg
        )

    def _process_image(self, file_name: str, file_bytes: bytes) -> ExtractedDocument:
        """Process image file (PNG, JPG, etc.)."""
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return ExtractedDocument(
            file_name=file_name,
            file_type="image",
            raw_text="",
            page_images=[img],
            page_count=1,
            is_scanned=True,
            status_message="Image uploaded. Multimodal Vision will visually extract lab data."
        )
