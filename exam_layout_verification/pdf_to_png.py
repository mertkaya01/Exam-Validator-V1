import argparse
import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

# Configure module-level logger
logger = logging.getLogger(__name__)

class PDFConversionError(Exception):
    """Base exception for PDF conversion errors."""
    pass

def convert_pdf_to_png(
    pdf_path: Path, 
    output_dir: Path, 
    dpi: int = 300
) -> List[Path]:
    """
    Converts a PDF file into high-resolution PNG images.

    Args:
        pdf_path: Path to the input PDF document.
        output_dir: Directory where the output PNGs will be saved.
        dpi: Target resolution for the output images. 300 is recommended for OCR.

    Returns:
        A list of paths to the generated PNG images.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF document not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_images: List[Path] = []

    try:
        document = fitz.open(pdf_path)
        logger.info("Opened PDF '%s' (%d pages).", pdf_path.name, len(document))
    except Exception as exc:
        raise PDFConversionError(f"Failed to open PDF file: {exc}") from exc

    # PDF standard is 72 DPI. We scale it up using a matrix for better clarity.
    zoom_factor = dpi / 72.0
    transform_matrix = fitz.Matrix(zoom_factor, zoom_factor)

    for page_index in range(len(document)):
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=transform_matrix)
        
        output_filename = f"{pdf_path.stem}_page_{page_index + 1}.png"
        output_file = output_dir / output_filename
        
        pixmap.save(str(output_file))
        saved_images.append(output_file)
        
        logger.debug("Saved: %s", output_file.name)

    document.close()
    logger.info(
        "Successfully converted %d pages to '%s'.", 
        len(saved_images), 
        output_dir
    )
    
    return saved_images

def main() -> None:
    """Entry point for CLI execution."""
    parser = argparse.ArgumentParser(
        prog="pdf_to_png",
        description="Convert a PDF document to high-resolution PNG images for layout parsing."
    )
    parser.add_argument(
        "--pdf", "-p", 
        type=Path, 
        required=True, 
        help="Path to the input PDF file (e.g., sample_exam.pdf)."
    )
    parser.add_argument(
        "--output", "-o", 
        type=Path, 
        default=Path("./examples"), 
        help="Output directory for the PNG images (default: ./examples)."
    )
    parser.add_argument(
        "--dpi", "-d", 
        type=int, 
        default=300, 
        help="Resolution in DPI. 300 is optimal for OpenCV/OCR (default: 300)."
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable DEBUG-level logging."
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        convert_pdf_to_png(pdf_path=args.pdf, output_dir=args.output, dpi=args.dpi)
    except Exception as exc:
        logger.error("Fatal error during conversion: %s", exc)
        raise SystemExit(1) from exc

if __name__ == "__main__":
    main()