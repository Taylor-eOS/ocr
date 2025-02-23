import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
import pytesseract
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(img):
    try:
        img = img.convert("L")
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        return img
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return img

def pdf_to_text(input_pdf, output_txt):
    try:
        full_text = []
        
        with fitz.open(input_pdf) as doc:
            for page_num, page in enumerate(doc):
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                processed_img = preprocess_image(img)
                page_text = pytesseract.image_to_string(processed_img, lang="eng")
                full_text.append(f"Page {page_num+1}:\n{page_text}\n")
                logger.info(f"Processed page {page_num+1}")
        with open(output_txt, "w") as f:
            f.write("\n".join(full_text))
        logger.info(f"Successfully saved OCR results to {output_txt}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    pdf_to_text("input.pdf", "output.txt")

