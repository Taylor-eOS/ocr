import fitz  #PyMuPDF
from PIL import Image, ImageEnhance
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import logging

#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Enhance image for better OCR results"""
def preprocess_image(img):
    try:
        img = img.convert("L")
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        return img.convert("RGB")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return img

def pdf_to_text(input_pdf, output_txt):
    try:
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
        logger.info("Loaded TrOCR model for printed text")
        full_text = []
        with fitz.open(input_pdf) as doc:
            for page_num, page in enumerate(doc):
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                processed_img = preprocess_image(img)
                pixel_values = processor(images=processed_img, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values)
                page_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                full_text.append(f"Page {page_num+1}:\n{page_text}\n")
                logger.info(f"Processed page {page_num+1}")
        with open(output_txt, "w") as f:
            f.write("\n".join(full_text))
        logger.info(f"Saved OCR results to {output_txt}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    pdf_to_text("input.pdf", "output.txt")

