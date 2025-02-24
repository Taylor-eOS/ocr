import json
import logging
import pytesseract
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance

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

def images_to_text(image_folder, output_txt="output.txt"):
    from PIL import Image, ImageOps, ImageFilter
    import os
    import pytesseract
    text = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, filename)
            try:
                img = Image.open(img_path)
                img = img.convert("L")
                img = ImageOps.autocontrast(img)
                img = img.filter(ImageFilter.SHARPEN)
                ocr_text = pytesseract.image_to_string(img, lang='deu', config='--psm 6 --oem 3')
                #text.append(f"Image: {filename}\n{ocr_text}\n")
                text.append(f"{ocr_text}\n")
            except Exception as e:
                text.append(f"Error processing {filename}: {str(e)}")
    
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(text))

if __name__ == "__main__":
    #input_folder = input("Input folder: ")
    input_folder = "input_folder"
    images_to_text(input_folder)

