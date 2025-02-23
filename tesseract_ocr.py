import fitz
import pytesseract
from PIL import Image, ImageOps, ImageFilter
import tempfile

#sudo apt install tesseract-ocr
#sudo apt install tesseract-ocr-deu

# Set Tesseract path (Linux)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def preprocess_image(img):
    """Improve image for OCR"""
    img = img.convert("L")  # Grayscale
    img = ImageOps.autocontrast(img)  # Enhance contrast
    img = img.filter(ImageFilter.SHARPEN)  # Sharpen
    return img

def extract_text_from_pdf_images(pdf_path, output_txt="output.txt"):
    text = []
    with tempfile.TemporaryDirectory() as temp_dir:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            images = page.get_images()
            if images:
                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_filename = f"{temp_dir}/page{page_num + 1}-img{img_index + 1}.{image_ext}"
                    
                    with open(image_filename, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    img = Image.open(image_filename)
                    img = preprocess_image(img)  # Preprocess
                    
                    # Try German first, then Fraktur if needed
                    try:
                        ocr_text = pytesseract.image_to_string(
                            img, 
                            lang='deu',
                            config='--psm 6 --oem 3'
                        )
                    except:
                        ocr_text = pytesseract.image_to_string(
                            img, 
                            lang='deu-frak',
                            config='--psm 6 --oem 3'
                        )
                    
                    text.append(ocr_text)
        
        full_text = "\n".join(text)
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Processed text saved to {output_txt}")

if __name__ == "__main__":
    extract_text_from_pdf_images("input.pdf", "output.txt")
