import fitz
import pytesseract
from PIL import Image
from pathlib import Path

def extract_images_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_path = Path(pdf_path).parent / f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)
            images.append(image_path)
    return images

def perform_ocr_on_images(image_paths, output_txt):
    with open(output_txt, "w") as txt_file:
        for image_path in image_paths:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang="eng")
            txt_file.write(text + "\n")
            print(f"Image done: {text[:30].replace('\n', '')}")

if __name__ == "__main__":
    pdf_path = "ocr_files/o.pdf"
    output_txt = "ocr_files/o.txt"
    image_paths = extract_images_from_pdf(pdf_path)
    perform_ocr_on_images(image_paths, output_txt)

