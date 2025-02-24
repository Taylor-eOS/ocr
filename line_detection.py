import os
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tesseract_ocr import preprocess_image

debug = True

def detect_line_gaps(image, min_gap=5, white_ratio=0.95):
    if isinstance(image, str):
        image = cv2.imread(image)
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    projection = np.sum(bw == 255, axis=1)
    max_val = np.max(projection)
    threshold = white_ratio * max_val
    gaps = []
    in_gap = False
    start = 0
    for i, val in enumerate(projection):
        if val >= threshold:
            if not in_gap:
                in_gap = True
                start = i
        else:
            if in_gap:
                if i - start >= min_gap:
                    gaps.append((start, i))
                in_gap = False
    if in_gap and (len(projection) - start >= min_gap):
        gaps.append((start, len(projection)))
    return [(s + e) // 2 for s, e in gaps]

def image_to_text(image_input, debug_dir="."):
    if isinstance(image_input, str):
        image = Image.open(image_input)
    else:
        image = image_input
    cv_image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gaps = detect_line_gaps(cv_image)
    if debug:
        print("Detected gaps:", gaps)
        print("Current working directory:", os.getcwd())
    width, height = image.size
    boundaries = [0] + gaps + [height]
    if debug: print("Computed line boundaries:", boundaries)
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    full_text = []
    margin = 1
    for i in range(len(boundaries) - 1):
        top = max(0, boundaries[i] - margin)
        bottom = min(height, boundaries[i+1] + margin)
        if debug: print(f"Processing line {i+1}: top={top}, bottom={bottom}")
        line_img = image.crop((0, top, width, bottom))
        if debug:
            debug_filename = os.path.join(debug_dir, f"debug_line_{i+1}.png")
            line_img.save(debug_filename)
            print(f"Saved debug image: {debug_filename}")
        line_img = preprocess_image(line_img)
        pixel_values = processor(images=line_img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        line_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if debug: print(f"OCR for line {i+1}: {line_text}")
        full_text.append(line_text.lower())
    return "\n".join(full_text)

if __name__ == "__main__":
    #gaps = detect_line_gaps("image.jpg")
    text = image_to_text("image.jpg")
    print(text)

