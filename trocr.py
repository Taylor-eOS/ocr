import requests
import PIL
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

image_path = "1.png"
image = PIL.Image.open(image_path)
image = PIL.ImageOps.exif_transpose(image)
image = image.convert("RGB")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

