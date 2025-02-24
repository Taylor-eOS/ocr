import cv2
import numpy as np

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

if __name__ == "__main__":
    gaps = detect_line_gaps("image.jpg")
    print(gaps)

