from pathlib import Path
import cv2

from pytorchocr.pytorch_paddle import PytorchPaddleOCR


root_dir = Path(__file__).resolve().parent


if __name__ == "__main__":
    pytorch_paddle_ocr = PytorchPaddleOCR()
    img = cv2.imread("general_ocr_rec_001.png")
    dt_boxes, rec_res = pytorch_paddle_ocr(img)
    ocr_res = []
    if not dt_boxes and not rec_res:
        ocr_res.append(None)
    else:
        tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
        ocr_res.append(tmp_res)
    print(ocr_res)
