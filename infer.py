import cv2
import os
import argparse
from pathlib import Path
from pytorchocr.pytorch_paddle import PytorchPaddleOCR
from utils.visualize import draw_ocr_results


root_dir = Path(__file__).resolve().parent


def process_image(image_path, pytorch_paddle_ocr, save_path=None, show_confidence=True):
    """
    处理单张图片
    Args:
        image_path: 图片路径
        pytorch_paddle_ocr: OCR模型
        save_path: 保存路径，如果为None则不保存
        show_confidence: 是否显示置信度
    Returns:
        OCR结果
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None

    # 执行OCR识别
    dt_boxes, rec_res = pytorch_paddle_ocr(img)

    # 格式化OCR结果
    ocr_res = []
    if not dt_boxes and not rec_res:
        ocr_res.append(None)
    else:
        tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
        ocr_res.append(tmp_res)

    # 如果需要保存结果，绘制并保存
    if save_path:
        # 绘制OCR结果
        vis_img = draw_ocr_results(img, ocr_res, show_confidence)

        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存图片
        cv2.imwrite(save_path, vis_img)
        print(f"结果已保存至: {save_path}")

    return ocr_res


def process_directory(
    dir_path, pytorch_paddle_ocr, save_dir=None, show_confidence=True
):
    """
    处理目录中的所有图片
    Args:
        dir_path: 目录路径
        pytorch_paddle_ocr: OCR模型
        save_dir: 保存目录，如果为None则不保存
        show_confidence: 是否显示置信度
    """
    # 支持的图片格式
    img_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    # 遍历目录中的所有文件
    for file_path in Path(dir_path).glob("*"):
        if file_path.suffix.lower() in img_extensions:
            print(f"处理图片: {file_path}")

            # 如果需要保存结果，设置保存路径
            save_path = None
            if save_dir:
                save_path = os.path.join(
                    save_dir, f"{file_path.stem}_result{file_path.suffix}"
                )

            # 处理图片
            ocr_res = process_image(
                file_path, pytorch_paddle_ocr, save_path, show_confidence
            )
            print(f"OCR结果: {ocr_res}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="PaddleOCR-Torch-Infer")
    parser.add_argument("--data_path", required=True, help="输入图片路径或目录路径")
    parser.add_argument("--save_path", help="保存结果的路径或目录（可选）")
    parser.add_argument(
        "--show_confidence",
        action="store_true",
        default=False,
        help="是否在结果图像中显示置信度（默认不显示）",
    )
    args = parser.parse_args()

    # 初始化OCR模型
    pytorch_paddle_ocr = PytorchPaddleOCR()

    # 获取输入路径
    data_path = Path(args.data_path)

    # 检查输入路径是否存在
    if not data_path.exists():
        print(f"错误: 路径不存在 {data_path}")
        exit(1)

    # 根据输入类型处理
    if data_path.is_file():
        # 处理单个文件
        save_path = args.save_path if args.save_path else None
        ocr_res = process_image(
            data_path, pytorch_paddle_ocr, save_path, args.show_confidence
        )
        print(f"OCR结果: {ocr_res}")
    elif data_path.is_dir():
        # 处理目录
        save_dir = args.save_path if args.save_path else None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        process_directory(data_path, pytorch_paddle_ocr, save_dir, args.show_confidence)
    else:
        print(f"错误: 无效的路径类型 {data_path}")
