import cv2
import time
import argparse
from pathlib import Path
from pytorchocr.pytorch_paddle import PytorchPaddleOCR
from utils.visualize import draw_ocr_results


def process_frame(frame, pytorch_paddle_ocr, show_confidence=True):
    """
    对单帧图像进行 OCR 推理并返回结果和推理时间
    """
    start_time = time.time()
    dt_boxes, rec_res = pytorch_paddle_ocr(frame)
    inference_time = time.time() - start_time

    ocr_res = []
    if not dt_boxes and not rec_res:
        ocr_res.append(None)
    else:
        tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
        ocr_res.append(tmp_res)

    return ocr_res, inference_time


def process_video(video_path, pytorch_paddle_ocr, output_dir, show_confidence=True):
    video_path = Path(video_path)
    video_name = video_path.stem
    label_dir = Path(output_dir) / "output_labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video_path = Path(output_dir) / f"{video_name}_ocr_result.mp4"
    out_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    frame_idx = 0
    total_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ocr_res, inference_time = process_frame(
            frame, pytorch_paddle_ocr, show_confidence
        )
        total_time += inference_time

        # 写入文字文件
        label_filename = label_dir / f"{video_name}_ocr_{frame_idx}.txt"
        with open(label_filename, "w", encoding="utf-8") as f:
            if ocr_res[0] is not None:
                for box, (text, conf) in ocr_res[0]:
                    f.write(f"{text}\t{box}\t{conf:.3f}\n")

        # 可视化并写入视频
        vis_frame = draw_ocr_results(frame, ocr_res, show_confidence)
        out_writer.write(vis_frame)

        print(f"帧 {frame_idx} 推理时间: {inference_time:.3f}秒")
        frame_idx += 1

    cap.release()
    out_writer.release()

    if frame_idx > 0:
        avg_time = total_time / frame_idx
        print(f"\n视频处理完成: {video_name}")
        print(f"总帧数: {frame_idx}")
        print(f"总推理时间: {total_time:.3f}秒")
        print(f"平均帧处理时间: {avg_time:.3f}秒")
    else:
        print("未处理任何帧")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频OCR推理工具")
    parser.add_argument(
        "--video_path", default="test_video/ocr_test.mp4", help="输入视频路径（.mp4）"
    )
    parser.add_argument("--output_dir", default="output_video", help="结果保存目录")
    parser.add_argument(
        "--show_confidence", default=True, help="是否在视频中显示置信度"
    )
    args = parser.parse_args()

    # 加载模型
    print("加载OCR模型...")
    load_start = time.time()
    pytorch_paddle_ocr = PytorchPaddleOCR()
    print(f"模型加载完成，用时: {time.time() - load_start:.3f}秒")

    process_video(
        args.video_path, pytorch_paddle_ocr, args.output_dir, args.show_confidence
    )
