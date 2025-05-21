# PaddleOCR-Torch-Infer

---

PaddleOCR是效果最好的开源OCR工具之一，然而，其原生只支持在paddlepaddle框架中运行。


PaddleOCR-Torch-Infer 是用Pytorch将PaddleOCR的模型推理



## 使用方法

### 命令行参数

- `--data_path`：**必需参数**，指定输入图片路径或目录路径
- `--save_path`：**可选参数**，指定保存结果的路径或目录
- `--show_confidence`：**可选参数**，是否在结果图像中显示置信度（默认不显示）

## 使用示例

### 单图片处理

```bash
python infer.py --data_path test_img/general_ocr_rec_001.png --save_path output/result.png
```

### 目录批量处理

```bash
python infer.py --data_path test_img --save_path output
```

## 输出说明

- 程序会在图像上绘制绿色边界框标注检测到的文本区域
- 在每个文本区域上方显示识别的文本内容（可选显示置信度）
- 如果指定了保存路径，会将可视化结果保存到指定位置
- 同时在控制台输出OCR识别结果

## 注意事项

- 支持的图片格式包括：jpg、jpeg、png、bmp、tiff
- 程序会自动创建保存目录（如果不存在）
- 对于中文显示，程序会自动检测并使用系统中可用的中文字体
- 如果系统中没有可用的中文字体，将使用默认字体，可能无法正确显示中文字符

## 项目结构

```
TorchocrInfer/
├── infer.py              # 主程序入口
├── utils/                # 工具函数目录
│   └── visualize.py      # 可视化相关函数
└── test_img/             # 测试图片目录
```

## 参考

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

- [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch)

- [MinerU](https://github.com/opendatalab/MinerU)

