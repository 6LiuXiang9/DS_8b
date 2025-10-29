from paddleocr import PaddleOCR
import os

image_path = r'G:\DeepSeek\DS_OCR\Paddle\PACS报告-陈晓茵.pdf'

print("=" * 60)
print("PP-OCRv5 文本识别测试")
print("=" * 60)

print("\n初始化 PP-OCRv5...")
ocr = PaddleOCR(
    text_detection_model_dir=r"G:\DeepSeek\DS_OCR\Paddle\paddle_models_v5\PP-OCRv5_server_det",
    text_recognition_model_dir=r"G:\DeepSeek\DS_OCR\Paddle\paddle_models_v5\PP-OCRv5_server_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="gpu:0",  # 或 "cpu"
)

print("开始识别...")
result = ocr.predict(image_path)

print("\n" + "=" * 60)
print("识别结果")
print("=" * 60)

# 创建输出目录
os.makedirs(r'G:\DeepSeek\DS_OCR\Paddle\output', exist_ok=True)

for res in result:
    # 获取识别的文本和置信度
    texts = res['rec_texts']
    scores = res['rec_scores']

    print(f"\n共检测到 {len(texts)} 行文本\n")

    # 打印所有文本
    for i, (text, score) in enumerate(zip(texts, scores), 1):
        print(f"{i:3d}. {text} (置信度: {score:.4f})")

    # 保存结果
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)

    img_path = r"G:\DeepSeek\DS_OCR\Paddle\output"
    json_path = r"G:\DeepSeek\DS_OCR\Paddle\output\result_v5.json"

    res.save_to_img(img_path)
    res.save_to_json(json_path)

    print(f"✓ 可视化图片已保存到: {img_path}")
    print(f"✓ JSON结果已保存到: {json_path}")

    # 保存纯文本文件（只包含识别内容）
    txt_path = r'G:\DeepSeek\DS_OCR\Paddle\output\result_v5.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(f"{text}\n")

    print(f"✓ 纯文本已保存到: {txt_path}")

    # 统计信息
    print("\n" + "=" * 60)
    print("统计信息")
    print("=" * 60)
    avg_score = sum(scores) / len(scores)
    print(f"平均置信度: {avg_score:.4f}")
    print(f"最高置信度: {max(scores):.4f}")
    print(f"最低置信度: {min(scores):.4f}")

    # 低置信度文本（<0.9）
    low_confidence = [(i + 1, text, score) for i, (text, score) in enumerate(zip(texts, scores)) if score < 0.9]
    if low_confidence:
        print(f"\n低置信度文本 (<0.9) 共 {len(low_confidence)} 行:")
        for idx, text, score in low_confidence:
            print(f"  第{idx}行: {text} ({score:.4f})")
    else:
        print("\n所有文本置信度均 ≥ 0.9")

print("\n" + "=" * 60)
print("识别完成！")
print("=" * 60)