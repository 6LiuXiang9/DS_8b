from paddleocr import PaddleOCR
import os
import re
import csv
from pathlib import Path
import fitz  # PyMuPDF - 用于 PDF 转图片
import tempfile
import time


class MedicalReportOCR:
    def __init__(self, det_model_dir, rec_model_dir, device="gpu:0"):
        """初始化 OCR 模型"""
        print("初始化 PP-OCRv5...")
        self.ocr = PaddleOCR(
            text_detection_model_dir=det_model_dir,
            text_recognition_model_dir=rec_model_dir,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
        )

    @staticmethod
    def safe_remove_file(file_path, max_retries=3, retry_delay=0.5):
        """
        安全删除文件，带重试机制

        参数:
            file_path: 要删除的文件路径
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）

        返回:
            bool: 是否成功删除
        """
        for attempt in range(max_retries):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    # 等待后重试
                    time.sleep(retry_delay)
                else:
                    # 最后一次失败，输出警告
                    print(f"    ⚠ 警告：无法删除临时文件 {os.path.basename(file_path)}: {e}")
                    return False
        return True

    def pdf_to_images(self, pdf_path, output_dir, dpi=300):
        """
        将PDF转换为图片列表，保存到指定目录

        参数:
            pdf_path: PDF文件路径
            output_dir: 输出目录
            dpi: 图片分辨率

        返回:
            list: 图片文件路径列表
        """
        images = []
        pdf_document = fitz.open(pdf_path)

        # 使用PDF文件名作为前缀（去除路径和扩展名）
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

        # 添加时间戳确保唯一性
        timestamp = int(time.time() * 1000)

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            # 转换为图片
            mat = fitz.Matrix(dpi / 72, dpi / 72)  # 设置分辨率
            pix = page.get_pixmap(matrix=mat)

            # 生成唯一文件名
            img_filename = f"{pdf_name}_{timestamp}_page_{page_num}.png"
            img_path = os.path.join(output_dir, img_filename)

            pix.save(img_path)
            images.append(img_path)

        pdf_document.close()
        return images

    def extract_text_from_pdf(self, pdf_path):
        """
        从 PDF 提取文本
        使用临时目录 + 重试机制确保稳定性

        参数:
            pdf_path: PDF文件路径

        返回:
            str: 提取的文本内容
        """
        all_text = ""

        # 使用临时目录（自动清理）
        with tempfile.TemporaryDirectory(prefix='ocr_temp_') as temp_dir:
            image_paths = []

            try:
                # 将 PDF 转换为图片（保存到临时目录）
                image_paths = self.pdf_to_images(pdf_path, temp_dir)

                # 对每页进行 OCR
                for img_path in image_paths:
                    result = self.ocr.predict(img_path)

                    # 提取文本
                    for res in result:
                        texts = res['rec_texts']
                        all_text += '\n'.join(texts) + '\n'

                    # OCR完成后立即删除图片（释放空间）
                    self.safe_remove_file(img_path)

            except Exception as e:
                # 确保清理所有临时文件
                for img_path in image_paths:
                    self.safe_remove_file(img_path)
                raise e

        # 退出 with 块时，临时目录会自动删除
        return all_text


class ReportParser:
    """报告解析器 - 使用优化后的正则规则"""

    @staticmethod
    def extract_field(text, field_name, default=""):
        """
        提取单行字段
        支持字段名变体和容错匹配
        """
        # 支持中英文冒号，以及可能的空格
        pattern = rf'{field_name}\s*[：:]\s*([^\n]*?)(?=\s+[\u4e00-\u9fa5]+\s*[：:]|\n|$)'
        match = re.search(pattern, text)

        if match:
            value = match.group(1).strip()
            # 过滤特殊值
            if value and value not in ['0', '']:
                return value

        return default

    @staticmethod
    def extract_long_field(text, start_keyword, end_keywords=None):
        """
        提取长文本字段（如意见、提示）

        参数:
            text: 完整文本
            start_keyword: 起始关键字
            end_keywords: 结束关键字列表（遇到任一即停止）
        """
        # 查找起始位置
        start_pattern = rf'{start_keyword}\s*[：:]\s*'
        start_match = re.search(start_pattern, text)

        if not start_match:
            return ""

        start_pos = start_match.end()

        # 如果没有指定结束关键字，使用默认的常见字段名
        if end_keywords is None:
            end_keywords = ['检查医生', '审核医生', '报告日期', '本报告']

        # 构建结束模式 - 匹配任一结束关键字
        end_pattern = '|'.join([rf'{kw}\s*[：:]' for kw in end_keywords])
        end_match = re.search(end_pattern, text[start_pos:])

        if end_match:
            end_pos = start_pos + end_match.start()
            content = text[start_pos:end_pos].strip()
        else:
            # 如果没找到结束标记，提取到文本末尾
            content = text[start_pos:].strip()

        # 清洗内容
        content = ReportParser.clean_long_text(content)

        return content

    @staticmethod
    def clean_long_text(text):
        """
        清洗长文本内容
        """
        # 移除可能的噪声
        noise_patterns = [
            r'本报告只作临床参考.*',  # 底部说明
            r'[A-F0-9]{32,}',  # Hash 码
            r'^\s*0\s*$',  # 单独的 0（但保留"0-1岁"这种）
        ]

        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

        # 清理多余空白
        text = re.sub(r'\n{3,}', '\n\n', text)  # 最多保留两个换行
        text = re.sub(r' {2,}', ' ', text)  # 多个空格变一个
        text = text.strip()

        return text

    @staticmethod
    def parse_report(ocr_text):
        """
        解析报告文本，提取所有字段
        采用方案B：字段别名都保留，分别作为独立列
        """
        data = {
            # 基本信息 - 兼容字段分别提取
            '检查号': ReportParser.extract_field(ocr_text, '检查号'),
            '编号': ReportParser.extract_field(ocr_text, '编号'),

            '姓名': ReportParser.extract_field(ocr_text, '姓名'),
            '性别': ReportParser.extract_field(ocr_text, '性别'),
            '年龄': ReportParser.extract_field(ocr_text, '年龄'),
            '门诊号': ReportParser.extract_field(ocr_text, '门诊号'),
            '病历号': ReportParser.extract_field(ocr_text, '病历号'),

            # 送诊信息 - 兼容字段分别提取
            '送诊单位': ReportParser.extract_field(ocr_text, '送诊单位'),
            '送诊科室': ReportParser.extract_field(ocr_text, '送诊科室'),

            '送诊医生': ReportParser.extract_field(ocr_text, '送诊医生'),
            '检查日期': ReportParser.extract_field(ocr_text, '检查日期'),
            '检查部位': ReportParser.extract_field(ocr_text, '检查部位'),
            '临床诊断': ReportParser.extract_field(ocr_text, '临床诊断'),
        }

        # 长文本字段
        data['意见'] = ReportParser.extract_long_field(
            ocr_text,
            '意见',
            end_keywords=['提示', '检查医生']
        )

        data['提示'] = ReportParser.extract_long_field(
            ocr_text,
            '提示',
            end_keywords=['检查医生', '审核医生', '报告日期']
        )

        # 报告信息
        data['检查医生'] = ReportParser.extract_field(ocr_text, '检查医生')
        data['审核医生'] = ReportParser.extract_field(ocr_text, '审核医生')
        data['报告日期'] = ReportParser.extract_field(ocr_text, '报告日期')

        return data


def process_pdf_folder(pdf_folder, output_csv, output_txt_folder,
                       det_model_dir, rec_model_dir, device="gpu:0"):
    """
    批量处理 PDF 文件夹
    """
    print("=" * 60)
    print("PP-OCRv5 批量处理医疗报告")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(output_txt_folder, exist_ok=True)

    # 初始化 OCR
    ocr_processor = MedicalReportOCR(det_model_dir, rec_model_dir, device)

    # 获取所有 PDF 文件
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    total_files = len(pdf_files)

    print(f"\n找到 {total_files} 个 PDF 文件\n")

    # 存储所有结果
    all_results = []
    success_count = 0
    failed_count = 0

    # 逐个处理
    for idx, pdf_filename in enumerate(pdf_files, start=1):
        pdf_path = os.path.join(pdf_folder, pdf_filename)

        print(f"[{idx}/{total_files}] 正在处理: {pdf_filename}")

        try:
            # 步骤1: OCR 提取文本
            print("  - OCR 识别中...")
            ocr_text = ocr_processor.extract_text_from_pdf(pdf_path)

            # 保存原始 OCR 文本（用于调试）
            txt_filename = os.path.splitext(pdf_filename)[0] + '.txt'
            txt_path = os.path.join(output_txt_folder, txt_filename)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(ocr_text)

            # 步骤2: 解析提取字段
            print("  - 正则解析中...")
            structured_data = ReportParser.parse_report(ocr_text)

            # 添加文件名
            structured_data['文件名'] = pdf_filename

            all_results.append(structured_data)
            success_count += 1

            print(f"  ✓ 成功提取")

            # 显示关键信息（优先显示有值的字段）
            name = structured_data.get('姓名', 'N/A')
            check_no = structured_data.get('检查号') or structured_data.get('编号', 'N/A')
            check_date = structured_data.get('检查日期', 'N/A')

            print(f"    姓名: {name}")
            print(f"    编号: {check_no}")
            print(f"    日期: {check_date}")

        except Exception as e:
            failed_count += 1
            print(f"  ✗ 处理失败: {str(e)}")

            # 记录失败信息
            error_data = {
                '文件名': pdf_filename,
                '检查号': '',
                '编号': f'ERROR: {str(e)}',
            }
            # 填充其他字段为空
            for key in ['姓名', '性别', '年龄', '门诊号', '病历号',
                        '送诊单位', '送诊科室', '送诊医生', '检查日期',
                        '检查部位', '临床诊断', '意见', '提示',
                        '检查医生', '审核医生', '报告日期']:
                error_data[key] = ''

            all_results.append(error_data)

    # 保存到 CSV
    print("\n" + "=" * 60)
    print("保存结果到 CSV...")
    print("=" * 60)

    save_to_csv(all_results, output_csv)

    # 最终统计
    print("\n" + "=" * 60)
    print("✓ 批量处理完成！")
    print("=" * 60)
    print(f"总文件数: {total_files}")
    print(f"✓ 成功: {success_count} 个 ({success_count / total_files * 100:.1f}%)")
    print(f"✗ 失败: {failed_count} 个 ({failed_count / total_files * 100:.1f}%)")
    print(f"\n输出文件:")
    print(f"  CSV 结果: {output_csv}")
    print(f"  OCR 原始文本: {output_txt_folder}/")
    print("=" * 60)


def save_to_csv(data_list, output_path):
    """保存到 CSV - 方案B：保留所有字段别名"""
    if not data_list:
        print("没有数据需要保存")
        return

    # 定义字段顺序 - 兼容字段都保留
    fieldnames = [
        '文件名',
        '检查号', '编号',  # 两个都保留
        '姓名', '性别', '年龄',
        '门诊号', '病历号',
        '送诊单位', '送诊科室',  # 两个都保留
        '送诊医生',
        '检查日期', '检查部位', '临床诊断',
        '意见', '提示',
        '检查医生', '审核医生', '报告日期'
    ]

    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_list)

    print(f"✓ 已保存 {len(data_list)} 条记录")
    print(f"✓ CSV 包含 {len(fieldnames)} 列")


# ============ 主程序 ============
if __name__ == "__main__":
    # ===== 配置参数 =====

    # 输入：PDF 文件夹路径
    PDF_FOLDER = r"E:\Deepseek\data_sorce\HZ\Data"

    # 输出：CSV 文件路径
    OUTPUT_CSV = r"所有报告_汇总_PPOCRv5.csv"

    # 输出：OCR 原始文本保存文件夹
    OUTPUT_TXT_FOLDER = r"OCR原始文本_PPOCRv5"

    # PP-OCRv5 模型路径
    DET_MODEL_DIR = r"G:\DeepSeek\DS_OCR\Paddle\paddle_models_v5\PP-OCRv5_server_det"
    REC_MODEL_DIR = r"G:\DeepSeek\DS_OCR\Paddle\paddle_models_v5\PP-OCRv5_server_rec"

    # 设备选择：'gpu:0' 或 'cpu'
    DEVICE = "gpu:0"

    # ===== 执行处理 =====
    process_pdf_folder(
        pdf_folder=PDF_FOLDER,
        output_csv=OUTPUT_CSV,
        output_txt_folder=OUTPUT_TXT_FOLDER,
        det_model_dir=DET_MODEL_DIR,
        rec_model_dir=REC_MODEL_DIR,
        device=DEVICE
    )