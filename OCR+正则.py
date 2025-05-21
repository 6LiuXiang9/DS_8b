import pytesseract
from PIL import Image
import pdfplumber
import json
import os

# Set the path for Tesseract (adjust the path as needed for your system)
pytesseract.pytesseract.tesseract_cmd = r"Tesseract\tesseract.exe"  # Change to your Tesseract path


# Function to extract text using OCR from a PDF
def extract_text_from_pdf_with_ocr(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Try to extract text
            text = page.extract_text()
            if text:
                all_text += text + "\n"
            else:
                # If no text, extract image and apply OCR
                for image in page.images:
                    page_image = page.to_image()
                    cropped_img = page_image.crop(image['x0'], image['top'], image['x1'], image['bottom']).original
                    ocr_text = pytesseract.image_to_string(cropped_img, lang='chi_sim')  # Using simplified Chinese OCR
                    all_text += ocr_text + "\n"
    return all_text


# Function to process a single PDF file
def process_pdf(pdf_path):
    # Extract text using OCR
    ocr_text_result = extract_text_from_pdf_with_ocr(pdf_path)

    # Check if "深圳市人民医院" is found in the extracted text
    if "深圳市人民医院" not in ocr_text_result:
        return {
            "error": "Condition not met: '深圳市人民医院' not found in the document."
        }

    # Extract content between "意见" and "提示"
    start_index_1 = ocr_text_result.find("意见：") + len("意见：")
    end_index_1 = ocr_text_result.find("提示")
    extracted_text_1 = ocr_text_result[start_index_1:end_index_1].strip()

    # Extract the line immediately following "提示"
    start_index_2 = ocr_text_result.find("提示")
    if start_index_2 != -1:
        # Extract the content right after "提示" until the next line break
        remaining_text = ocr_text_result[start_index_2 + len("提示"):].strip()
        extracted_text_2 = remaining_text.split("\n")[0].strip()  # Extract only the next line after "提示:"
    else:
        extracted_text_2 = "提示 not found."

    # Return the extracted data in the required format
    return {
        "question": extracted_text_1,
        "response": extracted_text_2
    }


# Folder path containing the PDF files
pdf_folder_path = r"E:\deepseek-r1-llama-8b\Data"  # Update to your folder location
output_json_path = r"E:\deepseek-r1-llama-8b\output.json"  # Path to save the output JSON file

# Dictionary to hold the processed data for each file
all_data = {}

# Loop over all files in the folder
for idx, pdf_filename in enumerate(os.listdir(pdf_folder_path)):
    if pdf_filename.endswith(".pdf"):  # Check if it's a PDF file
        pdf_path = os.path.join(pdf_folder_path, pdf_filename)

        # Process each PDF and store the result
        data = process_pdf(pdf_path)

        # If the data is valid (no error), add it to the all_data dictionary
        if "error" not in data:
            all_data[str(idx + 1).zfill(3)] = data
        else:
            # If there is an error, we can skip or add the error message to the data
            all_data[str(idx + 1).zfill(3)] = data

# Save the aggregated data as a JSON file
with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(all_data, json_file, ensure_ascii=False, indent=4)

# Display a message confirming the process is complete
print(f"\nAll PDFs have been processed and the results have been saved to {output_json_path}")
