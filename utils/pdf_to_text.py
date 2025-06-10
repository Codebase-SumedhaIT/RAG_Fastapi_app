import os
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path, output_path):
    try:
        # Try PyPDF2 first
        reader = PdfReader(pdf_path)
        full_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                full_text.append(text)
        if full_text:
            complete_text = "\n\n".join(full_text)
        else:
            # Fallback to OCR
            pages = convert_from_path(pdf_path, poppler_path=r'C:\Program Files\poppler\Library\bin', dpi=300)
            full_text = [pytesseract.image_to_string(page) for page in pages]
            complete_text = "\n\n".join(full_text)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(complete_text)
        return output_path
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None