# RAG Based Learning Assistant

A full-stack application for document-based question answering and MCQ generation using Retrieval-Augmented Generation (RAG), FastAPI, React, and local LLMs.

---

## Features

- **Document Upload:** Upload PDFs to custom domains for indexing and retrieval.
- **Query Assistant:** Ask questions and get answers with source references from your uploaded documents.
- **MCQ Generator:** Generate multiple-choice questions (MCQs) from your documents, with both student and teacher views.
- **Domain Management:** Organize documents and knowledge by domain.
- **Local LLM Support:** Uses local models (Mistral, Llama-2, DeepSeek) via GGUF and llama.cpp.
- **OCR Support:** Extracts text from scanned PDFs using Tesseract OCR.
- **FAISS Embeddings:** Fast similarity search using FAISS and Sentence Transformers.

---

## Project Structure

```
rag-fastapi-react-app/
│
├── rag-frontend/           # React frontend (UI)
├── Embeddor/faiss_index/   # Domain-wise FAISS indexes and metadata
├── utils/                  # PDF/text processing utilities
├── fastapi_app.py          # FastAPI backend (main API)
├── faiss_engine.py         # FAISS search engine class
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository

```sh
git clone <repo-url>
cd rag-fastapi-react-app
```

### 2. Python Backend Setup

- **Create and activate a virtual environment:**
  ```sh
  python -m venv venv
  # On Windows:
  venv\Scripts\activate
  # On Linux/Mac:
  source venv/bin/activate
  ```

- **Install dependencies:**
  ```sh
  pip install -r requirements.txt
  ```

- **Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract):**
  - Download and install from [here](https://github.com/tesseract-ocr/tesseract/wiki).
  - Add the install path (e.g., `C:\Program Files\Tesseract-OCR`) to your system PATH, or set it in `fastapi_app.py`:
    ```python
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ```

- **Install [Poppler](http://blog.alivate.com.au/poppler-windows/) for PDF processing (Windows only):**
  - Download and extract, then add the `bin` folder to your PATH.

- **Download and place your GGUF LLM models** in the paths specified in `fastapi_app.py` under `MODEL_PATHS`.

---

### 3. React Frontend Setup

```sh
cd rag-frontend
npm install
npm start
```
- The frontend will run at [http://localhost:3000](http://localhost:3000).

---

### 4. Start the Backend

From the project root:

```sh
uvicorn fastapi_app:app --reload
```
- The backend will run at [http://localhost:8000](http://localhost:8000).

---

## Usage

- **Upload PDFs** in the "File Upload" tab, specifying a domain.
- **Ask questions** in the "Query" tab, selecting a domain.
- **Generate MCQs** in the "MCQ Generator" tab, choosing domain, topic, number, and difficulty.
- **Download MCQs** as JSON for further use.

---

## Notes

- **Domains** are folders under `Embeddor/faiss_index/`. Each domain has its own FAISS index and metadata.
- **Models** must be downloaded separately and paths set in `fastapi_app.py`.
- **OCR** is used automatically for scanned PDFs.
- **If you move the project to another PC**, copy the entire folder, including `Embeddor/faiss_index/`, your models, and install all dependencies.

---

## Troubleshooting

- **Port in use:** Use `netstat -ano | findstr :8000` and `taskkill /PID <PID> /F` to free ports.
- **Tesseract not found:** Ensure it's installed and the path is set in your environment or code.
- **Poppler not found:** Ensure the `bin` folder is in your PATH.
- **No domains/models:** Make sure you have uploaded files and placed models in the correct paths.

---

## License

MIT License

---

## Credits

- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://react.dev/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
