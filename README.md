# 📄 Document Analyzer

An AI-powered **Universal Document Intelligence Tool** built with **Streamlit** that helps you analyze PDF documents through:

- **Named Entity Recognition (NER)** with spaCy  
- **Semantic clause search** with sentence-transformer embeddings  
- **Structured export** of extracted entities (CSV/Excel)  
- **Raw text inspection** for debugging extraction quality  

---

## ✨ Features

- 📂 **PDF Text Extraction**
  - Extracts text from uploaded PDF files using `pdfplumber`
  - Supports multi-page workflows
  - Includes cleaning and normalization utilities

- 🏷️ **Entity Analysis (NER)**
  - Detects business-relevant entities (e.g., `PERSON`, `ORG`, `MONEY`, `DATE`, `PRODUCT`, `LOC`)
  - Interactive label filtering
  - Entity summary + table view
  - displaCy-powered visual entity rendering

- 🔍 **Semantic Search**
  - Split document into configurable overlapping text segments
  - Encode segments using `sentence-transformers`
  - Find top-k semantically similar clauses to a user query using cosine similarity
  - Export similarity results to CSV

- 📝 **Raw Text Explorer**
  - View full cleaned text
  - View per-page extracted text
  - Useful for OCR/extraction validation and debugging

- ⬇️ **One-click Export**
  - Export extracted entities to:
    - CSV
    - Excel (`.xlsx`)
  - Export semantic search results to CSV

---

## 🧱 Project Structure

```text
Document_Analyzer/
├── app.py                  # Main Streamlit app
├── requirements.txt
├── .streamlit/             # Streamlit config (if present)
└── src/
    ├── __init__.py
    ├── extractor.py        # PDF text extraction + text stats
    ├── nlp_pipeline.py     # spaCy loading + entity filtering + summary/dataframe
    ├── similarity.py       # Segmenting + embeddings + similarity ranking
    └── utils.py            # Export helpers, truncation, highlighting
```

---

## ⚙️ Tech Stack

- **Frontend/UI:** Streamlit  
- **NLP:** spaCy, spacy-streamlit  
- **Embeddings & Semantic Search:** sentence-transformers, transformers, torch  
- **Data Processing:** pandas, scikit-learn, numpy  
- **PDF Parsing:** pdfplumber  
- **Export:** openpyxl (Excel output)

---

## 📦 Installation

### 1) Clone the repository

```bash
git clone https://github.com/swayamshah01/Document_Analyzer.git
cd Document_Analyzer
```

### 2) Create and activate virtual environment (recommended)

```bash
python -m venv .venv
```

- **Windows (PowerShell):**
  ```bash
  .venv\Scripts\Activate.ps1
  ```
- **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Download spaCy English model(s)

```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
```

> If you only want the default model, install `en_core_web_sm` at minimum.

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).

---

## 🧠 How It Works

1. **Upload PDF** in the app  
2. Text is extracted and cleaned  
3. spaCy processes the text for entities  
4. Entities are filtered by selected labels and displayed in:
   - summary pills
   - displaCy visualization
   - tabular format  
5. Text is split into overlapping segments  
6. Segments are embedded using the selected sentence-transformer model  
7. User query is embedded and compared against segments via cosine similarity  
8. Top matches are shown with similarity scores and downloadable output

---

## 🎛️ Configurable Settings (UI)

- **NER model:**  
  - `en_core_web_sm`  
  - `en_core_web_md`  
  - `en_core_web_lg`

- **Similarity model:**  
  - `all-MiniLM-L6-v2`  
  - `all-mpnet-base-v2`  
  - `paraphrase-MiniLM-L6-v2`

- **Segmentation controls:**  
  - Segment size (words)  
  - Segment overlap  
  - Top-k similarity results

- **Entity label filters:** choose which NER categories to include

---

## 📌 Example Use Cases

- Insurance policy analysis  
- Contract clause discovery  
- Compliance and legal review support  
- Financial document intelligence  
- Quick semantic lookup in long PDFs

---

## ⚠️ Limitations

- Best performance on **text-selectable PDFs**  
- Scanned/image-only PDFs may require OCR before upload  
- Very large PDFs may increase processing time and memory usage  
- NER/similarity quality depends on selected models and text quality

---

## 🚀 Future Improvements (Suggested)

- OCR pipeline integration (Tesseract/EasyOCR)  
- Batch processing for multiple PDFs  
- Entity linking and custom domain ontologies  
- Persistent vector index for cross-document search  
- Authentication + deployment-ready multi-user mode

---

## 🤝 Contributing

Contributions are welcome.  
If you’d like to improve extraction quality, model selection, or UI/UX:

1. Fork the repo  
2. Create a feature branch  
3. Commit your changes  
4. Open a pull request

---

## 📄 License

Add your preferred license here (e.g., MIT, Apache-2.0).

Example (MIT):
```text
MIT License © 2026 swayamshah01
```

---

## 👤 Author

**Swayam Shah**  
GitHub: [@swayamshah01](https://github.com/swayamshah01)
