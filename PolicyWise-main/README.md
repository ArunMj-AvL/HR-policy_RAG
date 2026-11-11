# PolicyWise – Source-Grounded HR Policy Assistant

**Group 1 Project**

PolicyWise is an AI-powered HR Policy Assistant that helps employees find information from company HR policy documents using natural-language questions.

The system uses a Retrieval-Augmented Generation (RAG) approach to retrieve relevant policy passages and generate answers grounded in the provided documents. It also provides source citations, including PDF filenames and page numbers, to improve transparency and reliability.

## Project Objectives

- Simplify access to HR policy information.
- Allow employees to ask questions in natural language.
- Retrieve relevant information from multiple HR policy PDFs.
- Generate answers grounded in the provided policy documents.
- Display source filenames and page numbers.
- Avoid unsupported answers through a predefined fallback response.

## Key Features

- PDF-based HR policy knowledge base.
- Page-level text extraction and metadata preservation.
- Text chunking with overlapping sections.
- Top-3 relevant passage retrieval.
- Current policy prioritization over superseded versions.
- Grounded answer generation.
- Source document and page citations.
- Fallback handling for unsupported questions.
- Optional OpenAI-based answer generation.
- Streamlit-based user interface.

## Technology Stack

| Component | technology |
|---|---|
| PDF Processing | PyMuPDF / PyPDF |
| Text Chunking | Custom sentence-aware chunking |
| Embeddings | Sentence Transformers MiniLM |
| Vector Search | FAISS / NumPy |
| Retrieval | Semantic vector search using FAISS / NumPy |
| Answer Generation | Optional OpenAI API |
| Interface | Streamlit |

## Project Architecture

The system follows this workflow:

**HR Policy PDFs → Document Ingestion → Text Chunking → Retrieval → Answer Generation → Source-Cited Response**

Refer to [System Architecture](docs/architecture.md) for further details.

## Project Structure

```text
PolicyWise/
│
├── data/
│   ├── G1_policy_1.pdf
│   ├── G1_policy_2.pdf
│   ├── G1_policy_3.pdf
│   └── G1_policy_4.pdf
│
├── ingestion/
│   ├── __init__.py
│   ├── extract.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── index_store.py
│   ├── pipeline.py
│   └── ingest.py
│
├── src/
│   ├── retrieval.py
│   └── answer_generator.py
│
├── evaluation/
│   ├── test_questions.json
│   └── test_results.md
│
├── docs/
│   ├── architecture.md
│   └── Ingestion_Report.md
│
├── scripts/
│   └── sanity_check.py
│
├── tests/
│   └── test_ingestion.py
│
├── app.py
├── config.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```


## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Anit-K-T/PolicyWise.git
cd PolicyWise
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```
### 5. Generate the policy index

```bash
python -m ingestion.ingest --pdf-dir data --index-dir index
```

### 6. Run the application

```bash
streamlit run app.py
```

Open the local Streamlit URL displayed in the terminal.

## Answer Generation

PolicyWise supports two answer-generation modes:

**Default Mode**
- Uses retrieved policy passages as evidence.
- Displays relevant policy excerpts.
- Does not require an API key.

**Optional OpenAI Mode**
- Uses an OpenAI model to generate a synthesized response.
- Answers are instructed to remain grounded in retrieved policy excerpts.
- Requires an OpenAI API key configured through environment variables.

## Fallback Handling

When the requested information cannot be supported by the provided policy documents, the system uses the following response:

`Not specified in the provided policies.`

## Evaluation and Testing

The project includes an evaluation dataset and results template.

- `evaluation/test_questions.json` – Contains questions used to test the system.
- `evaluation/test_results.md` – Records expected results, actual results and test status.

The system will be evaluated based on:

1. Answer accuracy.
2. Retrieval relevance.
3. Source citation correctness.
4. Unsupported-question handling.
5. Current and superseded policy prioritization.

## Team Contributions

| Team Member | Responsibility |
|---|---|
| Anit | Team Lead, Streamlit frontend, integration, testing and documentation |
| Arun | PDF ingestion, text extraction and chunking |
| Aamir | RAG retrieval and answer generation |
| Niranjan | Test-question preparation and evaluation support |

## Reliability and Safety

- Answers should be based on the provided HR policy documents.
- Source filenames and page references should be preserved.
- Unsupported information should not be fabricated.
- Superseded policy information should be distinguished from current policy.
- API keys and other credentials must not be committed to GitHub.

## Project Status

PolicyWise is under development. Document ingestion, retrieval, answer generation and frontend components are being integrated and tested.

## Repository

[PolicyWise – GitHub Repository](https://github.com/Anit-K-T/PolicyWise)
