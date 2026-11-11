# PolicyWise – System Architecture

## 1. Project Overview

PolicyWise is a Source-Grounded HR Policy Assistant that uses Retrieval-Augmented Generation (RAG) to answer employee questions based on provided HR policy documents.

The system retrieves relevant information from HR policy PDFs and generates answers with source document names and page references.

## 2. System Workflow

The system follows these stages:

1. **User Input:** Employees enter HR-related questions through the Streamlit interface.
2. **Document Ingestion:** HR policy PDFs are loaded and their text is extracted using PyMuPDF.
3. **Text Chunking:** Extracted text is divided into smaller overlapping chunks while preserving document and page metadata.
4. **Retrieval:** Relevant policy chunks are identified using the retrieval module.
5. **Answer Generation:** The answer generation module produces responses based on retrieved policy evidence.
6. **Source Citation:** The system displays supporting PDF filenames and page numbers.
7. **Fallback Handling:** If the requested information is unavailable, the system responds with: "Not specified in the provided policies."

## 3. Main Components

| Component | Responsibility |
|---|---|
| Streamlit Frontend | Accepts questions and displays answers |
| PDF Ingestion | Extracts and processes HR policy documents |
| Text Chunking | Divides documents into manageable sections |
| Retrieval Module | Finds the most relevant policy sections |
| Answer Generation | Generates evidence-based responses |
| Citation Handling | Displays source filenames and page numbers |

## 4. Technology Stack

- Programming Language: Python
- Frontend: Streamlit
- PDF Processing: PyPDF with pdfplumber fallback
- Text Chunking: Sentence-aware chunking with overlap
- Embeddings: Sentence Transformers MiniLM
- Offline Embeddings: Hashing-based embedding fallback
- Vector Database: FAISS with NumPy fallback
- Similarity Search: Cosine similarity
- Retrieval: Top-3 semantic vector retrieval
- Answer Generation: OpenAI API with offline evidence fallback
- Version Handling: Current and superseded policy metadata
- Version Control: Git and GitHub

## 5. Architecture Diagram

```mermaid
flowchart TD
    A[HR Policy PDFs] --> B[PDF Extraction]
    B --> C[Page Metadata]
    C --> D[Text Chunking]
    D --> E[Overlapping Chunks]
    E --> F[Embedding Generation]
    F --> G[FAISS / NumPy Vector Index]

    H[User Question] --> I[Query Embedding]
    I --> J[Vector Retrieval]
    G --> J
    J --> K[Policy Version Handling]
    K --> L[Top-3 Relevant Passages]
    L --> M[Answer Generation]
    M --> N[Source Filename and Page]
    N --> O[Streamlit Response]

## 6. Reliability and Safety

- Answers must be grounded in the provided HR policy documents.
- Source references should include the PDF filename and page number.
- Unsupported questions should receive the specified fallback response.
- Superseded policy information should not be presented as current policy.
- API credentials must be stored securely using environment variables.

## 7. Project Architecture Status

The architecture describes the planned integration of document ingestion, retrieval, answer generation and the Streamlit interface. Implementation details may be updated as the modules are integrated and tested.