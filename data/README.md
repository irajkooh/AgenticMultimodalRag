---
dataset: AgenticMultiModalRag_dataset
tags:
  - multimodal
  - rag
  - agentic
  - chromadb
  - document-retrieval
  - table-extraction
  - image-processing
  - sql
  - huggingface-dataset
description: |
  This dataset stores all persistent data for the AgenticMultimodalRag application, including indexed documents, tables, and images. It is used to restore the state of the Hugging Face Space and ensure data persistence across redeployments. All user-uploaded and processed files are synced here automatically by the app backend and deployment scripts.

  - All files in `data/`, `vectorstore/`, and `data/tables/` are managed here.
  - Binary files (PDF, PNG, DOCX, etc.) are excluded from the Space repo and only live in this dataset.
  - The dataset is updated automatically on every deploy or data change.

  For more details, see the main app repo: https://github.com/irajkooh/AgenticMultimodalRag

license: apache-2.0
dataset_type: "multimodal, persistent, app-backend"
---

# AgenticMultiModalRag Dataset

This dataset is used by the [AgenticMultiModalRag](https://github.com/irajkooh/AgenticMultimodalRag) app to persistently store all user and system data, including:

- Indexed documents (PDF, DOCX, TXT, etc.)
- Extracted tables and images
- ChromaDB vectorstore files
- Any other persistent state required for robust RAG workflows

**Do not delete or modify files here unless you know what you are doing.**

For questions, see the app repo or open an issue on GitHub.