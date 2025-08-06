# 🧠 Semantic Spotter - Retrieval-Augmented QA using Open-Source Models
Llamaindex | RAG | RAGAS | Evaluation metrics | Open sources models 

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **open-source models** from Hugging Face to semantically answer questions from large insurance documents (PDFs). Evaluation of the system is done using **RAGAS** metrics.

---

## 🚀 Objectives

- Build a fully open-source RAG pipeline.
- Replace OpenAI APIs with lightweight HuggingFace models.
- Enable semantic question-answering over long insurance PDFs.
- Evaluate output quality using RAGAS metrics:
  - Faithfulness
  - Answer relevancy
  - Context precision

---

## 🧱 Tech Stack

| Layer          | Tool/Model Used                                  |
|----------------|--------------------------------------------------|
| Chunking       | `SimpleNodeParser` from LlamaIndex               |
| Embedding      | `BAAI/bge-small-en-v1.5`                         |
| Retriever      | Top-k semantic retrieval                         |
| LLM            | `tiiuae/falcon-rw-1b` (via HuggingFaceLLM)       |
| Evaluation     | `RAGAS` (faithfulness, relevancy, context score) |
| Environment    | Google Colab / Local (GPU recommended)           |

---

## 🧠 Architecture Overview

1. **Document Chunking**  
   → Break PDFs into semantic chunks using LlamaIndex.

2. **Embedding Generation**  
   → Use `BAAI/bge-small-en-v1.5` to generate embeddings.

3. **Query Embedding + Retrieval**  
   → Embed query, compute semantic similarity, and fetch top-k nodes.

4. **LLM Response Generation**  
   → Use `tiiuae/falcon-rw-1b` to answer the query using retrieved context.

5. **Evaluation (Optional)**  
   → Use `RAGAS` to evaluate the output with ground-truth answers.

---

## 🔧 Setup Instructions

### ✅ Requirements

- Python >= 3.9
- GPU (optional, recommended)
- Install dependencies:
  ```bash
  pip install llama-index
  pip install ragas
  pip install datasets transformers accelerate

