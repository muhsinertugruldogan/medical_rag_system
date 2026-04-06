# Medical Multimodal RAG System

This repository implements a multimodal Retrieval-Augmented Generation (RAG) system for medical report and chest X-ray question answering using the Indiana University chest X-ray dataset.

The system supports:
- text-only queries
- image + text queries
- retrieval over radiology reports and chest X-ray images
- cross-encoder reranking
- grounded answer generation through an API

---

# 1. System Overview

The pipeline consists of four main stages:

## 1.1 Retrieval
Relevant radiology reports and images are retrieved using dense embeddings and ChromaDB.

## 1.2 Reranking
Retrieved candidates are reranked with a medical cross-encoder model:
- `ncbi/MedCPT-Cross-Encoder`

## 1.3 Generation
The final answer is generated using:
- `Qwen/Qwen2.5-3B-Instruct`

The generator is grounded on retrieved radiology reports.

## 1.4 Evaluation
The repository includes:
- performance evaluation over 100 mixed queries
- automatic metric evaluation against report references

---

# 2. Repository Structure

```text
medical-rag/
├── app.py
├── retrieval/
├── generation/
├── scripts/
├── data/
├── images/
├── chroma_db/
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
