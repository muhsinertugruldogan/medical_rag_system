Medical Multimodal RAG System - Docker Usage Guide

This project implements a multimodal Retrieval-Augmented Generation (RAG) system for medical report and chest X-ray question answering.

--------------------------------------------------
1. REQUIREMENTS
--------------------------------------------------

- Docker installed
- Project directory containing:
  - images/
  - chroma_db/
  - data/
- Dataset link: https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university?select=indiana_reports.csv
NOTE:
The dataset (~14GB) is NOT included inside the Docker image.
It must be mounted at runtime.

--------------------------------------------------
2. BUILD DOCKER IMAGE
--------------------------------------------------

Run the following command in the project root:

docker build -t medical-rag .

--------------------------------------------------
3. RUN CONTAINER
--------------------------------------------------

Run the container with volume mounting:

docker run -p 8000:8000 \
  -v $(pwd)/images:/app/images \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/data:/app/data \
  medical-rag

Explanation:
- images/      → chest X-ray images
- chroma_db/   → vector database
- data/        → metadata and manifest files

--------------------------------------------------
4. ACCESS API
--------------------------------------------------

Swagger UI:
http://localhost:8000/docs

Query Endpoint:
http://localhost:8000/query

--------------------------------------------------
5. EXAMPLE REQUEST
--------------------------------------------------

curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Is there evidence of acute cardiopulmonary abnormality?",
    "image_path": "images/images_normalized/1_IM-0001-4001.dcm.png"
  }'

--------------------------------------------------
6. EXAMPLE RESPONSE
--------------------------------------------------

{
  "answer": "Answer: No evidence of acute cardiopulmonary abnormality identified in any of the reports.\nEvidence summary: Multiple reports explicitly state the absence of acute cardiopulmonary abnormalities.\nConfidence: high",
  "sources": [
    {
      "uid": "3338",
      "impression": "No acute cardiopulmonary abnormality identified.",
      "rerank_score": 0.9999996423721313
    }
  ],
  "latency_ms": {
    "retrieval_ms": 30.24,
    "generation_ms": 613.84,
    "total_ms": 894.23
  }
}

--------------------------------------------------
7. USAGE
--------------------------------------------------

Endpoint:
POST /query

Text-only request:
{
  "question": "Is there cardiomegaly?"
}

Image + text request:
{
  "question": "What are the main radiographic findings in this chest X-ray?",
  "image_path": "images/images_normalized/1_IM-0001-4001.dcm.png"
}

--------------------------------------------------
8. PERFORMANCE
--------------------------------------------------

The system was evaluated on 100 queries.

Metrics:
- Retrieval time
- Generation time
- Total latency

Results saved in:
performance_results_api.csv

Typical latency:
- Retrieval: ~30–50 ms
- Reranking: ~60 ms
- Generation: ~400–700 ms

--------------------------------------------------
9. AUTOMATIC EVALUATION
--------------------------------------------------

Metrics:
- BLEU
- ROUGE
- METEOR
- BERTScore

Outputs:
- evaluation_metrics_results.csv
- evaluation_summary.json

Note:
Lexical scores may be low due to summary-style outputs.

--------------------------------------------------
10. LIMITATIONS
--------------------------------------------------

- Focused on radiology reports and chest X-rays
- Limited support for treatment-related questions
- Designed to avoid hallucinated medical outputs

--------------------------------------------------
Author:
Ertuğrul Doğan
