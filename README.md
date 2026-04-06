Medical Multimodal RAG System

This repository implements a multimodal Retrieval-Augmented Generation
(RAG) system for medical report and chest X-ray question answering using
the Indiana University dataset.

Repository Structure Main folders:

retrieval/: embedding, vector DB, reranking generation/: answer
generation and image understanding scripts/: evaluation, preprocessing,
and test scripts data/: manifest files and structured metadata images/:
chest X-ray images chroma_db/: persisted vector database

Main Models and Components Embedding model:
hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 Vector
database: ChromaDB Reranker: ncbi/MedCPT-Cross-Encoder Generator:
Qwen/Qwen2.5-3B-Instruct API framework: FastAPI

Docker Deployment

Docker is the recommended way to run the system.

Since the dataset is large (\~14GB), runtime folders are NOT included in
the image. Instead, they are mounted at runtime:

images/ chroma_db/ data/

3.1 Build the Docker image

docker build -t medical-rag .

3.2 Run the container

docker run -p 8000:8000 -v ( p w d ) / i m a g e s : / a p p / i m a g e
s − v (pwd)/chroma_db:/app/chroma_db -v \$(pwd)/data:/app/data
medical-rag

3.3 Access the API

Swagger UI: http://localhost:8000/docs

Query endpoint: http://localhost:8000/query

3.4 Example API request

curl -X POST \"http://localhost:8000/query \" -H \"Content-Type:
application/json\" -d \'{ \"question\": \"Is there evidence of acute
cardiopulmonary abnormality?\", \"image_path\":
\"images/images_normalized/1_IM-0001-4001.dcm.png\" }\'

3.5 Example response

{ \"answer\": \"Answer: No evidence of acute cardiopulmonary abnormality
identified in any of the reports.\\nEvidence summary: Multiple reports
explicitly state the absence of acute cardiopulmonary
abnormalities.\\nConfidence: high\", \"sources\": \[ { \"uid\":
\"3338\", \"impression\": \"No acute cardiopulmonary abnormality
identified.\", \"from_text\": true, \"from_image\": false,
\"text_rank\": 5, \"image_rank\": null, \"rerank_score\":
0.9999996423721313 } \], \"latency_ms\": { \"retrieval_ms\": 30.24,
\"generation_ms\": 613.84, \"total_ms\": 894.23 } }

Usage Endpoint: POST /query

Text-only request

{ \"question\": \"Is there cardiomegaly?\" }

Image + text request

{ \"question\": \"What are the main radiographic findings in this chest
X-ray?\", \"image_path\":
\"images/images_normalized/1_IM-0001-4001.dcm.png\" }

Response fields answer: generated grounded answer sources: retrieved and
reranked sources latency_ms: latency breakdown

Performance Evaluation The system includes an API-based evaluation
pipeline.

Measures:

retrieval time generation time total latency

Setup:

100 mixed queries text-only + multimodal short / medium / long

Output:

performance_results_api.csv

Latency:

Retrieval: \~30--50 ms Reranking: \~60 ms Generation: \~400--700 ms

Automatic Evaluation Metrics Computed metrics:

BLEU ROUGE METEOR BERTScore

Outputs:

evaluation_metrics_results.csv evaluation_summary.json

Note: Scores may be low due to summary-style outputs vs full reports.

Limitations Focused on radiology reports and chest X-rays Limited
support for treatment or clinical management Avoids hallucinated medical
recommendations

Author

Ertuğrul Doğan
