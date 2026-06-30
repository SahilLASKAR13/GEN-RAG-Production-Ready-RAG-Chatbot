from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.rag_pipeline import RAGPipeline

# Singleton instances used across the application

ingestion_pipeline = IngestionPipeline()
rag_pipeline = RAGPipeline()