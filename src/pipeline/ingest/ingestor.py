import uuid
from src.database.qdrant.driver import QdrantDriver
from openai import OpenAI


class Ingestor:
    def __init__(self, pdf_path: str, collection_name: str, embedding_dim: int = 384):
        self.pdf_path = pdf_path
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.qdrant_driver = QdrantDriver(host="localhost", port=6333)
        self.qdrant_driver.create_collection(
            collection_name=self.collection_name, size=self.embedding_dim
        )
        self.client = OpenAI()

    def ingest(
        self,
        chunks: list[dict],
        page_number: int,
        chapter: str,
        section: str,
        tags: list[str],
    ) -> None:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=chunks,
            dimensions=self.embedding_dim,
            encoding_format="float",
        )

        for idx, (chunk, embedding_obj) in enumerate(zip(chunks, response.data)):
            self.qdrant_driver.insert_point(
                collection_name=self.collection_name,
                point={
                    "id": str(uuid.uuid4()),
                    "vector": embedding_obj.embedding,
                    "payload": {
                        "text": chunk,
                        "source": str(self.pdf_path),
                        "page": page_number,
                        "chunk_index": idx,
                        "chapter": chapter,
                        "section": section,
                        "tags": tags,
                    },
                },
            )
