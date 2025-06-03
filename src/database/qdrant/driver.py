from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class QdrantDriver:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)

    def create_collection(
        self,
        collection_name: str,
        size: int = 384,
        distance: Distance = Distance.COSINE,
    ):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=size, distance=distance),
        )

    def insert_point(self, collection_name: str, point: dict):
        self.client.upsert(collection_name=collection_name, points=[point])

    def insert_batch(
        self, collection_name: str, points: list[dict], batch_size: int = 100
    ):
        try:
            for i in range(0, len(points), batch_size):
                batch_points = points[i : i + batch_size]
                self.client.upsert(collection_name=collection_name, points=batch_points)
        except Exception as e:
            print(f"Erro ao inserir batch: {e}")
