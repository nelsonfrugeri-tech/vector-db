from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document


def chunk_page_text(
    text: str, chunk_size: int = 300, chunk_overlap: int = 50
) -> list[str]:
    parser = SimpleNodeParser.from_defaults(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    nodes = parser.get_nodes_from_documents([Document(text=text)])
    return [node.text for node in nodes]
