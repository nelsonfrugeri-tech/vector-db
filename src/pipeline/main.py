import fitz
import re
from pathlib import Path
from src.pipeline.extract.chunk import chunk_page_text
from src.pipeline.ingest.ingestor import Ingestor
from src.model.openai import OpenAIModel  # Importa o modelo OpenAI


def clean_page_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()

        # Ignora linhas vazias
        if not line:
            continue

        # Remove cabeçalhos e rodapés comuns
        if re.match(
            r"^(E-book gerado| Sumário |Página \d+|©|Todos os direitos|Sumário)", line
        ):
            continue

        # Ignora nomes de capítulo em caixa alta com poucas palavras
        if line.isupper() and len(line.split()) <= 4:
            continue

        # Ignora linhas muito curtas com símbolos
        if len(line) < 10 and re.match(r"^[\d\-–\.\s]+$", line):
            continue

        cleaned.append(line)

    return " ".join(cleaned)


def extract_metadata(openai_model, page_text: str) -> dict:
    prompt = (
        "Dado o texto de uma página de um livro, extraia:\n"
        "- O nome do capítulo principal (campo: chapter) caso exista, se não, tente entender o assunto da página e atribua um nome\n"
        "- O nome da seção principal, se existir (campo: section) caso não exista, tente entender o assunto da página e atribua um nome\n"
        "- Uma lista curta de tags resumindo os principais assuntos presentes (campo: tags)\n"
        "Responda APENAS com um objeto JSON, por exemplo:\n"
        "{\n"
        '  "chapter": "Capítulo 3 - Elasticidade",\n'
        '  "section": "Elasticidade em sistemas reativos",\n'
        '  "tags": ["elasticidade", "reatividade", "sistemas"]\n'
        "}\n"
        "Texto:\n"
        f"{page_text}\n"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Você é um extrator automático de metadados de páginas de livros. "
                "Sempre responda exclusivamente com um JSON conforme o exemplo."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return openai_model.chat(messages)


def main():
    pdf_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "file"
        / "source"
        / "sistemas_reativos.pdf"
    )
    max_pages = 320
    chunk_size = 500
    chunk_overlap = 120
    start_chunking = False

    doc = fitz.open(pdf_path)

    ingestor = Ingestor(
        pdf_path=pdf_path, collection_name="sistemas_reativos_v3", embedding_dim=384
    )
    openai_model = OpenAIModel()

    for page_number in range(min(max_pages, len(doc))):
        page = doc[page_number].get_text()
        if not start_chunking:
            if re.search(r"cap[ií]tulo\s+1", page, re.IGNORECASE):
                start_chunking = True
            else:
                continue

        page_text = clean_page_text(page)
        if not page_text.strip():
            continue

        # Extrai metadados com modelo OpenAI
        metadata = extract_metadata(openai_model, page_text)

        chunks = chunk_page_text(
            text=page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        if not chunks:
            continue

        ingestor.ingest(
            chunks=chunks,
            page_number=page_number,
            chapter=metadata.get("chapter"),
            section=metadata.get("section"),
            tags=metadata.get("tags"),
        )

    print("Ingestão finalizada com sucesso.")


if __name__ == "__main__":
    main()
