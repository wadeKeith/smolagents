import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping

import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from unidecode import unidecode


class CompanyRAGStore:
    """
    Lightweight wrapper around a persistent Chroma vector database that stores
    documents grouped by company name, while also keeping raw snapshots on disk.
    """

    def __init__(
        self,
        persist_directory: str = "rag_vector_db",
        corpus_directory: str = "rag_corpus",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 600,
        chunk_overlap: int = 120,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.corpus_directory = Path(corpus_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.corpus_directory.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self._collections: dict[str, chromadb.api.models.Collection.Collection] = {}
        self._text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained("thenlper/gte-small"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", "。", "！", "?", " ", ""],
        )

    # ------------------------------------------------------------------ #
    # Collection helpers
    # ------------------------------------------------------------------ #
    def _slugify(self, value: str) -> str:
        text = unicodedata.normalize("NFKC", value)
        translit = unidecode(text)  # 例如 "中文有限公司" -> "Zhong Wen You Xian Gong Si"

        ascii_slug = re.sub(r"[^a-zA-Z0-9]+", "-", translit).strip("-").lower()
        if ascii_slug:
            return ascii_slug[:128]

        # 兜底：对原始值做稳定哈希，得到合法且确定的标识符
        digest = hashlib.sha256(unicodedata.normalize("NFC", value).encode("utf-8")).hexdigest()[:32]
        return f"h-{digest}"

    def _collection_name(self, company_name: str) -> str:
        return f"company_{self._slugify(company_name)}"

    def _get_collection(self, company_name: str):
        collection_name = self._collection_name(company_name)
        if collection_name not in self._collections:
            self._collections[collection_name] = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"company_name": company_name},
                embedding_function=self._embedding_fn,
            )
        return self._collections[collection_name]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_documents(
        self,
        company_name: str,
        contents: Iterable[str],
        metadata: Mapping[str, str] | None = None,
    ) -> int:
        """
        Add raw documents for a company. Returns the number of vector chunks stored.
        """
        metadata = metadata or {}
        timestamp = time.strftime("%Y%m%d%H%M%S")
        collection = self._get_collection(company_name)
        slug = self._slugify(company_name)

        total_chunks = 0
        for index, text in enumerate(contents):
            if not text or not text.strip():
                continue
            normalized_text = text.strip()
            doc_hash = hashlib.md5(normalized_text.encode("utf-8")).hexdigest()
            raw_path = (
                self.corpus_directory / slug / f"{timestamp}_{index}_{doc_hash[:8]}.json"
            )
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_payload = {
                "company_name": company_name,
                "content": normalized_text,
                "metadata": dict(metadata),
                "stored_at": timestamp,
            }
            raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            chunks = self._text_splitter.split_text(normalized_text)
            if not chunks:
                continue

            chunk_ids = [f"{slug}_{doc_hash}_{i}" for i in range(len(chunks))]
            chunk_metadatas = [
                dict(metadata) | {"chunk_index": i, "doc_hash": doc_hash, "raw_path": str(raw_path)}
                for i in range(len(chunks))
            ]
            # Upsert so reruns refresh the vector store without duplication errors.
            collection.upsert(ids=chunk_ids, documents=chunks, metadatas=chunk_metadatas)
            total_chunks += len(chunks)

        return total_chunks

    def query(self, company_name: str, query: str, top_k: int = 5) -> list[dict[str, str]]:
        """
        Retrieve top matching chunks for a company.
        Returns a list of dictionaries containing chunk content and metadata.
        """
        if top_k <= 0:
            return []
        collection = self._get_collection(company_name)
        if collection.count() == 0:
            return []
        results = collection.query(query_texts=[query], n_results=top_k)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        return [
            {
                "content": doc,
                "metadata": meta or {},
            }
            for doc, meta in zip(documents, metadatas)
        ]
