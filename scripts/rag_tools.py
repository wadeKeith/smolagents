from __future__ import annotations

from typing import Any

from smolagents import Tool

from scripts.company_rag_store import CompanyRAGStore


class CompanyRAGRetrieveTool(Tool):
    name = "company_rag_retrieve"
    description = (
        "从本地向量数据库中检索与目标企业相关的已缓存资料。"
        "在联网搜索前先调用此工具，可以减少重复抓取。"
    )
    inputs = {
        "company_name": {
            "type": "string",
            "description": "企业名称，用于定位对应的本地向量库。",
        },
        "query": {
            "type": "string",
            "description": "检索关键词或主题描述，使用陈述句更有利于匹配。",
        },
        "top_k": {
            "type": "integer",
            "description": "返回的向量匹配数量，默认 5。",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, rag_store: CompanyRAGStore, company_name_resolver=None):
        super().__init__()
        self._store = rag_store
        self._company_name_resolver = company_name_resolver

    def forward(self, company_name: str, query: str, top_k: int | None = 5) -> str:
        target_company = (company_name or "").strip()
        if not target_company and callable(self._company_name_resolver):
            target_company = self._company_name_resolver() or ""
        if not target_company:
            return "company_name 不能为空。"
        if not query.strip():
            return "query 不能为空。"
        results = self._store.query(company_name=target_company, query=query, top_k=top_k or 5)
        if not results:
            return "未在本地知识库中找到相关内容，可继续进行网页搜索。"
        formatted_chunks = []
        for idx, item in enumerate(results, start=1):
            meta = item.get("metadata", {})
            formatted_chunks.append(
                f"[RAG-{idx}] chunk_index={meta.get('chunk_index', 'NA')} "
                f"doc_hash={meta.get('doc_hash', 'NA')} 来源文件: {meta.get('raw_path', 'NA')}\n"
                f"{item.get('content', '').strip()}"
            )
        return "\n\n".join(formatted_chunks)


class CompanyRAGIngestTool(Tool):
    name = "company_rag_ingest"
    description = (
        "将新获取的网页或新闻内容写入本地企业向量数据库，"
        "以便后续无需重复抓取即可快速检索。"
    )
    inputs = {
        "company_name": {
            "type": "string",
            "description": "企业名称，用于确定写入的向量库目录。",
        },
        "content": {
            "type": "string",
            "description": "要写入的文本内容，建议是一段较完整的网页正文或新闻摘要。",
        },
        "source": {
            "type": "string",
            "description": "信息来源（如 URL、媒体名称），便于后续追溯。",
            "nullable": True,
        },
        "category": {
            "type": "string",
            "description": "内容类别（如官网公告、主流媒体、地方媒体、舆情等），用于元数据标注。",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, rag_store: CompanyRAGStore, company_name_resolver=None):
        super().__init__()
        self._store = rag_store
        self._company_name_resolver = company_name_resolver

    def forward(self, company_name: str, content: str, source: str | None = None, category: str | None = None) -> str:
        target_company = (company_name or "").strip()
        if not target_company and callable(self._company_name_resolver):
            target_company = self._company_name_resolver() or ""
        if not target_company:
            return "company_name 不能为空。"
        if not content.strip():
            return "content 不能为空。"
        metadata: dict[str, Any] = {}
        if source:
            metadata["source"] = source
        if category:
            metadata["category"] = category
        stored_chunks = self._store.add_documents(
            company_name=target_company,
            contents=[content],
            metadata=metadata,
        )
        return f"已入库，生成 {stored_chunks} 个向量分片。"
