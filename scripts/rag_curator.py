from __future__ import annotations

import datetime as _dt
from typing import Optional

from smolagents.models import MessageRole


class RAGCurator:
    """
    Lightweight helper that distills newly fetched content before pushing it to the RAG store.

    Heuristics:
    - Very short snippets are stored verbatim with metadata.
    - Longer content is summarised with the provided model (if any) to remove noise, deduplicate against existing notes,
      and keep explicit references to source + category.
    """

    def __init__(self, model, max_tokens: int = 512):
        self.model = model
        self.max_tokens = max_tokens

    def curate(
        self,
        *,
        company_name: str,
        location_hint: str,
        source: str,
        category: str,
        new_text: str,
        existing_context: str,
    ) -> Optional[str]:
        text = (new_text or "").strip()
        if not text:
            return None

        timestamp = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        header = f"[来源] {source} | [类别] {category} | [时间] {timestamp}"

        # Short snippets - keep as-is with header.
        if len(text) <= 320:
            return f"{header}\n{text}"

        if self.model is None:
            return f"{header}\n{text}"

        system_text = (
            "你是企业背景调查的资料整理助手。"
            "请基于最新获取的原始文本，总结对企业尽调最有价值的要点，避免与已有内容重复，"
            "必要时指出矛盾或需跟进事项。输出中文要点列表，适合写入知识库。"
        )
        user_text = (
            f"企业名称：{company_name}\n重点地区：{location_hint}\n来源类别：{category}\n原始渠道：{source}\n\n"
            f"已有资料节选（可能为空）：\n{existing_context or '（暂无）'}\n\n"
            f"新增原文：\n{text}\n\n"
            "请输出不超过 6 条要点，每条以“- ”开头，包含事实、数字或事件，并指明该要点来自本次文本；"
            "若与已有资料重复或矛盾，请标记【重复】或【需核实】并说明原因。"
        )

        messages = [
            {"role": MessageRole.SYSTEM, "content": [{"type": "text", "text": system_text}]},
            {"role": MessageRole.USER, "content": [{"type": "text", "text": user_text}]},
        ]
        try:
            response = self.model.generate(messages, max_tokens=self.max_tokens)
            content = response.content if hasattr(response, "content") else response
        except Exception:
            content = text

        content = (content or "").strip()
        if not content:
            return None

        # Prevent enormous generations from polluting RAG.
        if len(content) > 2000:
            content = content[:2000] + "..."

        return f"{header}\n{content}"

    def update_playbook(
        self,
        *,
        company_name: str,
        location_hint: str,
        source: str,
        category: str,
        existing_playbook: str,
        new_entry: str,
    ) -> str:
        existing = (existing_playbook or "").strip()
        addition = (new_entry or "").strip()
        if not addition:
            return existing_playbook

        if self.model is None:
            if existing:
                return f"{existing}\n\n{addition}"
            return addition

        system_text = (
            "你是一名企业尽调知识库的维护者。目标是维护一份精简、结构化的 playbook，"
            "聚焦关键事实、风险、机会和关注点；需随时间迭代，避免重复冗余，保留出处提示。"
        )
        user_text = (
            f"企业名称：{company_name}\n重点地区：{location_hint}\n来源类别：{category}\n原始渠道：{source}\n\n"
            f"现有 Playbook：\n{existing or '（暂无）'}\n\n"
            f"新增笔记：\n{addition}\n\n"
            "请输出最新的 Playbook，使用中文，多级条目或分段均可：\n"
            "- 保留关键信息、时间、来源；\n"
            "- 去掉重复和价值较低的内容；\n"
            "- 对潜在矛盾或待核实事项标注【需核实】；\n"
            "- 控制在 800 字以内。"
        )

        messages = [
            {"role": MessageRole.SYSTEM, "content": [{"type": "text", "text": system_text}]},
            {"role": MessageRole.USER, "content": [{"type": "text", "text": user_text}]},
        ]
        try:
            response = self.model.generate(messages, max_tokens=self.max_tokens)
            content = response.content if hasattr(response, "content") else response
        except Exception:
            content = f"{existing}\n\n{addition}" if existing else addition

        content = (content or "").strip()
        if len(content) > 4000:
            content = content[:4000] + "..."
        return content
