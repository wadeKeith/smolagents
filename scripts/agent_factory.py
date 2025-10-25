from __future__ import annotations

import os
from typing import Any

from scripts.company_rag_store import CompanyRAGStore
from scripts.curation_monitor import log_curation_event
from scripts.rag_curator import RAGCurator
from scripts.rag_tools import CompanyRAGIngestTool, CompanyRAGRetrieveTool
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer
from smolagents import CodeAgent, GoogleSearchTool, OpenAIServerModel, ToolCallingAgent
from smolagents.cli import load_model
from smolagents.models import MessageRole

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
)

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": USER_AGENT},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def _build_remote_model(
    model_id: str,
    api_key_envs: list[str],
    api_base_envs: list[str],
    default_api_base: str,
) -> OpenAIServerModel:
    api_key = None
    for env in api_key_envs:
        if env and os.getenv(env):
            api_key = os.getenv(env)
            break
    if api_key is None:
        raise ValueError(f"Missing API key for model {model_id}. Checked env vars: {api_key_envs}")

    api_base = None
    for env in api_base_envs:
        if env and os.getenv(env):
            api_base = os.getenv(env)
            break
    if api_base is None:
        api_base = default_api_base

    return OpenAIServerModel(api_key=api_key, api_base=api_base, model_id=model_id)


def create_agent(
    search_max_steps=20,
    critic_max_steps=20,
    manage_max_steps=12,
    company_context: dict[str, Any] | None = None,
):
    context = (company_context or {}).copy()
    text_limit = 100000
    rag_store = CompanyRAGStore()

    manager_model = _build_remote_model(
        model_id=os.getenv("MANAGER_MODEL_ID", "gpt-4o"),
        api_key_envs=["MANAGER_API_KEY", "OPENAI_API_KEY"],
        api_base_envs=["MANAGER_API_BASE", "OPENAI_API_BASE"],
        default_api_base="https://aihubmix.com/v1",
    )
    if manager_model is None:
        raise ValueError(
            "无法初始化 manager 模型：请设置 MANAGER_API_KEY/OPENAI_API_KEY，或通过 CLI 提供可用的 model_type/model_id。"
        )

    search_model = _build_remote_model(
        model_id=os.getenv("SEARCH_MODEL_ID", "gemini-2.5-flash"),
        api_key_envs=["SEARCH_API_KEY", "OPENAI_API_KEY"],
        api_base_envs=["SEARCH_API_BASE", "OPENAI_API_BASE"],
        default_api_base="https://aihubmix.com/v1",
    )
    if search_model is None:
        search_model = manager_model

    critic_model = _build_remote_model(
        model_id=os.getenv("CRITIC_MODEL_ID", "claude-haiku-4-5"),
        api_key_envs=["CRITIC_API_KEY", "FIREWORKS_API_KEY", "OPENAI_API_KEY"],
        api_base_envs=["CRITIC_API_BASE", "FIREWORKS_API_BASE", "OPENAI_API_BASE"],
        default_api_base="https://aihubmix.com/v1",
    )
    if critic_model is None:
        critic_model = manager_model

    curator_model = _build_remote_model(
        model_id=os.getenv("CURATOR_MODEL_ID", "gpt-4o-mini"),
        api_key_envs=["CURATOR_API_KEY", "OPENAI_API_KEY"],
        api_base_envs=["CURATOR_API_BASE", "OPENAI_API_BASE"],
        default_api_base="https://aihubmix.com/v1",
    )
    if curator_model is None:
        curator_model = manager_model
    rag_curator = RAGCurator(curator_model)
    location_hint_fallback = context.get("company_site") or "目标地区"

    def resolve_company_name() -> str:
        name = context.get("company_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        return "未指定企业"

    def resolve_location_hint() -> str:
        hint = context.get("company_site")
        if isinstance(hint, str) and hint.strip():
            return hint.strip()
        return location_hint_fallback

    def curate_for_rag(content: str, source: str, category: str) -> str | None:
        text = content.strip()
        if not text:
            return None
        company_name = resolve_company_name()
        query_hint = text[:400]
        existing_chunks = rag_store.query(company_name=company_name, query=query_hint, top_k=3)
        existing_context = "\n\n".join(chunk["content"] for chunk in existing_chunks if chunk.get("content"))
        curated = rag_curator.curate(
            company_name=company_name,
            location_hint=resolve_location_hint(),
            source=source,
            category=category,
            new_text=text,
            existing_context=existing_context,
        )
        if curated:
            log_curation_event(
                company_name=company_name,
                location_hint=resolve_location_hint(),
                source=source,
                category=category,
                input_chars=len(text),
                output_chars=len(curated),
            )
        return curated

    def ingest_text_if_possible(content: str, source: str, category: str) -> None:
        curated = curate_for_rag(content, source, category)
        if not curated:
            return
        company_name = resolve_company_name()
        rag_store.add_documents(
            company_name=company_name,
            contents=[curated],
            metadata={"source": source, "category": category},
        )
        rag_store.update_playbook(
            company_name=company_name,
            location_hint=resolve_location_hint(),
            source=source,
            category=category,
            curated_entry=curated,
            curator=rag_curator,
        )

    class CachingVisitTool(VisitTool):
        def forward(self, url: str) -> str:  # type: ignore[override]
            output = super().forward(url)
            address = getattr(self.browser, "address", url)  # type: ignore[attr-defined]
            title = getattr(self.browser, "page_title", "")  # type: ignore[attr-defined]
            body = getattr(self.browser, "page_content", "")  # type: ignore[attr-defined]
            full_text = f"URL: {address}\nTitle: {title}\n\n{body}"
            ingest_text_if_possible(full_text, source=url, category="web_visit")
            return output

    class CachingArchiveSearchTool(ArchiveSearchTool):
        def forward(self, url, date) -> str:  # type: ignore[override]
            output = super().forward(url, date)
            resolved_source = getattr(self.browser, "address", url)  # type: ignore[attr-defined]
            title = getattr(self.browser, "page_title", "")  # type: ignore[attr-defined]
            body = getattr(self.browser, "page_content", "")  # type: ignore[attr-defined]
            full_text = f"URL: {resolved_source}\nTitle: {title}\n\n{body}"
            ingest_text_if_possible(full_text, source=resolved_source, category="web_archive")
            return output

    class CachingGoogleSearchTool(GoogleSearchTool):
        def forward(self, query: str, filter_year: int | None = None) -> str:  # type: ignore[override]
            import requests
            if not self.api_key:
                try:
                    from ddgs import DDGS

                    ddgs = DDGS()
                    results = list(ddgs.text(query, max_results=10))
                except Exception as exc:  # pragma: no cover - fallback path
                    raise ValueError(f"搜索失败：{exc}") from exc
                if not results:
                    return f"No results found for '{query}'."
                formatted_lines = ["## Search Results"]
                distilled_entries = []
                seen_links: set[str] = set()
                for idx, page in enumerate(results):
                    title = page.get("title", "")
                    link = page.get("href", "")
                    snippet = page.get("body", "")
                    if link in seen_links:
                        continue
                    seen_links.add(link)
                    formatted_lines.append(f"{idx}. [{title}]({link})\n{snippet}")
                    distilled_entries.append(
                        "\n".join(
                            [
                                f"标题: {title}",
                                f"链接: {link}",
                                f"摘要: {snippet or '（无摘要）'}",
                            ]
                        )
                    )
                company_name = resolve_company_name()
                if distilled_entries and company_name:
                    ingest_text_if_possible(
                        "\n\n".join(distilled_entries),
                        source="google_search",
                        category="search_results",
                    )
                return "\n\n".join(formatted_lines)

            if self.provider == "serpapi":
                params = {
                    "q": query,
                    "api_key": self.api_key,
                    "engine": "google",
                    "google_domain": "google.com",
                }
                base_url = "https://serpapi.com/search.json"
            else:
                params = {
                    "q": query,
                    "api_key": self.api_key,
                }
                base_url = "https://google.serper.dev/search"
            if filter_year is not None:
                params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                results = response.json()
            else:
                raise ValueError(response.json())

            if self.organic_key not in results.keys():
                if filter_year is not None:
                    raise Exception(
                        f"No results found for query: '{query}' with filtering on year={filter_year}. "
                        "Use a less restrictive query or do not filter on year."
                    )
                raise Exception(f"No results found for query: '{query}'. Use a less restrictive query.")

            organic_results = results.get(self.organic_key, [])
            if len(organic_results) == 0:
                year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
                return (
                    f"No results found for '{query}'{year_filter_message}. "
                    "Try with a more general query, or remove the year filter."
                )

            formatted_lines: list[str] = ["## Search Results"]
            distilled_entries: list[str] = []
            seen_links: set[str] = set()
            for idx, page in enumerate(organic_results):
                date_published = page.get("date")
                source = page.get("source")
                snippet = page.get("snippet", "")
                title = page.get("title", "")
                link = page.get("link", "")
                if link in seen_links:
                    continue
                if link:
                    seen_links.add(link)
                line_parts = [f"{idx}. [{title}]({link})" if title else f"{idx}. {link}"]
                if date_published:
                    line_parts.append(f"Date published: {date_published}")
                if source:
                    line_parts.append(f"Source: {source}")
                if snippet:
                    line_parts.append(snippet)
                formatted_lines.append("\n".join(line_parts))

                distilled_entries.append(
                    "\n".join(
                        [
                            f"标题: {title}" if title else f"链接: {link}",
                            f"链接: {link}",
                            f"日期: {date_published or '未知'}",
                            f"来源: {source or '未知'}",
                            f"摘要: {snippet or '（无摘要）'}",
                        ]
                    )
                )

            company_name = resolve_company_name()
            if distilled_entries and company_name:
                ingest_text_if_possible(
                    "\n\n".join(distilled_entries),
                    source="google_search",
                    category="search_results",
                )

            return "\n\n".join(formatted_lines)

    class CachingTextInspectorTool(TextInspectorTool):
        def forward(self, file_path, question: str | None = None) -> str:  # type: ignore[override]
            result = self.md_converter.convert(file_path)

            if file_path[-4:].lower() in [".png", ".jpg"]:
                raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

            if ".zip" in file_path.lower():
                output = result.text_content
            elif not question:
                output = result.text_content
            else:
                if self.model is None:
                    raise ValueError("TextInspectorTool requires a model when a question is provided.")
                messages = [
                    {
                        "role": MessageRole.SYSTEM,
                        "content": [
                            {
                                "type": "text",
                                "text": "You will have to write a short caption for this file, then answer this question:"
                                + question,
                            }
                        ],
                    },
                    {
                        "role": MessageRole.USER,
                        "content": [
                            {
                                "type": "text",
                                "text": "Here is the complete file:\n### "
                                + str(result.title)
                                + "\n\n"
                                + result.text_content[: self.text_limit],
                            }
                        ],
                    },
                    {
                        "role": MessageRole.USER,
                        "content": [
                            {
                                "type": "text",
                                "text": "Now answer the question below. Use these three headings: '1. Short answer', '2. Extremely detailed answer', '3. Additional Context on the document and question asked'."
                                + question,
                            }
                        ],
                    },
                ]
                output = self.model(messages).content

            ingest_text_if_possible(result.text_content, source=file_path, category="file_inspection")
            return output

    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    caching_text_inspector = CachingTextInspectorTool(manager_model, text_limit)

    WEB_TOOLS = [
        CompanyRAGRetrieveTool(rag_store, resolve_company_name),
        CachingGoogleSearchTool(provider="serper"),
        CachingVisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        CachingArchiveSearchTool(browser),
        caching_text_inspector,
        CompanyRAGIngestTool(rag_store, resolve_company_name, resolve_location_hint, rag_curator),
    ]

    location_hint = resolve_location_hint()
    search_agent_instructions = f"""你是负责公开资料搜集与证据整理的分析员。每次接到 manager_agent 的子任务时：
    - 先写下检索计划（目标主题、优先渠道、关键词/时间窗口），再执行工具调用。
    - 若任务说明中包含企业名称，务必首先调用 `company_rag_retrieve` 阅读最新 Playbook，复用已整理的要点，再决定是否开展新的联网检索。
    - 重点检索公司官网公告、政府/监管数据库、本地与主流媒体报道、行业资讯站、法院与执行公告、知识产权/商标专利库、信用与商业数据库、社交/口碑平台、短视频与论坛，以及{location_hint}当地产经/政务新闻和政策发布，尤其关注企业舆情相关的热词、事件与互动数据。
    - 每条发现需包含：事实摘要、原始来源链接、发布日期或抓取时间、来源类型（官网/监管/主流媒体/地方媒体/社交/投诉等）、可信度判断，以及对应到报告模版中的章节（该模版见任务描述第一部分）。
    - 对同名企业或未经核实的线索要显著标注，并提出仍需验证的内容、可能的补充渠道或采访对象；针对舆情事件要说明传播范围、企业回应与后续状态。
    - 在 run summary 中说明已覆盖的章节、关键结论、舆情信号、未解问题与下一步计划。
    - 收到 critic_agent 的反馈后必须逐条回应：写明已采取的补救措施、仍受限的原因及替代方案。
    - 对于自动工具未覆盖的线下资料或额外文件，可调用 `company_rag_ingest` 写入知识库；避免对同一网页或搜索结果重复入库。
    - 禁止编造或臆测，如信息缺失需说明限制并建议线下或高权限取证路径。"""
    text_webbrowser_agent = ToolCallingAgent(
        model=search_model,
        tools=WEB_TOOLS,
        max_steps=search_max_steps,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        instructions=search_agent_instructions,
        provide_run_summary=True,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += f"""You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information.
    优先覆盖 {location_hint} 地区的政务公告、主流/地方媒体和公共舆情渠道；阶段性检索完成后，请先用自然语言总结主要发现、来源分布与缺口，再调用 final_answer，以便 critic_agent 评估；总结中需回应上一轮 critic_agent 的逐条建议。"""
    text_webbrowser_agent.state["company_site"] = location_hint

    critic_agent_instructions = f"""你是公开信息尽调的质量评审员，需判断现有材料能否支持对企业近期重点与舆情态势的准确总结。
每轮评审时：
- 覆盖度：逐条比对报告模版中的章节（见任务描述开头），确认官网/监管公告、主流媒体、{location_hint} 地方媒体及社交/投诉渠道是否都有涉猎，尤其评估“企业舆情雷达”部分是否提供事件时间轴、情绪分析与企业回应；检查 search_agent 是否已阅读最新 Playbook，并仅在自动入库未覆盖时调用 `company_rag_ingest` 补充材料。
- 可信度：检查引用是否来自权威公开渠道，链接是否有效，是否存在同名实体混淆或未经证实的传闻。
- 时效性：核对信息是否落在要求的时间窗口内，对过时或无日期的线索给出处理建议。
- 汇总性：评估 search_agent 的总结是否提炼出“近期战略重点”“监管热点”“舆情焦点”等关键信息，并指出影响权重。
输出结构化评价：
1. 总体评级（高/中/低）与一句话理由；
2. 已满足的亮点（≤3 条）；
3. 主要缺口或风险点（逐条写影响与建议行动）；
4. 建议直接交给 search_agent 的后续检索或验证指令列表；
5. Playbook 更新情况与建议的清理/补充动作；
6. 需要 manager_agent 提供的额外上下文（如有）。
仅基于提交材料做判断，不得凭空推断。"""

    critic_agent = ToolCallingAgent(
        model=critic_model,
        tools=[],
        max_steps=critic_max_steps,
        verbosity_level=2,
        planning_interval=4,
        name="critic_agent",
        description="评估 search_agent 阶段性输出的覆盖度、可信度与下一步检索建议的评论员。",
        instructions=critic_agent_instructions,
    )

    manager_instructions = f"""你是企业背景调查的项目统筹，目标是基于公开渠道、主流/地方媒体与线上舆情数据，完成报告模版（见任务描述开头）所需的最新洞察。
工作流程：
1. 解读初始任务，拆分为若干阶段目标（如信息来源梳理、战略重点梳理、监管舆情跟踪、舆情热度分析），明确告诉 search_agent 需覆盖的来源类型与时间窗口，特别强调 {location_hint} 的政务/媒体/社区信息渠道，并提醒其优先使用 `company_rag_retrieve` 复用缓存信息；只有当自动入库未覆盖线索时，才指示 search_agent 使用 `company_rag_ingest` 补充资料。
2. 在每个重要阶段结束后，把 search_agent 的阶段总结提交给 critic_agent，请其评估覆盖度与提炼质量；随后将关键反馈转述给 search_agent，要求其逐条回应并更新检索计划。
3. 维护任务看板，实时记录已完成章节、缺失信息、阻塞原因以及下一步行动，必要时调整优先级或向用户请求额外上下文，若 {location_hint} 线索不足需显著标注。
4. 协同维护企业 Playbook：确保 search_agent 的重要发现都写入并符合整理标准，必要时触发清理过期信息或补充关键空白。
5. 仅当 critic_agent 给出“中”及以上评级，且高优先级缺口（尤其是舆情雷达中的关键事件）都有来源支持或合理解释时，才进入归纳输出阶段；否则继续组织补充检索。
6. 生成最终报告时，确保聚焦近期战略重点、监管/合规动态、舆情热度及声誉风险评估，并在附录中列出完整的资料清单与舆情监测建议，注明 {location_hint} 当地渠道的监测计划及 Playbook 更新日期。
保持指令清晰可执行，避免冗长描述，确保多轮协作高效闭环。"""

    manager_agent = CodeAgent(
        model=manager_model,
        tools=[visualizer, caching_text_inspector],
        max_steps=manage_max_steps,
        verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=4,
        instructions=manager_instructions,
        managed_agents=[text_webbrowser_agent, critic_agent],
    )

    return manager_agent
