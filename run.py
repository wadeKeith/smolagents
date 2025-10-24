import argparse
import os
import threading

from dotenv import load_dotenv
from huggingface_hub import login
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

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    # InferenceClientModel,
    LiteLLMModel,
    ToolCallingAgent,
)
from smolagents.cli import load_model
from jinja2 import StrictUndefined, Template
from typing import TYPE_CHECKING, Any, Literal, Type, TypeAlias, TypedDict, Union

def populate_template(template: str, variables: dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")

company_template = """
请以公开渠道与主流媒体为核心，为“{{company_name}}”撰写企业背景调查报告（司法管辖提示：{{jurisdiction_hint | default('CN')}}，重点关注最近 {{time_window_months | default(24)}} 个月内的动态，并补充更早但仍影响判断的事件），使用 {{report_language | default('中文')}} 输出。

交付结构
一、任务概述与信息范围
- 说明本次调研目标、重点问题、覆盖地域与排除项，列出关注的核心法人实体、品牌或业务条线。

二、信息来源总览
- 汇总已检索与计划检索的公开渠道，至少涵盖：公司官网与公告、当地政府及监管数据库（工商、证监、交易所、税务、环保、安全生产等）、主流与行业媒体、地方新闻站点、权威商业资讯平台、法院及执行公告、知识产权/商标专利库、信用评级或商业数据服务、社交媒体与口碑平台、消费者投诉网站。为每条来源记录：渠道名称、访问链接或检索方式、最新抓取时间、可获取内容类型（新闻/公告/数据等）、访问限制（公开/付费/需登陆）。

三、企业概况速览
- 概述企业历史沿革、核心业务与产品组合、主营市场、管理层及股权结构要点，引用官方或权威公开资料。若存在同名公司或关联主体，需标注差异。

四、近期战略重点
- 结合官网动态、官方公告、主流媒体与地方媒体报道，总结企业在 {{time_window_months | default(24)}} 个月内的主要动作（如产线扩张、市场进入、合作伙伴、技术发布、投融资等），并指出报道来源、发布时间及其对业务的潜在影响。

五、合规与监管动向
- 汇总监管公告、政府公示、司法或行政处罚、环保/安全检查、税务信息等公开记录；说明事件时间、主管机构、处理状态与对企业运营或外部合作的影响。

六、企业舆情雷达
- 采集主流媒体、地方新闻门户、行业评论、社交/视频平台（微博、微信公号、抖音、B 站、知乎、Glassdoor 等）以及消费者投诉渠道对企业的讨论热度、情绪倾向、核心议题与关键词演变；区分官方声明、第三方报道与匿名信息，标注发布时间、传播渠道、互动量或传播级别（如阅读量/转发/点赞）。
- 对重大舆情事件提供时间轴，说明事件触发点、企业回应、外部反馈与当前状态；如存在谣言或未经证实的消息，要指出核实状态与建议的监测动作。

七、风险与机会评估
- 基于上述信息，对企业近期的核心风险点（如经营、合规、声誉、财务）与亮点机会逐项分析；说明来源支持、影响程度（【高】/【中】/【低】）以及可能的后续观察指标。

八、信息缺口与后续建议
- 指出尚未获取但对判断关键的资料，说明原因（如付费、权限限制、需访谈），并建议下一步验证路径或需要关注的渠道。

附录：资料清单
- 按引用顺序列出所有来源（名称、URL/访问路径、发布日期或检索日期、来源类型、可信度评估）。如引用二手渠道或推断，请明确标注并说明依据。
"""

DEFAULT_COMPANY_VARIABLES: dict[str, Any] = {
    "company_name": "韶关得利包装科技有限公司",
    "jurisdiction_hint": "CN",
    "time_window_months": 24,
    "report_language": "中文",
    "company_site": "韶关市",
}
    


def resolve_task_prompt(args: argparse.Namespace) -> str:
    variables = DEFAULT_COMPANY_VARIABLES.copy()
    updates = {
        "company_name": getattr(args, "company_name", None),
        "jurisdiction_hint": getattr(args, "jurisdiction_hint", None),
        "time_window_months": getattr(args, "time_window_months", None),
        "report_language": getattr(args, "report_language", None),
        "company_site": getattr(args, "company_site", None),
    }
    for key, value in updates.items():
        if value is not None:
            variables[key] = value
    return populate_template(company_template, variables=variables)


load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--company-name",
        type=str,
        help="Company name to inject into the default background check template.",
        default="无锡三代科技有限公司",
    )
    parser.add_argument(
        "--jurisdiction-hint",
        type=str,
        help="Jurisdiction hint (e.g. CN, US) for the template.",
        default="CN",
    )
    parser.add_argument(
        "--time-window-months",
        type=int,
        help="How many months of company history to review in the template.",
        default=24,
    )
    parser.add_argument(
        "--report-language",
        type=str,
        help="Language to request for the generated report.",
        default="Chinese",
    )
    parser.add_argument(
        "--company-site",
        type=str,
        help="City or region to emphasise when looking up company information.",
        default=None,
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="OpenAIServerModel",
        help="The model type to use (e.g., OpenAIServerModel, LiteLLMModel, TransformersModel, InferenceClientModel, VLLMModel)",
    )
    parser.add_argument("--model-id", type=str, default="o1")
    parser.add_argument(
        "--provider",
        type=str,
        # default='novita',
        help="The inference provider to use for the model",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="https://aihubmix.com/v1",
        help="The API base to use for the model",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="sk-T6U5Pf6OSazCEFjqEdCa4a88E89242A898Bc453b07AeE156",
        help="The API key to use for the model",
    )
    parser.add_argument(
        "--search_max_steps",
        type=int,
        default=50,
        help="The maximum number of steps the agent can take",
    )
    parser.add_argument(
        "--critic_max_steps",
        type=int,
        default=50,
        help="The maximum number of steps the agent can take",
    )
    parser.add_argument(
        "--manage_max_steps",
        type=int,
        default=100,
        help="The maximum number of steps the agent can take",
    )
    return parser.parse_args()


custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def create_agent(model_type, model_id="o1", provider=None, api_base=None, api_key=None, search_max_steps=20, critic_max_steps=20, manage_max_steps=12):

    model = load_model(model_type, model_id, provider=provider, api_base=api_base, api_key=api_key)

    text_limit = 100000
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    WEB_TOOLS = [
        GoogleSearchTool(provider="serper"),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]
    search_agent_instructions = """你是负责公开资料搜集与证据整理的分析员。每次接到 manager_agent 的子任务时：
- 先写下检索计划（目标主题、优先渠道、关键词/时间窗口），再执行工具调用。
- 重点检索公司官网公告、政府/监管数据库、本地与主流媒体报道、行业资讯站、法院与执行公告、知识产权/商标专利库、信用与商业数据库、社交/口碑平台、短视频与论坛，以及地方新闻和政策发布，尤其关注企业舆情相关的热词、事件与互动数据。
- 每条发现需包含：事实摘要、原始来源链接、发布日期或抓取时间、来源类型（官网/监管/主流媒体/地方媒体/社交/投诉等）、可信度判断，以及对应到 company_template 的章节。
- 对同名企业或未经核实的线索要显著标注，并提出仍需验证的内容、可能的补充渠道或采访对象；针对舆情事件要说明传播范围、企业回应与后续状态。
- 在 run summary 中说明已覆盖的章节、关键结论、舆情信号、未解问题与下一步计划。
- 收到 critic_agent 的反馈后必须逐条回应：写明已采取的补救措施、仍受限的原因及替代方案。
- 禁止编造或臆测，如信息缺失需说明限制并建议线下或高权限取证路径。"""
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=manage_max_steps,
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
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information.
    阶段性检索完成后，请先用自然语言总结主要发现、来源分布与缺口，再调用 final_answer，以便 critic_agent 评估；总结中需回应上一轮 critic_agent 的逐条建议。"""

    critic_agent_instructions = """你是公开信息尽调的质量评审员，需判断现有材料能否支持对企业近期重点与舆情态势的准确总结。
每轮评审时：
- 覆盖度：逐条比对 company_template 章节，确认官网/监管公告、主流媒体、地方媒体及社交/投诉渠道是否都有涉猎，尤其评估“企业舆情雷达”部分是否提供事件时间轴、情绪分析与企业回应。
- 可信度：检查引用是否来自权威公开渠道，链接是否有效，是否存在同名实体混淆或未经证实的传闻。
- 时效性：核对信息是否落在要求的时间窗口内，对过时或无日期的线索给出处理建议。
- 汇总性：评估 search_agent 的总结是否提炼出“近期战略重点”“监管热点”“舆情焦点”等关键信息，并指出影响权重。
输出结构化评价：
1. 总体评级（高/中/低）与一句话理由；
2. 已满足的亮点（≤3 条）；
3. 主要缺口或风险点（逐条写影响与建议行动）；
4. 建议直接交给 search_agent 的后续检索或验证指令列表；
5. 需要 manager_agent 提供的额外上下文（如有）。
仅基于提交材料做判断，不得凭空推断。"""
    critic_agent = ToolCallingAgent(
        model=model,
        tools=[],
        max_steps=critic_max_steps,
        verbosity_level=2,
        planning_interval=4,
        name="critic_agent",
        description="评估 search_agent 阶段性输出的覆盖度、可信度与下一步检索建议的评论员。",
        instructions=critic_agent_instructions,
    )

    manager_instructions = """你是企业背景调查的项目统筹，目标是基于公开渠道、主流/地方媒体与线上舆情数据，完成 company_template 所需的最新洞察。
工作流程：
1. 解读初始任务，拆分为若干阶段目标（如信息来源梳理、战略重点梳理、监管舆情跟踪、舆情热度分析），明确告诉 search_agent 需覆盖的来源类型与时间窗口。
2. 在每个重要阶段结束后，把 search_agent 的阶段总结提交给 critic_agent，请其评估覆盖度与提炼质量；随后将关键反馈转述给 search_agent，要求其逐条回应并更新检索计划。
3. 维护任务看板，实时记录已完成章节、缺失信息、阻塞原因以及下一步行动，必要时调整优先级或向用户请求额外上下文。
4. 仅当 critic_agent 给出“中”及以上评级，且高优先级缺口（尤其是舆情雷达中的关键事件）都有来源支持或合理解释时，才进入归纳输出阶段；否则继续组织补充检索。
5. 生成最终报告时，确保聚焦近期战略重点、监管/合规动态、舆情热度及声誉风险评估，并在附录中列出完整的资料清单与舆情监测建议。
保持指令清晰可执行，避免冗长描述，确保多轮协作高效闭环。"""

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, TextInspectorTool(model, text_limit)],
        max_steps=search_max_steps,
        verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=4,
        instructions=manager_instructions,
        managed_agents=[text_webbrowser_agent, critic_agent],
    )

    return manager_agent


def main():
    args = parse_args()

    agent = create_agent(
        model_type=args.model_type,
        model_id=args.model_id,
        provider=args.provider,
        api_base=args.api_base,
        api_key=args.api_key,
        search_max_steps=args.search_max_steps,
        critic_max_steps = args.critic_max_steps,
        manage_max_steps=args.manage_max_steps,
    )

    task_prompt = resolve_task_prompt(args)
    answer = agent.run(task_prompt)

    print(f"Got this answer: {answer}")


if __name__ == "__main__":
    main()
