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
为求职者对“{{company_name}}”开展深入背景调查（司法管辖提示：{{jurisdiction_hint | default('CN')}}，重点关注最近 {{time_window_months | default(24)}} 个月内的重要情况，必要时保留更早的关键事件），请用 {{report_language | default('中文')}} 输出结论。
请回答以下问题：
1. 该品牌背后对应的法律实体有哪些？主要实体的注册地、统一社会信用代码或注册号是什么？是否存在母子公司、关联公司或同名企业导致混淆？
2. 该公司的基础信息及经营概况如何，包括法定代表人/高管、成立时间、企业类型、主营行业和经营范围、注册地址与实际经营地、官网或常用域名、人员规模等？有什么可验证的企业亮点或风险信号？
3. 注册资本、实缴资本（包含金额、币种、最近确认日期）以及资本变动、融资或股权结构调整情况如何？主要股东、受益所有人、董事/管理层、重要子公司或关联公司的情况如何？
4. 在所述时间范围内，该公司是否涉及重大诉讼、仲裁、执行、行政处罚、监管调查、制裁/黑名单、产品/安全/数据隐私事件或严重负面舆情？关键事件的时间、监管/司法机关、处理状态、金额或影响如何？这些事件对求职者意味着什么？
5. 在就业和社交平台（如脉脉、知乎、微博、Glassdoor、Indeed、Blind、Reddit 小红书 等）上，该公司被如何评价？主要讨论话题、整体情绪（正/中/负）、具有代表性的匿名观点是什么？来源链接和时间点如何？
6. 该公司最近 {{time_window_months | default(24)}} 个月内的新闻报道和公开信息有哪些？涉及的主要事件、时间节点、来源和影响如何？这些信息对求职者意味着什么？
7. 还有哪些尚未核实、因付费/登录/合规限制而无法获取、或值得求职者后续关注的要点？请清楚说明缺口和建议的跟进方向。
请结合可信的公开来源，例如从：中国裁判文书网、{{company_site}}的政府公开网、全国企业信用信息公示系统来查询，并注明关键信息的出处和获取日期，给出能够帮助求职者评估入职风险与机会的洞察。
同时，请系统梳理来自当地新闻报道、行业媒体、权威门户以及其他可靠网页的信息，综合分析该公司的业务发展、政策环境、舆情走向、竞争格局和市场定位，并指出这些信号对求职者意味着什么。
"""

DEFAULT_COMPANY_VARIABLES: dict[str, Any] = {
    "company_name": "韶关得利包装科技有限公司",
    "jurisdiction_hint": "CN",
    "time_window_months": 24,
    "report_language": "中文",
    "company_site": "韶关市",
}


def build_company_request(
    company_name: str | None = None,
    jurisdiction_hint: str | None = None,
    time_window_months: int | None = None,
    report_language: str | None = None,
    company_site: str | None = None,
) -> str:
    variables = DEFAULT_COMPANY_VARIABLES.copy()
    updates = {
        "company_name": company_name,
        "jurisdiction_hint": jurisdiction_hint,
        "time_window_months": time_window_months,
        "report_language": report_language,
        "company_site": company_site,
    }
    for key, value in updates.items():
        if value is not None:
            variables[key] = value
    return populate_template(company_template, variables=variables)


def resolve_task_prompt(args: argparse.Namespace) -> str:
    if getattr(args, "question", None):
        return args.question
    return build_company_request(
        company_name=getattr(args, "company_name", None),
        jurisdiction_hint=getattr(args, "jurisdiction_hint", None),
        time_window_months=getattr(args, "time_window_months", None),
        report_language=getattr(args, "report_language", None),
        company_site=getattr(args, "company_site", None),
    )


company_request = build_company_request()


load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        type=str,
        help="Prompt describing the company background check task for the agent.",
        default=None,
    )
    parser.add_argument(
        "--company-name",
        type=str,
        help="Company name to inject into the default background check template.",
        default=None,
    )
    parser.add_argument(
        "--jurisdiction-hint",
        type=str,
        help="Jurisdiction hint (e.g. CN, US) for the template.",
        default=None,
    )
    parser.add_argument(
        "--time-window-months",
        type=int,
        help="How many months of company history to review in the template.",
        default=None,
    )
    parser.add_argument(
        "--report-language",
        type=str,
        help="Language to request for the generated report.",
        default=None,
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
        "--code_max_steps",
        type=int,
        default=50,
        help="The maximum number of steps the agent can take",
    )
    parser.add_argument(
        "--tool_max_steps",
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


def create_agent(model_type, model_id="o1", provider=None, api_base=None, api_key=None, code_max_steps=20, tool_max_steps=12):
    # model_params = {
    #     "model_id": model_id,
    #     "custom_role_conversions": custom_role_conversions,
    #     "max_completion_tokens": 8192,
    # }
    # if model_id == "o1":
    #     model_params["reasoning_effort"] = "high"
    # model = LiteLLMModel(**model_params)

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
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=tool_max_steps,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, TextInspectorTool(model, text_limit)],
        max_steps=code_max_steps,
        verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
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
        code_max_steps=args.code_max_steps,
        tool_max_steps=args.tool_max_steps,
    )

    task_prompt = resolve_task_prompt(args)
    answer = agent.run(task_prompt)

    print(f"Got this answer: {answer}")


if __name__ == "__main__":
    main()
