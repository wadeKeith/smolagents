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
请按照资深背调专家的流程，为企业合作风控与尽调目的，对“{{company_name}}”执行结构化背景调查（司法管辖提示：{{jurisdiction_hint | default('CN')}}，重点覆盖最近 {{time_window_months | default(24)}} 个月的重要动态，并回溯能改变结论的历史事件），以 {{report_language | default('中文')}} 输出。

交付格式与要点如下：
一、任务设定与范围边界
- 说明此次调查的目标（合作/投资/供应链控制等）与深度，列出覆盖的主题与排除项，确认主要法律实体及可访问的数据地域（含 {{company_site}} 重点渠道）。

二、信息来源与取证记录
- 按来源类型列出已访问渠道、数据库与检索方法并标注时间（UTC+8）；如有受限或需付费渠道请单独备注。
- 重点覆盖：公司官网与公告（含上市公司年报）；工商注册与监管披露平台（如 ACRA、新加坡企业管制局、国家企业信用信息公示系统、香港公司注册处、SEC EDGAR、CSRC 公告、SGX 公告）；法律与诉讼数据库（中国裁判文书网、PACER、LawNet、地方法院公告、执行信息公开网）；行业监管或专项合规平台（生态环境部、安全生产或 OSHA、税务与环保公开信息）；信用评级与商业数据服务（Dun & Bradstreet、Experian、Credit Bureau Singapore 等）；商标与专利数据库（WIPO PatentScope、各国知识产权局）；主流与行业媒体；行业/市场研究与协会网站（麦肯锡、行业协会、BBB、全国工商联等）；公共统计库（国家统计局、世行、IMF、UN Global Compact、ITC TradeMap）；供应链与贸易数据（海关、船运记录、贸易数据库）；行业会议与专家访谈线索；学术研究平台（Google Scholar、ResearchGate、大学图书馆案例库）；社交与职业平台（LinkedIn、Glassdoor 等）；消费者投诉与地方政府公开渠道（CASE、中国消协、地方经发局、补贴或处罚公告）。

三、企业概况与里程碑
- 汇总法律实体、统一社会信用代码或注册号、成立/变更时间、企业类型、核心业务线、总部与运营地点、重要历史事件，并提示可能的同名或关联混淆。

四、组织治理与高管背景
- 梳理控股股东、受益所有人、董事会成员及核心高管，概述其教育/履历/行业声誉；如有司法或合规记录需点明来源与影响。

五、股权结构与资本安排
- 展示最新股权结构、注册资本与实缴情况（注明币种与确认时间）、主要子公司/分支、资本增减或股东变更时间线，分析出资真实性与潜在控制关系。

六、业务版图与市场定位
- 描述主营产品/服务、目标客户群、收入或订单结构、关键合作伙伴、地域覆盖，评估与工商登记经营范围的一致性，分析竞争对手与行业位置。
- 结合行业报告、市场研究、公共统计数据（国家统计局、世行、IMF 等）以及供应链/贸易数据库（如海关或船运记录）验证业务规模、上下游关系和跨境交易情况。

七、财务健康度分析
- 汇总官方财务报表、监管披露文件、信用评级和商业数据平台信息，分析盈利能力、现金流、偿债能力等指标。
- 无法获取完整数据时说明缺口并推断合理性，指出潜在财务风险信号或需要第三方验证的重点。

八、合规、法律与监管风险
- 按时间顺序列出诉讼/仲裁/执行、行政处罚、监管调查、制裁或黑名单、数据与隐私事件，说明主管机关、金额或影响、当前状态及对合作方的意义。
- 特别关注税务、环保（生态环境部、地方环保局、OSHA 等安全监管）及劳动监管部门的公告，以及法院裁判文书与执行信息公开网，区分已结案与在办事项并评估整改或追偿进展。

九、媒体舆情与公众反馈
- 汇总传统媒体、行业报告及社交平台（知乎、微博、Glassdoor、Reddit、小红书等）的主要话题、情绪取向、代表性观点，区分未经证实的传言与可核实事实。
- 纳入消费者保护协会、投诉平台与评分网站的反馈，并记录企业官方回应或整改情况。

十、第三方验证与访谈线索
- 若存在来自前员工、供应商、客户或专业调查机构的公开信息，提炼核心结论；若缺失，提出建议的访谈对象与验证方法。
- 收录行业会议、专家访谈、专业协会或学术研究的关键观点，并说明可信度与适用性。

十一、交叉验证与风险评级
- 说明关键信息的交叉验证情况，对矛盾数据给出解释或进一步验证建议；从财务、市场、合规、法律、声誉等维度给出风险等级（【高】/【中】/【低】）及依据。

十二、持续跟踪计划
- 建议后续应持续监测的指标、渠道或触发条件（如公告、监管更新、媒体监控节奏），提出复查频率与责任建议。

十三、结论与行动建议
- 用 3-6 条结论归纳公司整体状况与关键风险机会，明确对合作/投资/供应链决策的影响，并提出可执行的下一步动作。

附录要求：
- 全文严格引用可信来源，对每条事实列明出处与检索日期；若来源可信度有限或需二次核实，请显著标注。
- 保持信息逻辑清晰，避免“搜到即列”，对重要程度高的信息给予更高权重与解释。
- 报告末尾提供“资料清单”，列出所有引用的来源名称、链接或访问路径、检索时间及可复核性说明。
- 在调查过程中须遵守当地法律与隐私合规要求，如有潜在限制需在报告中提示。
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
