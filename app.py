from run import (
    DEFAULT_COMPANY_VARIABLES,
    build_company_request,
    create_agent,
    parse_args,
)

import gradio as gr


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

initial_variables = DEFAULT_COMPANY_VARIABLES.copy()
for field in ("company_name", "jurisdiction_hint", "report_language", "company_site"):
    override = getattr(args, field, None)
    if override is not None:
        initial_variables[field] = override
if getattr(args, "time_window_months", None) is not None:
    initial_variables["time_window_months"] = args.time_window_months


def _sanitize_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def run_background_check(company_name, jurisdiction_hint, time_window_months, report_language, company_site):
    sanitized_months = _sanitize_text(time_window_months)
    months = None
    if sanitized_months is not None:
        try:
            months = int(sanitized_months)
        except ValueError:
            raise gr.Error("time_window_months 需要为数字")

    prompt = build_company_request(
        company_name=_sanitize_text(company_name),
        jurisdiction_hint=_sanitize_text(jurisdiction_hint),
        time_window_months=months,
        report_language=_sanitize_text(report_language),
        company_site=_sanitize_text(company_site),
    )
    answer = agent.run(prompt)
    return prompt, answer


with gr.Blocks(title="企业背景调查助手", theme="ocean") as demo:
    gr.Markdown("## 输入公司信息，生成背景调查报告")

    with gr.Row():
        with gr.Column():
            company_name = gr.Textbox(
                label="公司名称",
                value=initial_variables["company_name"],
                placeholder="请输入企业名称",
            )
            jurisdiction_hint = gr.Textbox(
                label="司法管辖范围（国家）",
                value=initial_variables["jurisdiction_hint"],
                placeholder="司法管辖提示，如 CN、US",
            )
            time_window = gr.Textbox(
                label="查询时间范围（月），例如 24",
                value=str(initial_variables["time_window_months"]),
                placeholder="查询时间范围（月），例如 24",
            )
            report_language = gr.Textbox(
                label="生成报告的语言，例如",
                value=initial_variables["report_language"],
                placeholder="生成报告的语言，例如 中文",
            )
            company_site = gr.Textbox(
                label="企业所在地或重点关注区域",
                value=initial_variables["company_site"],
                placeholder="企业所在地或重点关注区域",
            )
            submit = gr.Button("生成调查报告", variant="primary")

        with gr.Column():
            prompt_output = gr.Textbox(
                label="生成的提示词",
                lines=12,
                interactive=False,
            )
            answer_output = gr.Markdown(label="模型回答")

    submit.click(
        run_background_check,
        inputs=[company_name, jurisdiction_hint, time_window, report_language, company_site],
        outputs=[prompt_output, answer_output],
    )

if __name__ == "__main__":
    demo.launch(share=True)
