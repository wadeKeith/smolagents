from run import build_company_prompt, build_company_variables, create_agent, parse_args

import gradio as gr


args = parse_args()

initial_variables = build_company_variables(
    company_name=args.company_name,
    jurisdiction_hint=args.jurisdiction_hint,
    time_window_months=args.time_window_months,
    report_language=args.report_language,
    company_site=args.company_site,
)


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

    updated_variables = build_company_variables(
        company_name=_sanitize_text(company_name),
        jurisdiction_hint=_sanitize_text(jurisdiction_hint),
        time_window_months=months,
        report_language=_sanitize_text(report_language),
        company_site=_sanitize_text(company_site),
    )
    prompt = build_company_prompt(updated_variables)
    agent = create_agent(
        model_type=args.model_type,
        model_id=args.model_id,
        provider=args.provider,
        api_base=args.api_base,
        api_key=args.api_key,
        search_max_steps=args.search_max_steps,
        critic_max_steps=args.critic_max_steps,
        manage_max_steps=args.manage_max_steps,
        company_context=updated_variables,
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
                label="重点关注城市/区域",
                value=initial_variables.get("company_site", ""),
                placeholder="例如 上海、深圳，或留空",
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
