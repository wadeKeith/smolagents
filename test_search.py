from smolagents import CodeAgent, TransformersModel, DuckDuckGoSearchTool,VLLMModel

model = VLLMModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    model_kwargs={
        # "dtype": "bfloat16",
        "tensor_parallel_size": 8,
        "gpu_memory_utilization": 0.6,
    },
)

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
)

# Now the agent can search the web!
result = agent.run("What is the current weather in Paris?",max_steps = 20)
print(result)
