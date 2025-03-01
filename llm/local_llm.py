from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine, TokensPrompt


def vllm_start_engine(
        model: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        tensor_parallel_size: int
) -> AsyncLLMEngine:
    engine_args = AsyncEngineArgs(
        model=model,
        trust_remote_code=True,
        disable_log_stats=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        enable_lora=True
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine


