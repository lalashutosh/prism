# tests package
# All tests are pure unit tests — no real LLM calls, no retrieval calls.
# LLM calls are intercepted by MagicMock clients injected via the llm_client
# parameter on each agent entry-point function.
