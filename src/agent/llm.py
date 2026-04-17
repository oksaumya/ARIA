from langchain_openai import ChatOpenAI

# Groq free-tier: https://console.groq.com
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"


def get_llm(temperature: float = 0.0, api_key: str = None) -> ChatOpenAI:
    """Returns a ChatOpenAI instance pointed at the Groq API (OpenAI-compatible).

    Args:
        api_key: Groq API key supplied by the user at runtime. Must not be empty.
    """
    if not api_key:
        raise ValueError(
            "A Groq API key is required. Please enter your key in the sidebar."
        )

    return ChatOpenAI(
        model=GROQ_MODEL,
        api_key=api_key,
        base_url=GROQ_BASE_URL,
        temperature=temperature,
        max_tokens=4096,
        max_retries=3,
        request_timeout=60,
    )
