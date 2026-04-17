from langchain_core.messages import SystemMessage, HumanMessage
from src.agent.state import ResearchState


def answer_question(state: ResearchState, question: str, api_key: str = None) -> tuple[str, ResearchState]:
    """Answer a follow-up question grounded in the research report."""
    from src.agent.llm import get_llm

    report_md = state.get("report_markdown", "")
    aggregated = state.get("aggregated_text", "")

    llm = get_llm(api_key=api_key)

    system = (
        "You are a research assistant answering follow-up questions about a completed research report. "
        "Answer ONLY based on the information in the report and research synthesis provided. "
        "If the answer is not found in the provided content, say: "
        "'This question is not addressed in the current research. You may want to run a new search.'\n\n"
        f"Research Report:\n{report_md}\n\n"
        f"Full Research Synthesis:\n{aggregated}"
    )

    try:
        resp = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=question),
        ])
        answer = resp.content.strip()
    except Exception as e:
        answer = f"I encountered an error while processing your question: {str(e)}"

    qa_history = list(state.get("qa_history", []))
    qa_history.append({"question": question, "answer": answer})

    updated_state = {**state, "qa_history": qa_history}
    return answer, updated_state
