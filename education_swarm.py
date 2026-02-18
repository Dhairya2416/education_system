from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
import os

# API KEY

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or api_key == "YOUR_API_KEY":
    fallback = os.getenv("GEMINI_API_KEY")
    if fallback:
        os.environ["GOOGLE_API_KEY"] = fallback

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)


def research_tool(topic: str):
    """
    Collects educational information, definitions,
    examples, and study notes.
    """
    prompt = f"""
    You are an academic researcher.
    Collect accurate study material on the topic:
    {topic}

    Include:
    - Definition
    - Key concepts
    - Examples
    - Exam-focused points
    """
    return llm.invoke(prompt).content


def writing_tool(research_notes: str):
    """
    Converts research notes into
    well-structured educational content.
    """
    prompt = f"""
    You are an education content writer.
    Convert the following research notes into
    clear, structured student-friendly content.

    Use:
    - Headings
    - Bullet points
    - Simple explanations

    Research Notes:
    {research_notes}
    """
    return llm.invoke(prompt).content

app = create_react_agent(
    model=llm,
    tools=[research_tool, writing_tool],
    name="education_agent",
    prompt="""
    You are an education assistant.
    For any study query, first call research_tool, then call writing_tool
    using the research output to produce final student-friendly content.
    """,
)

config = {"configurable": {"thread_id": "edu-thread-1"}}


def _export_graph_png() -> None:
    try:
        image = app.get_graph().draw_mermaid_png()
        with open("education_swarm.png", "wb") as f:
            f.write(image)
    except Exception:
        
        pass


def run_cli() -> None:
    _export_graph_png()

    while True:
        user_input = input("\nEnter education topic (or 'exit'): ")

        if user_input.lower() == "exit":
            break

        result = app.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
        )

        print("\nFinal Educational Content:\n")
        for m in result["messages"]:
            if hasattr(m, "pretty_print"):
                m.pretty_print()
            else:
                print(getattr(m, "content", m))


if __name__ == "__main__":
    run_cli()
