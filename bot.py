import re
from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain.memory import ConversationTokenBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType


# Initialize the main LLM used for:
# 1) Intent classification
# 2) Emotion-based generation
# 3) Semantic conversation fallback
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)


# Global stores for per-session state:
# - memory: conversation history (for context awareness)
# - sessions: stores the user's name + marks
# - agent_store: caches the agent instance per session
memory_store: Dict[str, ConversationTokenBufferMemory] = {}
sessions: Dict[str, Dict[str, Any]] = {}
agent_store: Dict[str, Any] = {}


def get_memory(session_id: str) -> ConversationTokenBufferMemory:
    """
    Create or return a memory buffer for a user's session.
    Keeps conversation context for the LLM agent.
    """
    if session_id not in memory_store:
        memory_store[session_id] = ConversationTokenBufferMemory(
            llm=llm,
            max_token_limit=3000,
            return_messages=True,
            memory_key="chat_history",
        )
    return memory_store[session_id]


def get_recent_history(session_id: str) -> str:
    """
    Get readable USER/BOT history text for prompting tools.
    """
    msgs = (
        get_memory(session_id).load_memory_variables({}).get("chat_history", []) or []
    )
    lines = []
    for m in msgs:
        role = "USER" if getattr(m, "type", "") == "human" else "BOT"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def get_or_create_session(session_id: str) -> Dict[str, Any]:
    """
    Ensure the session exists and return its dictionary.
    """
    if session_id not in sessions:
        sessions[session_id] = {"name": None, "marks": {}}
    return sessions[session_id]


# ---------------------------
# Academic Score Processing
# ---------------------------


def calculate_grade(score: int) -> str:
    """
    User-specific grading scale.
    """
    if score >= 90:
        return "S"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 60:
        return "C"
    if score >= 50:
        return "D"
    if score >= 40:
        return "E"
    return "F"


def parse_subject_scores(text: str):
    """
    Extract subjects and marks from flexible formats.
    Supports multiple natural styles:
    - Maths - 90
    - 95 in Physics
    - English 88
    """
    pairs = []

    # Format: Subject - Score
    for subj, score in re.findall(r"([A-Za-z ]+)\s*[-=:]\s*(\d{1,3})", text):
        pairs.append((subj.strip().title(), int(score)))

    # Format: Score in Subject
    for score, subj in re.findall(
        r"(\d{1,3})\s+in\s+([A-Za-z ]+)", text, flags=re.IGNORECASE
    ):
        pairs.append((subj.strip().title(), int(score)))

    # Format: Subject Score (fallback)
    if not pairs:
        tokens = text.replace(",", " ").split()
        for i in range(len(tokens) - 1):
            if tokens[i + 1].isdigit():
                pairs.append((tokens[i].strip().title(), int(tokens[i + 1])))

    return pairs


def add_or_update_marks(text: str, session_id: str) -> str:
    """
    Store or update marks, recalculate grade summary, return formatted table.
    """
    marks = get_or_create_session(session_id)["marks"]
    pairs = parse_subject_scores(text)

    if not pairs:
        return "I could not detect any valid subject scores."

    for sub, sc in pairs:
        marks[sub] = max(0, min(int(sc), 100))  # clamp 0–100

    avg = sum(marks.values()) / len(marks)
    grade = calculate_grade(int(round(avg)))

    lines = [
        "Updated Performance",
        "| Subject | Marks | Grade |",
        "|--------|-------|-------|",
    ]
    for sub, sc in marks.items():
        lines.append(f"| {sub} | {sc} | {calculate_grade(sc)} |")

    lines.append("")
    lines.append(f"Overall: {avg:.2f}% → Grade {grade}")
    lines.append("(Scale: S≥90, A≥80, B≥70, C≥60, D≥50, E≥40, F<40)")
    return "\n".join(lines)


def show_marks_table(session_id: str) -> str:
    """
    Display all stored marks and final computed grade.
    """
    marks = get_or_create_session(session_id)["marks"]
    if not marks:
        return "No marks saved yet."

    avg = sum(marks.values()) / len(marks)
    grade = calculate_grade(int(round(avg)))

    lines = [
        "Grade Summary",
        "| Subject | Marks | Grade |",
        "|--------|-------|-------|",
    ]
    for sub, sc in marks.items():
        lines.append(f"| {sub} | {sc} | {calculate_grade(sc)} |")
    lines.append("")
    lines.append(f"Overall: {avg:.2f}% → Grade {grade}")
    return "\n".join(lines)


# ---------------------------
# Tool Definitions
# ---------------------------


def positive_prompt_tool(text: str, session_id: str):
    """
    Motivational and upbeat responses.
    Used when classifier detects a positive emotional context.
    """
    prompt = f"""
Conversation:
{get_recent_history(session_id)}

User: {text}

Respond supportive and uplifting in 2 sentences.
"""
    return llm.invoke(prompt).content.strip()


def negative_prompt_tool(text: str, session_id: str):
    """
    Empathy + one quick actionable suggestion.
    """
    prompt = f"""
Conversation:
{get_recent_history(session_id)}

User: {text}

Respond calm with one simple helpful suggestion. 2 sentences.
"""
    return llm.invoke(prompt).content.strip()


def student_marks_tool(text: str, session_id: str):
    """
    Academic assistant tool.
    If numbers detected → update marks
    Else → display the current grade table
    """
    if re.search(r"\d", text):
        return add_or_update_marks(text, session_id)
    return show_marks_table(session_id)


def self_harm_safety_tool(_: str, session_id: str):
    """
    Emergency supportive response when user expresses direct personal intent for self-harm.
    """
    return (
        "I am sorry you are feeling like this.\n"
        "Please reach out to someone who can support you.\n"
        "India support lines: Aasra +91 9820466726 | iCall 022-25521111"
    )


def clarification_question(_: str, __: str):
    """
    Used only when safety intent is unclear.
    Helps prevent false emergency triggers.
    """
    return "Are you talking about yourself?"


# ---------------------------
# Agent Setup (fallback generic handling)
# ---------------------------


def get_agent(session_id: str):
    """
    Builds and caches a LangChain agent for generic queries with:
    - Conversation memory
    - Tool access
    """
    if session_id in agent_store:
        return agent_store[session_id]

    memory = get_memory(session_id)
    tools = [
        Tool(
            "PositiveResponse",
            lambda t: positive_prompt_tool(t, session_id),
            "Positive feelings",
        ),
        Tool(
            "NegativeResponse",
            lambda t: negative_prompt_tool(t, session_id),
            "Negative feelings",
        ),
        Tool(
            "AcademicHelper",
            lambda t: student_marks_tool(t, session_id),
            "Scores & results",
        ),
        Tool(
            "Safety",
            lambda t: self_harm_safety_tool(t, session_id),
            "Self-harm emergency",
        ),
        Tool(
            "ClarifyIntent",
            lambda t: clarification_question(t, session_id),
            "Ask if message refers to self",
        ),
    ]

    agent = initialize_agent(
        tools,
        llm,
        memory=memory,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=False,
    )
    agent_store[session_id] = agent
    return agent


# ---------------------------
# Semantic Intent Classifier
# ---------------------------


def classify_intent(text: str) -> str:
    """
    LLM classifies message meaning → determines correct tool.
    Suicide context is only 'safety' when intent is real and personal.
    """
    prompt = f"""
Classify intent of the message:

"{text}"

Rules:
- Only classify "safety" if the speaker expresses real personal intent to self-harm.
  Examples: "I want to die", "I will kill myself", "I plan to end my life"
- If suicide/self-harm is mentioned but about a movie, story, someone else, or joking → NOT safety.
- If emotional struggle but not explicit intent → negative.
- If clearly happy → positive.
- If message about marks/grades → academic.
- Else → generic.

Return one word:
academic | positive | negative | safety | generic | unclear
"""
    return llm.invoke(prompt).content.strip().lower()


# ---------------------------
# Name Handling
# ---------------------------


def extract_name(text: str) -> str:
    """
    Extract user's name if they introduce themselves.
    Fallback: use last token as a name.
    """
    m = re.search(r"(?:my name is|i am|i'm|this is)\s+([A-Za-z]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).title()
    return (text.strip().split()[-1] or "Friend").title()


def set_name(name: str, session_id: str = "default") -> str:
    """
    Store user's name and acknowledge properly.
    """
    session = get_or_create_session(session_id)
    clean = extract_name(name)
    session["name"] = clean

    get_memory(session_id).save_context(
        {"input": name}, {"output": f"Stored name: {clean}."}
    )
    return f"Nice to meet you, {clean}. What should we do next?"


# ---------------------------
# Main Chat Router
# ---------------------------


def chat(message: str, session_id: str = "default") -> str:
    """
    Main entrypoint:
    - Name collection (first message)
    - Semantic intent routing (classification)
    - Tool chosen by context, not keywords
    - Safety clarification when uncertain
    """
    session = get_or_create_session(session_id)
    name = session.get("name")
    agent = get_agent(session_id)

    text = (message or "").strip()
    if not text:
        return "Please type something."

    if name is None:
        # First message → assume name
        return set_name(text, session_id)

    # Classify semantic meaning of the message
    intent = classify_intent(text)

    # Route based on model intelligence
    if intent == "academic":
        reply = student_marks_tool(text, session_id)
    elif intent == "positive":
        reply = positive_prompt_tool(text, session_id)
    elif intent == "negative":
        reply = negative_prompt_tool(text, session_id)
    elif intent == "safety":
        reply = self_harm_safety_tool(text, session_id)
    elif intent == "unclear":
        reply = clarification_question(text, session_id)
    else:
        # Generic fallback → agent thinking + tools
        reply = agent.run(text)

    # Save response to memory for conversation context
    get_memory(session_id).save_context({"input": text}, {"output": reply})
    return reply
