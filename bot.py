# bot.py

import re
from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain.memory import ConversationTokenBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType

# -------------------------------------------------------------------
# LLM SETUP
# -------------------------------------------------------------------
# Requires GROQ_API_KEY in environment (.env is loaded in main.py)
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)

# -------------------------------------------------------------------
# GLOBAL PER-SESSION STATE
# -------------------------------------------------------------------
memory_store: Dict[str, ConversationTokenBufferMemory] = {}
sessions: Dict[str, Dict[str, Any]] = (
    {}
)  # {session_id: {"name": str | None, "marks": dict}}
agent_store: Dict[str, Any] = {}  # {session_id: langchain agent}


def get_memory(session_id: str) -> ConversationTokenBufferMemory:
    if session_id not in memory_store:
        memory_store[session_id] = ConversationTokenBufferMemory(
            llm=llm,
            max_token_limit=8000,
            return_messages=True,
            memory_key="chat_history",
        )
    return memory_store[session_id]


def get_recent_history(session_id: str) -> str:
    memory = get_memory(session_id)
    msgs = memory.load_memory_variables({}).get("chat_history", []) or []
    lines = []
    for m in msgs:
        role = getattr(m, "type", "")
        who = "USER" if role == "human" else "BOT"
        lines.append(f"{who}: {m.content}")
    return "\n".join(lines)


def get_or_create_session(session_id: str) -> Dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = {"name": None, "marks": {}}
    return sessions[session_id]


# -------------------------------------------------------------------
# GRADE ENGINE (YOUR SCALE)
# -------------------------------------------------------------------
def calculate_grade(score: int) -> str:
    if score >= 90:
        return "S"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    elif score >= 40:
        return "E"
    else:
        return "F"


def parse_subject_scores(text: str):
    """
    Extract (subject, score) pairs from flexible text patterns.
    Supports:
      - "Maths - 90, Sci - 98"
      - "I scored 90 in Maths"
      - "Maths 90 Physics 80"
    """
    pairs = []

    # Pattern: "subject - score"
    for subj, score in re.findall(r"([A-Za-z ]+)\s*[-=:]\s*(\d{1,3})", text):
        subject_clean = subj.strip().title()
        score_int = int(score)
        pairs.append((subject_clean, score_int))

    # Pattern: "score in subject"
    for score, subj in re.findall(
        r"(\d{1,3})\s+in\s+([A-Za-z ]+)", text, flags=re.IGNORECASE
    ):
        subject_clean = subj.strip().title()
        score_int = int(score)
        pairs.append((subject_clean, score_int))

    # Fallback: "subject score"
    if not pairs:
        tokens = text.replace(",", " ").split()
        for i in range(len(tokens) - 1):
            if tokens[i + 1].isdigit():
                subject_clean = tokens[i].strip().title()
                score_int = int(tokens[i + 1])
                pairs.append((subject_clean, score_int))

    return pairs


def add_or_update_marks(text: str, session_id: str) -> str:
    session = get_or_create_session(session_id)
    marks = session["marks"]

    pairs = parse_subject_scores(text)
    if not pairs:
        return "I couldn’t find any marks. Try like: 'Maths - 91, Physics - 80'."

    for subject, score in pairs:
        score = max(0, min(int(score), 100))
        marks[subject] = score

    # Build table
    lines = [
        "Here are your updated grades:\n",
        "| Subject | Marks | Grade |",
        "|--------|-------|-------|",
    ]
    total = 0
    for subject, score in marks.items():
        grade = calculate_grade(score)
        total += score
        lines.append(f"| {subject} | {score} | {grade} |")

    avg = total / len(marks) if marks else 0
    overall_grade = calculate_grade(int(round(avg)))

    lines.append("")
    lines.append(f"Overall: **{avg:.2f}% → Grade {overall_grade}**.")

    if avg >= 80:
        lines.append("Great work. That’s the right direction—keep pushing yourself.")
    elif avg >= 60:
        lines.append(
            "Good progress. With steady effort, you can push this even higher."
        )
    elif avg >= 40:
        lines.append("You’re passing. Let’s focus on lifting the weaker subjects next.")
    else:
        lines.append(
            "It’s okay to have low scores sometimes. We can build a better plan from here."
        )

    lines.append("\n(Grade scale: S≥90, A≥80, B≥70, C≥60, D≥50, E≥40, F<40)")
    return "\n".join(lines)


def show_marks_table(session_id: str) -> str:
    session = get_or_create_session(session_id)
    marks = session["marks"]

    if not marks:
        return "I don’t have any marks saved yet. Tell me your scores like: 'Maths - 91, Physics - 80'."

    lines = [
        "Here is your grade summary:\n",
        "| Subject | Marks | Grade |",
        "|--------|-------|-------|",
    ]
    total = 0
    for subject, score in marks.items():
        grade = calculate_grade(score)
        total += score
        lines.append(f"| {subject} | {score} | {grade} |")

    avg = total / len(marks)
    overall_grade = calculate_grade(int(round(avg)))

    lines.append("")
    lines.append(f"Overall: **{avg:.2f}% → Grade {overall_grade}**.")
    lines.append("\n(Grade scale: S≥90, A≥80, B≥70, C≥60, D≥50, E≥40, F<40)")

    return "\n".join(lines)


# -------------------------------------------------------------------
# TOOLS (CORE LOGIC)
# -------------------------------------------------------------------
def positive_prompt_tool(text: str, session_id: str) -> str:
    history = get_recent_history(session_id)
    prompt = f"""
You are a motivating study mentor.

Conversation so far:
{history}

User: {text}

Goals:
- Acknowledge their positive feeling.
- Sound energetic but not cringe.
- Focus only on the user (use "you", not "I").
- NEVER talk about your own feelings or what "someone told you".
- End with one mentor-style question like:
  "What achievement made your day?" or
  "What are you most proud of today?"
Keep it within 2–3 sentences.
"""
    return llm.invoke(prompt).content.strip()


def negative_prompt_tool(text: str, session_id: str) -> str:
    history = get_recent_history(session_id)
    prompt = f"""
You are a calm, motivating mentor.

Conversation so far:
{history}

User: {text}

Goals:
- Acknowledge their feeling (stress, sadness, etc.).
- Normalize the struggle (it's okay, it happens).
- Suggest one small, specific action they can take today to feel more in control.
- Keep the focus on "you", not "I".
- Keep it concise (2–3 sentences).
"""
    return llm.invoke(prompt).content.strip()


def student_marks_tool(text: str, session_id: str) -> str:
    """
    Deterministic academic helper:
    - If text contains numbers → new / update marks
    - If asking for grades/average/table/scale → show marks or scale
    """
    lower = text.lower()

    # Direct grade scale / criteria question
    if any(kw in lower for kw in ["scale", "grading", "criteria", "grade range"]):
        return "(Grade scale: S≥90, A≥80, B≥70, C≥60, D≥50, E≥40, F<40)"

    has_number = bool(re.search(r"\d", text))
    wants_report = any(
        key in lower
        for key in [
            "grade",
            "grades",
            "average",
            "marks",
            "mark",
            "result",
            "report",
            "table",
            "summary",
        ]
    )

    if has_number:
        return add_or_update_marks(text, session_id)

    if wants_report:
        return show_marks_table(session_id)

    return "Tell me your marks like: 'Maths - 91, Physics - 80' and I’ll calculate grades for you."


def self_harm_safety_tool(_: str, session_id: str) -> str:
    return (
        "I’m really sorry you’re feeling like this.\n"
        "I’m not able to help directly, but you should reach out to someone who can support you right now.\n"
        "India → Aasra: +91 9820466726 | iCall: 022-25521111\n"
        "You’re not alone. Please talk to someone immediately."
    )


# -------------------------------------------------------------------
# LANGCHAIN AGENT (FALLBACK FOR GENERIC QUERIES)
# -------------------------------------------------------------------
def get_agent(session_id: str):
    """
    Creates one LangChain agent per session.
    Tools are session-bound (no global current_session_id), so
    multiple users can talk in parallel without clashing.
    """
    if session_id in agent_store:
        return agent_store[session_id]

    memory = get_memory(session_id)

    # Session-bound wrappers
    def positive_wrapper(t: str, _sid=session_id) -> str:
        return positive_prompt_tool(t, _sid)

    def negative_wrapper(t: str, _sid=session_id) -> str:
        return negative_prompt_tool(t, _sid)

    def marks_wrapper(t: str, _sid=session_id) -> str:
        return student_marks_tool(t, _sid)

    def safety_wrapper(_: str, _sid=session_id) -> str:
        return self_harm_safety_tool("", _sid)

    tools = [
        Tool(
            name="PositiveResponse",
            func=positive_wrapper,
            description="Use for happy, excited, proud, or positive messages.",
        ),
        Tool(
            name="NegativeResponse",
            func=negative_wrapper,
            description="Use for sad, stressed, anxious, lonely, or negative messages.",
        ),
        Tool(
            name="AcademicHelper",
            func=marks_wrapper,
            description="Use for anything about marks, grades, scores, subjects, averages, or academic performance.",
        ),
        Tool(
            name="Safety",
            func=safety_wrapper,
            description="Use if the user mentions suicide, self-harm, or wanting to die.",
        ),
    ]

    agent = initialize_agent(
        tools,
        llm,
        memory=memory,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
    )
    agent_store[session_id] = agent
    return agent


# -------------------------------------------------------------------
# NAME HANDLING
# -------------------------------------------------------------------
def extract_name(text: str) -> str:
    """
    Extract name from phrases like:
    - 'hello I am Arjun'
    - 'my name is Arjun'
    - 'I'm Arjun'
    Fallback: last word.
    """
    m = re.search(r"(?:my name is|i am|i'm|this is)\s+([A-Za-z]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).title()

    tokens = text.strip().split()
    return tokens[-1].title() if tokens else "Friend"


def set_name(name: str, session_id: str = "default") -> str:
    session = get_or_create_session(session_id)
    clean = extract_name(name)
    session["name"] = clean

    memory = get_memory(session_id)
    memory.save_context(
        {"input": f"My name is {clean}"},
        {"output": f"Stored name: {clean}."},
    )

    return f"Nice to meet you, {clean}. What would you like to work on today?"


# -------------------------------------------------------------------
# MAIN CHAT ROUTER
# -------------------------------------------------------------------
SUICIDE_KEYWORDS = [
    "suicide",
    "kill myself",
    "want to die",
    "end my life",
    "hurt myself",
    "no point living",
    "cut myself",
]


def chat(message: str, session_id: str = "default") -> str:
    agent = get_agent(session_id)
    session = get_or_create_session(session_id)
    name = session.get("name")

    text = (message or "").strip()
    if not text:
        return "Please type something."

    lower = text.lower()

    # 1) Safety always overrides everything
    if any(kw in lower for kw in SUICIDE_KEYWORDS):
        reply = self_harm_safety_tool(text, session_id)
        mem = get_memory(session_id)
        mem.save_context({"input": text}, {"output": reply})
        return reply

    # 2) Exit phrases
    if any(
        exit_word in lower for exit_word in ["bye", "goodbye", "see you", "take care"]
    ):
        return (
            f"Bye {name or 'Friend'}. Keep going—you’re capable of more than you think."
        )

    # 3) Name phase
    if name is None:
        return set_name(text, session_id)

    # 4) Academic (marks / grades) → deterministic tool
    if re.search(r"\d", text) or any(
        key in lower
        for key in [
            "grade",
            "grades",
            "score",
            "scores",
            "marks",
            "mark",
            "average",
            "result",
            "report",
            "table",
        ]
    ):
        reply = student_marks_tool(text, session_id)
        mem = get_memory(session_id)
        mem.save_context({"input": text}, {"output": reply})
        return reply

    # 5) Positive emotion → positive tool
    if any(
        word in lower
        for word in ["happy", "excited", "great", "awesome", "glad", "delighted"]
    ):
        reply = positive_prompt_tool(text, session_id)
        mem = get_memory(session_id)
        mem.save_context({"input": text}, {"output": reply})
        return reply

    # 6) Negative emotion → negative tool
    if any(
        word in lower
        for word in [
            "sad",
            "upset",
            "depressed",
            "stressed",
            "anxious",
            "worried",
            "lonely",
            "tired",
            "angry",
            "frustrated",
            "overwhelmed",
        ]
    ):
        reply = negative_prompt_tool(text, session_id)
        mem = get_memory(session_id)
        mem.save_context({"input": text}, {"output": reply})
        return reply

    # 7) Fallback → LangChain Agent (for generic queries)
    try:
        return agent.run(text)  # deprecation warning is fine for now
    except Exception as e:
        print("Agent error:", repr(e))
        return "I had trouble understanding that. Try rephrasing or ask something else."
