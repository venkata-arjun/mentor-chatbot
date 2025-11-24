import uuid
from typing import List, Tuple, Union

import gradio as gr
import requests

# Backend API endpoint configuration
API_BASE_URL = "http://127.0.0.1:8000"
TIMEOUT_SEC = 30  # Request timeout for backend calls


HistoryTuple = Tuple[str, str]
HistoryType = List[Union[HistoryTuple, dict]]


def respond(message: str, history: HistoryType, session_id: str):
    # Generate new session ID if not yet initialized
    if not session_id:
        session_id = f"user_{uuid.uuid4().hex[:6]}"

    try:
        # Send user message to backend API
        resp = requests.post(
            f"{API_BASE_URL}/chat",
            json={"message": message, "session_id": session_id},
            timeout=TIMEOUT_SEC,
        )
        resp.raise_for_status()
        reply = resp.json().get("reply", "No response.")

        # Maintain history format properly depending on type structure
        if isinstance(history, list) and (not history or isinstance(history[0], dict)):
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": reply})
        else:
            history.append((message, reply))

        # Clear text input and return updated chat state
        return "", history, session_id

    except Exception:
        # Handle backend connection failure gracefully
        if isinstance(history, list) and (not history or isinstance(history[0], dict)):
            history.append({"role": "user", "content": message})
            history.append(
                {"role": "assistant", "content": "Error: Backend not available."}
            )
        else:
            history.append((message, "Error: Backend not available."))

        return "", history, session_id


def init_session_id() -> str:
    # Create a unique session ID when UI loads
    return f"user_{uuid.uuid4().hex[:6]}"


# Build Gradio UI layout
with gr.Blocks(title="Study Buddy") as demo:

    # App heading section
    gr.Markdown(
        """
        <h1 style="text-align: center; font-weight: 600; margin-bottom: 8px;">
            Study Buddy
        </h1>
        <p style="text-align: center; font-size: 14px; margin-top: 0;">
            Your personal learning companion
        </p>
        """
    )

    with gr.Column():
        # Chat display area with initial bot greeting
        chatbot = gr.Chatbot(
            height=600,
            label=None,
            value=[
                {
                    "role": "assistant",
                    "content": (
                        "I am Study Buddy, your academic and wellness companion.\n"
                        "What should I call you?"
                    ),
                }
            ],
        )

        # User input box for sending messages
        msg = gr.Textbox(
            show_label=False,
            placeholder="Type here…",
        )

    # Store session ID invisibly
    session_state = gr.State()

    # Generate session ID at UI load
    demo.load(init_session_id, inputs=None, outputs=session_state)

    # Submit message and update chat on Enter key
    msg.submit(respond, [msg, chatbot, session_state], [msg, chatbot, session_state])

    # Focus cursor into textbox for smoother first interaction
    try:
        msg.focus()
    except Exception:
        pass


if __name__ == "__main__":
    # Run Gradio on local network with share link enabled
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
