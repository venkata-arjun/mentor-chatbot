from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

# Load environment variables (requires GROQ_API_KEY for LLM use)
load_dotenv()

from bot import chat, set_name  # Import core chatbot logic

# Create FastAPI application instance
app = FastAPI(title="Study Buddy API", version="2.0 - Agent + Grades")


class MessageRequest(BaseModel):
    # Request model for general chat messages
    message: str
    session_id: str = "default"


class NameRequest(BaseModel):
    # Request model for name-setting messages
    name: str
    session_id: str = "default"


@app.post("/chat")
async def chat_endpoint(req: MessageRequest):
    # Route to chatbot handler for user messages
    reply = chat(req.message, req.session_id)
    return {"reply": reply}


@app.post("/set-name")
async def set_name_endpoint(req: NameRequest):
    # Route to explicitly assign user name
    reply = set_name(req.name, req.session_id)
    return {"reply": reply}


@app.get("/")
async def root():
    # Basic health-check endpoint
    return {"message": "Study Buddy Agent API running"}


if __name__ == "__main__":
    # Local development server entry point
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
