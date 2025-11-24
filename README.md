
# Study Buddy — LangChain AI Mentor

Study Buddy is a LangChain-based multi-tool agent that combines FastAPI on the backend and Gradio for a modern chat interface. It uses ConversationTokenBufferMemory to maintain context, enabling accurate responses across positive and negative emotional support, academic performance tracking, and personalized assistance with grades and marks.

---

## Features

* Enter marks in natural sentences
* Auto-calculated grade table with average and scale
* Personalized encouraging / supportive responses
* Safety detection for self-harm keywords
* Per-session memory with unique session ID
* Clean, responsive Gradio web interface

---

## Project Structure

```
project-root/
│
├── bot.py
├── main.py
├── frontend_gradio.py
├── requirements.txt
└── assets/
    ├── chatbot-start.png
    ├── chatbot-marks.png
    └── chatbot-concern.png
```

---

## 🔧 Requirements

* Python 3.9+
* Valid GROQ_API_KEY inside a `.env` file

Example `.env`:

```
GROQ_API_KEY=your_api_key_here
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the App

### 1) Start FastAPI backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Verify:

```
GET http://127.0.0.1:8000/
→ {"message": "Study Buddy Agent API running"}
```

---

### 2) Launch Web UI (Gradio)

Open another terminal:

```bash
python frontend_gradio.py
```

UI opens at:

```
http://127.0.0.1:7860
```

---

## 💬 How To Use

Example interaction:

```
👤: Hi I am Rahul
🤖: Nice to meet you, Rahul…

👤: Maths - 91, Physics 80
🤖: Grade table + average score

👤: Show my grades
🤖: Displays saved table
```

Self-harm concerns always switch to an immediate safety response.

---

## Screenshots

### Welcome Chat View
![Chat Start](chatbot-start.png)

### Grades Table Display
![Grades Table](chatbot-marks.png)

### Emotion Based Response
![Emotion Response](chatbot-concern.png)
