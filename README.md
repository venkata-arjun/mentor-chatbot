
## 📌 Study Buddy — AI Mentor with Memory, Emotional Intelligence 

Study Buddy is a FastAPI-powered assistant built using LangChain’s **conversation memory** and a **multi-tool intelligent agent**. It analyzes emotion, understands academic inputs, tracks marks over time, and ensures safe handling of sensitive messages — giving students a consistent, context-aware mentoring experience.

Core chatbot logic lives inside bot.py 
Backend API is main.py 
Frontend UI runs through frontend_gradio.py 
Dependencies in requirements.txt 

---

## ✨ Features

* Enter marks in natural sentences
* Auto-calculated grade table with average and scale
* Personalized encouraging / supportive responses
* Safety detection for self-harm keywords
* Per-session memory with unique session ID
* Clean, responsive Gradio web interface

---

## 📂 Project Structure

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
    └── chatbot-table.png
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

## ▶️ Run the App

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

## 📸 Screenshots

Add your actual files later into the `assets/` folder using these exact names:

```markdown
### Welcome Chat View
![Chat Start](assets/chatbot-start.png)

### Adding Marks Example
![Marks Entry](assets/chatbot-marks.png)

### Grades Table Display
![Grades Table](assets/chatbot-table.png)
```

---

If you want, I can also add a small **system architecture diagram** so companies reviewing your portfolio instantly understand the tech behind it.
