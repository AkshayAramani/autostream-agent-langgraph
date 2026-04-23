# 🎬 AutoStream Conversational AI Agent

### ServiceHive Inflx Assignment — Built by Akshay Aramani

---

## 🚀 Overview

This project is an **agentic AI assistant** built for a fictional SaaS product **AutoStream** — an automated video editing platform for content creators.

The goal of this project is to simulate a **real-world business chatbot** that can:

* Answer product-related questions
* Identify high-intent users
* Capture leads automatically

This mimics how modern companies use AI for **sales + support automation**.

---

## ⚙️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/AkshayAramani/autostream-agent
cd autostream-agent
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API Key

* Get your Gemini API key: https://aistudio.google.com/app/apikey
* Add to `.env`:

```env
GOOGLE_API_KEY=your-key-here
```

### 5. Run the agent

```bash
python agent.py
```

---

## 🧠 How It Works (Architecture)

This project uses **LangGraph** to build a multi-step conversational workflow.

Unlike basic chatbots, this agent:

* Maintains conversation state
* Routes logic dynamically
* Handles multi-turn interactions

### 🔹 State Management

All conversation data is stored in a central `AgentState`, including:

* Chat history
* Detected intent
* Lead details (name, email, platform)
* Current step in the flow

---

### 🔹 Agent Flow

1. User sends message

2. `detect_intent` classifies it as:

   * Greeting
   * Inquiry
   * High Intent

3. Based on intent:

   * Greeting → friendly response
   * Inquiry → answered using knowledge base (RAG)
   * High Intent → triggers lead capture

4. Lead flow:

```
Name → Email → Platform
```

5. Once completed → lead is captured

---

### 🔹 RAG Approach

Instead of a full vector database, this project uses a **lightweight RAG setup**:

* Product data stored in `knowledge_base.json`
* Injected into prompt for contextual answers

---

## 💬 Example Conversation

```
You: What's the difference between Basic and Pro?
AutoStream: Basic ($29) includes 10 videos and 720p...
Pro ($79) includes unlimited videos, 4K, AI captions...

You: I want to buy Pro
AutoStream: Great! What's your name?
...
```

---

## 📲 WhatsApp Deployment (Concept)

This agent can be deployed on WhatsApp using webhooks:

* Use Meta WhatsApp Business API
* Store user state in Redis/PostgreSQL
* Process incoming messages via webhook
* Respond using the agent

---

## 🛠️ Tech Stack

| Component       | Technology       |
| --------------- | ---------------- |
| Agent Framework | LangGraph        |
| LLM             | Gemini 1.5 Flash |
| Orchestration   | LangChain        |
| Knowledge Base  | JSON (RAG)       |
| Language        | Python           |

---

## 🎯 Key Learnings

* Building multi-step AI agents using LangGraph
* Handling structured LLM outputs
* Designing conversational workflows
* Implementing simple RAG without vector DB
* Managing state across interactions

---

## 👨‍💻 Author

Akshay Aramani
MBA (Business Analytics)
