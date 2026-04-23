"""
AutoStream Conversational AI Agent
Built with LangGraph + Gemini 1.5 Flash + RAG
ServiceHive Inflx Assignment — Akshay Aramani
"""
from dotenv import load_dotenv
import os

load_dotenv()
print("API KEY:", os.environ.get("GOOGLE_API_KEY"))
import json
import os
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ─────────────────────────────────────────────
# LOAD KNOWLEDGE BASE (RAG)
# ─────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.3
)
def load_knowledge_base() -> str:
    """Load AutoStream knowledge base from JSON file."""
    with open("knowledge_base.json", "r") as f:
        kb = json.load(f)

    # Convert JSON to readable text for RAG context
    kb_text = f"""
AUTOSTREAM PRODUCT KNOWLEDGE BASE
===================================

PRODUCT: {kb['product']}
{kb['tagline']}

PRICING PLANS:
--------------
1. {kb['plans']['basic']['name']} — {kb['plans']['basic']['price']}
   Features: {', '.join(kb['plans']['basic']['features'])}

2. {kb['plans']['pro']['name']} — {kb['plans']['pro']['price']}
   Features: {', '.join(kb['plans']['pro']['features'])}

COMPANY POLICIES:
-----------------
- Refund Policy: {kb['policies']['refund']}
- Support: {kb['policies']['support']}
- Trial: {kb['policies']['trial']}
- Cancellation: {kb['policies']['cancellation']}


FREQUENTLY ASKED QUESTIONS:
----------------------------
"""
    for faq in kb['faq']:
        kb_text += f"Q: {faq['q']}\nA: {faq['a']}\n\n"

    return kb_text


KNOWLEDGE_BASE = load_knowledge_base()


# ─────────────────────────────────────────────
# MOCK LEAD CAPTURE TOOL
# ─────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Mock API function to capture lead details.
    Only called after all three fields are collected.
    """
    print(f"\n{'='*50}")
    print(f"✅ LEAD CAPTURED SUCCESSFULLY!")
    print(f"   Name:     {name}")
    print(f"   Email:    {email}")
    print(f"   Platform: {platform}")
    print(f"{'='*50}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"


# ─────────────────────────────────────────────
# STATE DEFINITION (LangGraph)
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    """
    State maintained across all conversation turns.
    This is what LangGraph uses to remember context.
    """
    messages: Annotated[List, add_messages]  # Full conversation history
    intent: str                               # Current detected intent
    lead_name: str                            # Collected lead name
    lead_email: str                           # Collected lead email
    lead_platform: str                        # Collected lead platform
    lead_captured: bool                       # Whether lead was captured
    awaiting_field: str                       # Which field we're waiting for


# ─────────────────────────────────────────────
# INITIALIZE LLM (Gemini 1.5 Flash)
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# NODE 1: INTENT DETECTION
# ─────────────────────────────────────────────
def detect_intent(state: AgentState) -> AgentState:

    # If already collecting lead, stay in that flow
    if state.get("awaiting_field") and not state.get("lead_captured"):
        return {**state, "intent": "lead_collection"}

    last_message = state["messages"][-1].content

    intent_prompt = f"""
You are an intent classifier.

Classify the user message into ONE of these:
- greeting
- inquiry
- high_intent

User message: "{last_message}"

Reply ONLY one word.
"""

    response = llm.invoke([HumanMessage(content=intent_prompt)])

    # 🔥 FIX: handle different response types
    intent = response.content

    if isinstance(intent, list):
        intent = intent[0]

    intent = str(intent).lower()   # safe conversion

    # 🔥 Clean output
    if "high_intent" in intent or "high intent" in intent:
        intent = "high_intent"
    elif "inquiry" in intent:
        intent = "inquiry"
    else:
        intent = "greeting"

    return {**state, "intent": intent}
def rag_response(state: AgentState) -> AgentState:
    """
    Answers product/pricing questions using the knowledge base.
    This is the RAG node — retrieves from KB and generates answer.
    """
    last_message = state["messages"][-1].content

    system_prompt = f"""You are AutoStream's helpful sales assistant.
Answer the user's question using ONLY the information in the knowledge base below.
Be friendly, concise, and helpful. If something is not in the knowledge base, say you don't have that info.

KNOWLEDGE BASE:
{KNOWLEDGE_BASE}
"""

    messages = [SystemMessage(content=system_prompt)] + [
        m for m in state["messages"]
    ]

    response = llm.invoke(messages)

    new_messages = state["messages"] + [AIMessage(content=response.content)]
    return {**state, "messages": new_messages}


# ─────────────────────────────────────────────
# NODE 3: GREETING RESPONSE
# ─────────────────────────────────────────────

def greeting_response(state: AgentState) -> AgentState:
    """Handles casual greetings."""
    last_message = state["messages"][-1].content

    system_prompt = """You are AutoStream's friendly AI assistant.
Greet the user warmly and let them know you can help with:
- Pricing information
- Product features
- Getting started with AutoStream
Keep it brief and inviting."""

    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)

    new_messages = state["messages"] + [AIMessage(content=response.content)]
    return {**state, "messages": new_messages}


# ─────────────────────────────────────────────
# NODE 4: HIGH INTENT — Start Lead Capture
# ─────────────────────────────────────────────

def start_lead_capture(state: AgentState) -> AgentState:
    """
    Triggered when high intent is detected.
    Starts collecting lead information one field at a time.
    """
    response_text = """That's great to hear! 🎉 I'd love to get you set up with AutoStream.

To get started, I just need a few quick details.

**What's your full name?**"""

    new_messages = state["messages"] + [AIMessage(content=response_text)]
    return {
        **state,
        "messages": new_messages,
        "awaiting_field": "name",
        "intent": "lead_collection"
    }


# ─────────────────────────────────────────────
# NODE 5: LEAD COLLECTION — Collect fields one by one
# ─────────────────────────────────────────────

def collect_lead_info(state: AgentState) -> AgentState:
    """
    Collects lead information across multiple turns.
    Asks for name → email → platform sequentially.
    Only calls mock_lead_capture() after ALL three are collected.
    """
    last_message = state["messages"][-1].content.strip()
    awaiting = state.get("awaiting_field", "name")

    lead_name = state.get("lead_name", "")
    lead_email = state.get("lead_email", "")
    lead_platform = state.get("lead_platform", "")

    response_text = ""
    new_awaiting = awaiting

    # ── Collect Name ──
    if awaiting == "name":
        lead_name = last_message
        response_text = f"Nice to meet you, **{lead_name}**! 😊\n\n**What's your email address?**"
        new_awaiting = "email"

    # ── Collect Email ──
    elif awaiting == "email":
        # Basic email validation
        if "@" in last_message and "." in last_message:
            lead_email = last_message
            response_text = f"Perfect! ✅\n\n**Which platform are you creating content for?**\n(e.g. YouTube, Instagram, TikTok, Twitter/X)"
            new_awaiting = "platform"
        else:
            response_text = "That doesn't look like a valid email. Could you please re-enter your email address? 📧"
            new_awaiting = "email"

    # ── Collect Platform ──
    elif awaiting == "platform":
        lead_platform = last_message
        new_awaiting = "done"

        # ── ALL THREE COLLECTED — Fire mock_lead_capture() ──
        capture_result = mock_lead_capture(lead_name, lead_email, lead_platform)

        response_text = f"""🎉 **You're all set, {lead_name}!**

Here's a summary of your details:
- 📧 Email: {lead_email}
- 🎬 Platform: {lead_platform}

Our team will reach out to you shortly to help you get started with AutoStream Pro!

Is there anything else I can help you with? 😊"""

    new_messages = state["messages"] + [AIMessage(content=response_text)]

    return {
        **state,
        "messages": new_messages,
        "lead_name": lead_name,
        "lead_email": lead_email,
        "lead_platform": lead_platform,
        "awaiting_field": new_awaiting,
        "lead_captured": new_awaiting == "done"
    }


# ─────────────────────────────────────────────
# ROUTER — decides which node to go to
# ─────────────────────────────────────────────

def route_intent(state: AgentState) -> str:
    """Routes to correct node based on detected intent."""
    intent = state.get("intent", "greeting")
    awaiting = state.get("awaiting_field", "")
    lead_captured = state.get("lead_captured", False)

    # If we're in lead collection flow and not done
    if awaiting and awaiting != "done" and not lead_captured:
        return "collect_lead_info"

    if intent == "high_intent":
        return "start_lead_capture"
    elif intent == "inquiry":
        return "rag_response"
    else:
        return "greeting_response"


# ─────────────────────────────────────────────
# BUILD LANGGRAPH
# ─────────────────────────────────────────────

def build_agent():
    """Builds and compiles the LangGraph agent."""

    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("greeting_response", greeting_response)
    graph.add_node("rag_response", rag_response)
    graph.add_node("start_lead_capture", start_lead_capture)
    graph.add_node("collect_lead_info", collect_lead_info)

    # Entry point
    graph.set_entry_point("detect_intent")

    # Conditional routing after intent detection
    graph.add_conditional_edges(
        "detect_intent",
        route_intent,
        {
            "greeting_response": "greeting_response",
            "rag_response": "rag_response",
            "start_lead_capture": "start_lead_capture",
            "collect_lead_info": "collect_lead_info"
        }
    )

    # All response nodes end the graph turn
    graph.add_edge("greeting_response", END)
    graph.add_edge("rag_response", END)
    graph.add_edge("start_lead_capture", END)
    graph.add_edge("collect_lead_info", END)

    return graph.compile()


# ─────────────────────────────────────────────
# MAIN CONVERSATION LOOP
# ─────────────────────────────────────────────

def main():
    """Main conversation loop."""

    print("\n" + "="*55)
    print("  🎬 AutoStream AI Assistant")
    print("  Powered by Inflx (ServiceHive)")
    print("="*55)
    print("  Type 'quit' to exit\n")

    # Initialize state
    state: AgentState = {
        "messages": [],
        "intent": "",
        "lead_name": "",
        "lead_email": "",
        "lead_platform": "",
        "lead_captured": False,
        "awaiting_field": ""
    }
    agent = build_agent()

    while True:
        # Get user input
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\nAutoStream: Thanks for chatting! Have a great day! 👋\n")
            break

        if not user_input:
            continue
        # Add user message to state
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        # Run the agent
        try:
            state = agent.invoke(state)

            # Get last AI message
            last_ai_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    last_ai_message = msg
                    break

            if last_ai_message:
                content = last_ai_message.content

                # Handle Gemini structured output
                if isinstance(content, list):
                    content = content[0].get("text", "")

                content = str(content)

                print(f"\nAutoStream: {content}\n")

        except Exception as e:
            print(f"\n⚠️ Error: {str(e)}\n")
            print("Please check your setup and try again.\n")

if __name__ == "__main__":
    main()