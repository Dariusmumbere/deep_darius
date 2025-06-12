from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai
import os

app = FastAPI(
    title="GitHub-Themed Gemini Chatbot",
    description="A modern chatbot with GitHub-inspired UI powered by Google's Gemini",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

class ChatMessage(BaseModel):
    content: str
    is_user: bool = False

class ChatRequest(BaseModel):
    message: str
    chat_history: list[ChatMessage] = []

@app.post("/api/chat")
async def chat_with_gemini(request: ChatRequest):
    try:
        # Build conversation history
        conversation = []
        for msg in request.chat_history:
            role = "user" if msg.is_user else "model"
            conversation.append({"role": role, "parts": [msg.content]})
        
        # Add new user message
        conversation.append({"role": "user", "parts": [request.message]})
        
        # Generate response
        response = model.generate_content(conversation)
        
        return {
            "response": response.text,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)
