from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
import os
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GitHub-Themed Gemini Chatbot",
    description="A modern chatbot with GitHub-inspired UI powered by Google's Gemini",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validate API key at startup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set")
    raise ValueError("GEMINI_API_KEY environment variable not set")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Test the connection
    test_model = genai.GenerativeModel('gemini-pro')
    test_response = test_model.generate_content("Test connection")
    logger.info("Successfully connected to Gemini API")
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {str(e)}")
    raise

class ChatMessage(BaseModel):
    content: str
    is_user: bool = False

class ChatRequest(BaseModel):
    message: str
    chat_history: List[ChatMessage] = []

class ErrorResponse(BaseModel):
    error: str
    details: str = None
    status: str = "error"

@app.post("/api/chat", response_model=dict, responses={
    400: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    502: {"model": ErrorResponse}
})
async def chat_with_gemini(request: ChatRequest):
    try:
        logger.info(f"Received chat request: {request.message[:50]}...")
        
        if not request.message.strip():
            logger.warning("Empty message received")
            raise HTTPException(
                status_code=400,
                detail={"error": "Validation Error", "details": "Message cannot be empty"}
            )
        
        # Build conversation history
        conversation = []
        for msg in request.chat_history:
            role = "user" if msg.is_user else "model"
            conversation.append({"role": role, "parts": [msg.content]})
        
        # Add new user message
        conversation.append({"role": "user", "parts": [request.message]})
        
        # Generate response
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(conversation)
            
            if not response or not hasattr(response, 'text'):
                logger.error("Invalid response format from Gemini API")
                raise HTTPException(
                    status_code=502,
                    detail={"error": "API Error", "details": "Invalid response from Gemini API"}
                )
            
            logger.info("Successfully generated response")
            return {
                "response": response.text,
                "status": "success"
            }
            
        except Exception as api_error:
            logger.error(f"Gemini API error: {str(api_error)}")
            raise HTTPException(
                status_code=502,
                detail={"error": "API Error", "details": str(api_error)}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "Server Error", "details": "Failed to process chat request"}
        )

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    try:
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        logger.error("index.html file not found")
        raise HTTPException(
            status_code=404,
            detail={"error": "File Not Found", "details": "index.html not found"}
        )
    except Exception as e:
        logger.error(f"Error loading interface: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Server Error", "details": "Failed to load chat interface"}
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail.get("error", "Error"),
            "details": exc.detail.get("details", ""),
            "status": "error"
        }
    )
