import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env file")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

# Initialize FastAPI app
app = FastAPI(title="GenAI Legal Backend")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(file: UploadFile) -> str:
    """Extract text from uploaded PDF using PyPDF2"""
    try:
        reader = PdfReader(file.file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parsing failed: {str(e)}")

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    # Step 1: Validate file
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Step 2: Extract text
    extracted_text = extract_text_from_pdf(file)

    if not extracted_text:
        raise HTTPException(status_code=400, detail="No text found in PDF")

    # Step 3: Construct prompt
    prompt = f"""
Analyze the following legal document/resume.

Return the response in STRICTLY valid JSON format with the following keys:
- summary (string)
- justification (string)
- risk_score (integer between 0 and 100)
- risk_label (string like "Low Risk", "Medium Risk", "High Risk")

Document Text:
----------------
{extracted_text}
"""

    # Step 4: Send to Gemini
    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()

        # Gemini sometimes wraps JSON in markdown, so clean it
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        parsed_response = json.loads(cleaned)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Gemini returned invalid JSON format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini API error: {str(e)}"
        )

    # Step 5: Return structured JSON
    return parsed_response
