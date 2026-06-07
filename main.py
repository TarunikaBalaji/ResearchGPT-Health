# -*- coding: utf-8 -*-
"""
ResearchGPT-Health FastAPI Backend
==================================
A minimal, production-ready FastAPI backend for clinical literature analysis,
summary generation, automated research gap detection, and context-aware chat.

This backend uses the official Google GenAI SDK to interact with Gemini models.
"""

import io
import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Optional, Literal

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("research-gpt-backend")

# Try importing the official modern google-genai SDK
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    HAS_NEW_SDK = True
except ImportError:
    # Fallback to the legacy google-generativeai SDK if only that is available
    try:
        import google.generativeai as genai_legacy
        HAS_NEW_SDK = False
    except ImportError:
        logger.error("Failed to import both google-genai and google-generativeai packages.")
        HAS_NEW_SDK = False

# Create FastAPI app instance
app = FastAPI(
    title="ResearchGPT-Health API",
    description="Minimal production-ready FastAPI backend for literature synthesis using Gemini.",
    version="1.0.0",
)

# Enable CORS so frontend callers from any domain (e.g. local dev servers) can communicate with it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# GERIATRIC & CLINICAL DATA MODELS (Pydantic Schemas matching frontend)
# =====================================================================

class MethodologySchema(BaseModel):
    studyDesign: str = Field(
        ..., 
        description="e.g., Randomized, Double-blind Clinical Trial, Population-based Cohort Study"
    )
    sampleSize: str = Field(
        ..., 
        description="Number of subjects, controls, and demographic profile (e.g. n=450 patients with Stage III breast cancer)"
    )
    intervention: str = Field(
        ..., 
        description="Active drug, dosage, therapeutic tactic, or care standard applied"
    )
    metrics: List[str] = Field(
        ..., 
        description="Core metrics and clinical endpoints assessed"
    )


class KeyFindingSchema(BaseModel):
    title: str = Field(..., description="A short, descriptive title of the finding")
    description: str = Field(..., description="Detailed quantitative outcomes, hazard ratios, p-values, or clinical percentiles")
    significance: Literal["High", "Medium", "Low"] = Field(
        ..., 
        description="Categorization of clinical relevance to bedside workflows"
    )


class PaperSummarySchema(BaseModel):
    abstract: str = Field(..., description="A high-density 2-3 sentence overview of why this study matters")
    objective: str = Field(..., description="The fundamental hypothesis or objective under investigation")
    methodology: MethodologySchema
    keyFindings: List[KeyFindingSchema] = Field(..., description="Major research endpoints achieved")
    clinicalImplications: List[str] = Field(..., description="Actionable bedside guidelines or clinic changes affected by this work")


class ResearchGapAnalysisSchema(BaseModel):
    limitationOfCurrentStudy: List[str] = Field(..., description="Methodological boundaries, low recruitment cohort, or bias")
    unansweredQuestions: List[str] = Field(..., description="Unexplored clinical paths or biological mechanisms left unexamined")
    methodologicalGaps: List[str] = Field(..., description="Ineffective assays, instrumentation bounds, or delivery limits")
    futureResearchDirections: List[str] = Field(..., description="Actionable proposals for future trials or wet-lab validations")
    priorityScore: int = Field(..., ge=1, le=100, description="1 to 100 severity indicator of research necessity")


class ResearchPaperMetadataSchema(BaseModel):
    title: str = Field(..., description="Academic title of the research paper")
    authors: str = Field(..., description="Authors, comma separated")
    journal: str = Field(..., description="Medical journal name, database, or conference")
    year: int = Field(..., description="Publication year")
    category: Literal["Oncology", "Cardiology", "Neurology", "Genetics", "Public Health", "General Medicine"] = Field(
        ..., 
        description="Broad healthcare category"
    )


# Combined upload output
class ResearchPaperResponse(BaseModel):
    id: str
    title: str
    authors: str
    journal: str
    year: int
    category: Literal["Oncology", "Cardiology", "Neurology", "Genetics", "Public Health", "General Medicine"]
    content: str
    summary: Optional[PaperSummarySchema] = None
    gaps: Optional[ResearchGapAnalysisSchema] = None
    uploadedAt: str


# Chat Requests
class ChatMessage(BaseModel):
    sender: Literal["user", "assistant"]
    text: str


class PaperContext(BaseModel):
    id: str
    title: str
    authors: str
    category: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    papersContext: List[PaperContext] = []


# Direct Input payload falls for upload
class UploadTextRequest(BaseModel):
    filename: Optional[str] = "Paper_Upload.txt"
    text: str


# =====================================================================
# GEMINI API INITIALIZER UTILITY
# =====================================================================

def get_gemini_client():
    """
    Initializes and returns the appropriate Gemini SDK Client.
    Examines environment secrets for direct execution.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key.strip() == "" or api_key == "MY_GEMINI_API_KEY":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The GEMINI_API_KEY environment variable is not configured. "
                   "Please obtain a key from public AI Studio settings and supply it to the environment variables."
        )
    
    if HAS_NEW_SDK:
        try:
            # Modern official google-genai Client
            return genai.Client(
                api_key=api_key,
                http_options={"headers": {"User-Agent": "aistudio-build"}}
            )
        except Exception as e:
            logger.warning(f"Error creating modern google-genai client: {e}. Attempting fallback.")
    
    # Legacy library fallback if required
    try:
        genai_legacy.configure(api_key=api_key)
        return "LEGACY_CLIENT"
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to initialize Gemini API Client or configure fallback: {str(e)}"
        )


# =====================================================================
# PDF & TEXT FILE EXTRACTION PIPELINE
# =====================================================================

def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """
    Extracts plain text from file bytes. Gracefully handles PDFs and raw text.
    """
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    
    if ext == "pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text_parts = []
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            parsed_text = "\n".join(text_parts)
            if parsed_text.strip():
                logger.info(f"Successfully extracted {len(parsed_text)} chars from {filename}")
                return parsed_text
        except ImportError:
            logger.warning("pypdf is not installed. To extract PDFs, run: pip install pypdf or PyPDF2")
        except Exception as e:
            logger.error(f"Error extracting PDF bytes: {e}")
            
    # Default fallback: Treat as raw string or standard decodable text
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_DATA,
            detail=f"Unable to decode text. Please upload plain text or a clean academic PDF: {str(e)}"
        )


def clean_and_parse_json(text_output: str) -> dict:
    """
    Cleans up response string from Gemini and transforms it to active dictionaries,
    handling standard markdown wrappers gracefully.
    """
    cleaned = text_output.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


# =====================================================================
# FASTAPI ENDPOINTS
# =====================================================================

@app.get("/")
def read_root():
    """Root health and directory overview."""
    return {
        "app": "ResearchGPT-Health FastAPI Backend",
        "status": "online",
        "endpoints": [
            "POST /upload-paper (multipart/form or json text payload)",
            "POST /generate-summary (generates complete structured academic synopsis)",
            "POST /detect-gaps (analyzes gaps and outputs methodological critiques)",
            "POST /chat (grounded medical conversation engine)"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/upload-paper", response_model=ResearchPaperResponse)
async def upload_paper(
    file: Optional[UploadFile] = File(None),
    raw_text_payload: Optional[UploadTextRequest] = None
):
    """
    Accepts scientific papers via standard multipart File Upload OR raw text payload.
    Uses Gemini to extract core metadata (title, authors, journal, year, category) and
    structures paper content as a validated clean model.
    """
    content_text = ""
    filename = "paper.txt"
    
    if file is not None:
        filename = file.filename
        file_bytes = await file.read()
        content_text = extract_text_from_bytes(file_bytes, filename)
    elif raw_text_payload is not None:
        filename = raw_text_payload.filename or "paper.txt"
        content_text = raw_text_payload.text
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either a multi-part file 'file' or 'raw_text_payload' JSON is required."
        )
        
    if not content_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Paper content cannot be empty."
        )
        
    # Query Gemini to extract metadata from the body text
    client = get_gemini_client()
    # Snippet used for extracting metadata to avoid overloading prompt limits
    sample_snippet = content_text[:15000]
    
    extraction_prompt = (
        "You are an expert medical librarian and database reviewer.\n"
        "Analyze the following text sample of a research paper and extract its key scientific parameters.\n"
        "Structure the output in JSON matching the specified Pydantic schemas schema constraints.\n"
        "Category must be exactly one of the following: "
        "['Oncology', 'Cardiology', 'Neurology', 'Genetics', 'Public Health', 'General Medicine']\n\n"
        f"Scientific Text Sample:\n{sample_snippet}"
    )
    
    try:
        if HAS_NEW_SDK and isinstance(client, genai.Client):
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=extraction_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ResearchPaperMetadataSchema,
                    temperature=0.1,
                ),
            )
            raw_response = response.text
        else:
            # Legacy or fallback string matching
            if client == "LEGACY_CLIENT":
                model = genai_legacy.GenerativeModel("gemini-3.5-flash")
                response = model.generate_content(
                    extraction_prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                raw_response = response.text
            else:
                raise ValueError("No initialized client")
                
        meta_dict = clean_and_parse_json(raw_response)
        
    except Exception as e:
        logger.error(f"Error during metadata extraction: {e}")
        # Build reasonable fallback metadata if clinical extraction fails
        meta_dict = {
            "title": filename.replace(".pdf", "").replace(".txt", "").replace("_", " "),
            "authors": "Unknown Clinical Authors",
            "journal": "Unspecified Medical Database",
            "year": datetime.now().year,
            "category": "General Medicine"
        }
        
    # Generate unique ID and structured response
    paper_id = f"paper_{uuid.uuid4().hex[:8]}"
    return ResearchPaperResponse(
        id=paper_id,
        title=meta_dict.get("title") or "Unnamed Research Paper",
        authors=meta_dict.get("authors") or "Unknown Authors",
        journal=meta_dict.get("journal") or "Unknown Journal",
        year=int(meta_dict.get("year") or datetime.now().year),
        category=meta_dict.get("category") or "General Medicine",
        content=content_text,
        uploadedAt=datetime.utcnow().isoformat() + "Z"
    )


class GenerateRequest(BaseModel):
    title: str
    content: str


@app.post("/generate-summary", response_model=PaperSummarySchema)
async def generate_summary(payload: GenerateRequest):
    """
    Takes the title and full string text of an academic paper.
    Asks Gemini 3.5-flash to write a high-density, structured summary with clinical endpoints,
    study methodology parameters, key Findings, and bedside Implications.
    """
    client = get_gemini_client()
    
    # Prune content slightly to fit standard context limits safely while preserving massive context
    truncated_content = payload.content[:35000]
    
    summary_prompt = (
        "You are a premier biomedical research director, journal reviewer, and bedside clinician.\n"
        "Thoroughly analyze the academic research paper text provided below and generate a dense clinical review.\n"
        "Ensure exact numbers, metrics, dosage, samples and statistical intervals (p-values, hazard ratios, etc.) are included.\n"
        "Your response MUST conform precisely to the JSON structure provided by the summary schema.\n\n"
        f"Title: {payload.title}\n"
        f"Academic Text:\n{truncated_content}"
    )
    
    try:
        if HAS_NEW_SDK and isinstance(client, genai.Client):
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=summary_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PaperSummarySchema,
                    temperature=0.2,
                )
            )
            raw_response = response.text
        else:
            model = genai_legacy.GenerativeModel("gemini-3.5-flash")
            response = model.generate_content(
                summary_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            raw_response = response.text
            
        summary_dict = clean_and_parse_json(raw_response)
        return summary_dict
        
    except Exception as e:
        logger.error(f"Error during summary generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate structured paper summary: {str(e)}"
        )


@app.post("/detect-gaps", response_model=ResearchGapAnalysisSchema)
async def detect_gaps(payload: GenerateRequest):
    """
    Analyzes literature text criticizing study bounds, recruitment errors, potential biases,
    and missing methodology gaps, compiling a comprehensive review of priority paths.
    """
    client = get_gemini_client()
    truncated_content = payload.content[:35000]
    
    gaps_prompt = (
        "You are an distinguished reviewer for clinical grant proposals and academic literature.\n"
        "Critically evaluate the research study text below to uncover hidden limitations, structural faults, "
        "reconciliation boundaries, and unanswered questions.\n"
        "Synthesize high-fidelity methodological critiques and priority directives for future lab researchers.\n"
        "Provide a priorityScore between 1 and 100 on how urgent it is to target these gaps.\n"
        "Structure your response strictly to conform with the gap analysis JSON schema models.\n\n"
        f"Title: {payload.title}\n"
        f"Literature Body:\n{truncated_content}"
    )
    
    try:
        if HAS_NEW_SDK and isinstance(client, genai.Client):
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=gaps_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ResearchGapAnalysisSchema,
                    temperature=0.3,
                )
            )
            raw_response = response.text
        else:
            model = genai_legacy.GenerativeModel("gemini-3.5-flash")
            response = model.generate_content(
                gaps_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            raw_response = response.text
            
        gaps_dict = clean_and_parse_json(raw_response)
        return gaps_dict
        
    except Exception as e:
        logger.error(f"Error during gap analysis extraction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze research gaps: {str(e)}"
        )


@app.post("/chat")
async def chat_with_lit(payload: ChatRequest):
    """
    An online conversational chatbot that reviews literature and answers complex
    biomedical queries grounded by selected studies.
    """
    if not payload.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message thread cannot be empty."
        )
        
    client = get_gemini_client()
    
    # Construct grounding context for selected papers
    papers_context_string = ""
    if payload.papersContext:
        context_parts = []
        for p in payload.papersContext:
            # Context snippet limit per paper to fit prompt window with spacious headroom
            snippet = p.content[:12000]
            context_parts.append(
                f"### COHORT PAPER: {p.id}\n"
                f"Title: \"{p.title}\"\n"
                f"Authors: {p.authors}\n"
                f"Regulatory Field: {p.category}\n"
                f"Text Transcript:\n{snippet}"
            )
        papers_context_string = "\n\n".join(context_parts)
    else:
        papers_context_string = "No research papers are currently selected by the researcher."
        
    system_instruction = (
        "You are ResearchGPT-Health, an advanced AI Biomedical Advisor for medical researchers.\n"
        "You help postdocs, medical students, and clinical directors review trial data and brainstorm models.\n"
        "Review the target medical papers listed below and provide factual, clinically objective responses "
        "grounded directly in the text.\n"
        "Always quote exact trial findings, cohort thresholds, p-values, or intervention durations if available.\n"
        "Cite research paper IDs where applicable. If the context does not hold the answer, clearly state this limit, "
        "and present a general clinical guideline backed up by standard biomedical protocols.\n"
        "Keep the formatting elegant with structured clear lists and markdown headers.\n\n"
        f"Grounded Scientific Papers in Active Library:\n{papers_context_string}"
    )
    
    # Compile conversation thread
    history_prompts = []
    for msg in payload.messages[:-1]:
        prefix = "User" if msg.sender == "user" else "Assistant"
        history_prompts.append(f"{prefix}: {msg.text}")
        
    active_query = payload.messages[-1].text
    unified_prompt = (
        "Active Researcher Query:\n"
        f"\"{active_query}\"\n\n"
        "Previous History:\n"
        + "\n".join(history_prompts)
    )
    
    try:
        if HAS_NEW_SDK and isinstance(client, genai.Client):
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=unified_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2, # low temperature to keep answers strictly aligned to paper text
                )
            )
            raw_text = response.text
        else:
            # Fallback legacy models
            model = genai_legacy.GenerativeModel(
                model_name="gemini-3.5-flash",
                system_instruction=system_instruction
            )
            response = model.generate_content(
                unified_prompt,
                generation_config={"temperature": 0.2}
            )
            raw_text = response.text
            
        return {
            "text": raw_text or "No response could be generated.",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except APIError as api_err:
        logger.error(f"Gemini API Exception: {api_err}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini API returned an failure response: {str(api_err)}"
        )
    except Exception as e:
        logger.error(f"Chat Exception: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clinical analysis chatbot error: {str(e)}"
        )


# =====================================================================
# Standalone CLI startup info
# =====================================================================
if __name__ == "__main__":
    import uvicorn
    # Print launch instructions
    print("=" * 70)
    print("ResearchGPT-Health FastAPI Service Starting...")
    print("To boot locally, execute:")
    print("  pip install fastapi uvicorn google-genai pypdf")
    print("  export GEMINI_API_KEY='your-key'")
    print("  uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)
