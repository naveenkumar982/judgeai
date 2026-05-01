import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application lifespan — initialise DB on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from database import init_db
        init_db()
    except Exception as e:
        logger.error(f"Database init failed: {e}")
    yield


app = FastAPI(
    title="JudgeAI",
    description="Extract, verify, and track Karnataka High Court orders using Gemini AI.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/api/health")
async def health():
    return {"status": "ok", "message": "JudgeAI API is running"}


# CORS — allow the Vite frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# ENDPOINT 1 — POST /extract
# ---------------------------------------------------------------------------

@app.post("/api/extract", summary="Extract data from a court order PDF")
async def extract(file: UploadFile = File(...)):
    """Upload a court order PDF → get AI-extracted structured data."""
    from extractor import extract_text_from_pdf, extract_with_gemini
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()

    # Step 1: pull raw text from PDF
    try:
        text = extract_text_from_pdf(pdf_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"PDF Text Extraction Failed: {str(exc)} (Type: {type(exc).__name__})"
        )

    # Step 2: send to Gemini for structured extraction
    try:
        result = extract_with_gemini(text)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini Extraction Failed: {str(exc)} (Type: {type(exc).__name__})"
        )

    return result


# ---------------------------------------------------------------------------
# ENDPOINT 2 — POST /approve
# ---------------------------------------------------------------------------

@app.post("/api/approve", summary="Approve extracted data")
async def approve(req_dict: dict):
    """Human reviewer approves (and optionally edits) extracted data."""
    from database import insert_action_plan, insert_audit_log
    
    # We use dict instead of pydantic model for extreme safety in case models.py fails to load
    # but let's try to load the model if possible
    try:
        from models import ApproveRequest
        req = ApproveRequest(**req_dict)
    except Exception as e:
        logger.error(f"Pydantic validation failed: {e}")
        # fallback or raise
        raise HTTPException(status_code=422, detail=f"Validation Error: {e}")

    # Save the approved record
    record_id = insert_action_plan(
        case_number=req.case_number,
        date=req.date,
        petitioner=req.petitioner,
        respondent=req.respondent,
        directions=req.directions,
        deadline=req.deadline,
        department=req.department,
        reviewer_name=req.reviewer_name,
    )

    # Build audit trail — compare AI originals vs human-approved values
    human_values = {
        "case_number": req.case_number,
        "date": req.date,
        "petitioner": req.petitioner,
        "respondent": req.respondent,
        "directions": req.directions,
        "deadline": req.deadline,
        "department": req.department,
    }

    if req.ai_original:
        ai_values = {
            "case_number": req.ai_original.case_number,
            "date": req.ai_original.date,
            "petitioner": req.ai_original.petitioner,
            "respondent": req.ai_original.respondent,
            "directions": req.ai_original.directions,
            "deadline": req.ai_original.deadline,
            "department": req.ai_original.department,
        }
    else:
        # If no AI original provided, treat the human values as both
        ai_values = human_values.copy()

    insert_audit_log(
        case_number=req.case_number,
        reviewer_name=req.reviewer_name,
        ai_values=ai_values,
        human_values=human_values,
    )

    return ApproveResponse(
        message="Record approved and saved successfully.",
        record_id=record_id,
    )


# ---------------------------------------------------------------------------
# ENDPOINT 3 — GET /dashboard
# ---------------------------------------------------------------------------

@app.get("/api/dashboard", summary="Get all approved action plans")
async def dashboard():
    """Return all action plans sorted by deadline (soonest first)."""
    from database import get_all_action_plans
    return get_all_action_plans()


# ---------------------------------------------------------------------------
# ENDPOINT 4 — GET /audit/{case_number}
# ---------------------------------------------------------------------------

@app.get("/api/audit/{case_number:path}", summary="Get audit log for a case")
async def audit(case_number: str):
    """Return the AI-vs-human audit trail for a specific case."""
    from database import get_audit_log
    from models import AuditEntry, AuditLogResponse
    rows = get_audit_log(case_number)
    if not rows:
        raise HTTPException(status_code=404, detail="No audit log found for this case.")

    entries = [
        AuditEntry(
            field_name=row["field_name"],
            ai_value=row["ai_value"],
            human_value=row["human_value"],
            changed=bool(row["changed"]),
        )
        for row in rows
    ]

    return AuditLogResponse(
        case_number=case_number,
        reviewer_name=rows[0]["reviewer_name"],
        approved_at=rows[0]["approved_at"],
        entries=entries,
    )
