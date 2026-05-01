"""Firestore database layer for JudgeAI — action_plans + audit_log collections."""

import json
from datetime import datetime, timezone
import os

from google.cloud import firestore
from google.oauth2 import service_account

_db = None

def get_db():
    global _db
    if _db is not None:
        return _db
        
    # Initialize Firestore client
    key_path = os.path.join(os.path.dirname(__file__), "service-account.json")
    if os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON"):
        # Load from environment variable string (useful for Vercel)
        creds_dict = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"])
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        _db = firestore.Client(project=creds_dict["project_id"], credentials=credentials, database="(default)")
    elif os.path.exists(key_path):
        # Load from local file
        credentials = service_account.Credentials.from_service_account_file(key_path)
        _db = firestore.Client(project="judgeai-db-12345", credentials=credentials, database="(default)")
    else:
        # Fallback to default credentials
        _db = firestore.Client(project="judgeai-db-12345", database="(default)")
    return _db


def init_db():
    """No-op for Firestore, collections are created implicitly."""
    get_db()
    pass


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def insert_action_plan(
    case_number: str,
    date: str,
    petitioner: str,
    respondent: str,
    directions: list[str],
    deadline: str,
    department: str,
    reviewer_name: str,
) -> str:
    """Insert a human-approved action plan. Returns the new document ID."""
    now = datetime.now(timezone.utc).isoformat()
    
    doc_ref = get_db().collection("action_plans").document()
    doc_ref.set({
        "case_number": case_number,
        "date": date,
        "petitioner": petitioner,
        "respondent": respondent,
        "directions": directions,
        "deadline": deadline,
        "department": department,
        "reviewer_name": reviewer_name,
        "approved_at": now
    })
    
    return doc_ref.id


def insert_audit_log(
    case_number: str,
    reviewer_name: str,
    ai_values: dict,
    human_values: dict,
):
    """Record per-field AI vs human comparison for audit."""
    now = datetime.now(timezone.utc).isoformat()
    fields = ["case_number", "date", "petitioner", "respondent",
              "directions", "deadline", "department"]
              
    batch = get_db().batch()
    
    for field in fields:
        ai_val = ai_values.get(field, "")
        human_val = human_values.get(field, "")
        
        # Normalise lists to JSON strings for comparison
        if isinstance(ai_val, list):
            ai_val = json.dumps(ai_val)
        if isinstance(human_val, list):
            human_val = json.dumps(human_val)
            
        changed = 1 if str(ai_val) != str(human_val) else 0
        
        doc_ref = get_db().collection("audit_log").document()
        batch.set(doc_ref, {
            "case_number": case_number,
            "field_name": field,
            "ai_value": str(ai_val),
            "human_value": str(human_val),
            "changed": changed,
            "reviewer_name": reviewer_name,
            "approved_at": now
        })
        
    batch.commit()


def get_all_action_plans() -> list[dict]:
    """Return all action plans sorted by deadline ascending."""
    docs = get_db().collection("action_plans").order_by(
        "deadline", direction=firestore.Query.ASCENDING
    ).stream()
    
    results = []
    for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        results.append(d)
        
    return results


def get_audit_log(case_number: str) -> list[dict]:
    """Return audit entries for a specific case number."""
    docs = get_db().collection("audit_log").where(
        "case_number", "==", case_number
    ).order_by("approved_at", direction=firestore.Query.ASCENDING).stream()
    
    results = []
    for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        results.append(d)
        
    return results
