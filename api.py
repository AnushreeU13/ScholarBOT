from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
from aligned_backend import AlignedScholarBotEngine
import os

app = FastAPI(title="ScholarBOT Clinical RAG API", 
              description="A Fail-Closed RAG system for Clinical Guidelines and Drug Labels.")

# Initialize the backend engine
engine = AlignedScholarBotEngine()

class QueryRequest(BaseModel):
    query: str
    user_uploaded_available: Optional[bool] = False

class QueryResponse(BaseModel):
    clinician_answer: str
    patient_answer: str
    citations: List[str]
    confidence: float
    status: str
    source_kbs: List[str]

@app.post("/api/query", response_model=QueryResponse)
def query_scholarbot(request: QueryRequest):
    try:
        ans, conf, metadata = engine.generate_response(
            query=request.query, 
            user_uploaded_available=request.user_uploaded_available
        )
        
        # Check if it abstained
        if ans.upper().strip() == "ABSTAIN":
            return QueryResponse(
                clinician_answer="ABSTAIN",
                patient_answer="ABSTAIN",
                citations=[],
                confidence=conf,
                status="abstained - insufficient evidence",
                source_kbs=metadata.get("source_kbs", [])
            )
            
        # Returning full RAG result fields 
        return QueryResponse(
            clinician_answer=metadata.get('clinician_answer', ans),
            patient_answer=metadata.get('patient_answer', ''),
            citations=metadata.get('citations', []),
            confidence=conf,
            status="success",
            source_kbs=metadata.get('source_kbs', [])
        )
    except Exception as e:
        logging.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
