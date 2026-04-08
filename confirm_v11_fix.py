
import sys, os
from pathlib import Path
import json

# Force v11 context
PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-key-here")

print("--- STARTING SCHOLARBOT v11 TRACE ---")

try:
    from aligned_backend import AlignedScholarBotEngine
    from router import route_query
    
    # 1. Test Router directly
    query = "What is TB, how is it diagnosed?"
    print(f"\n[TRACE 1] Testing Router for: '{query}'")
    decision = route_query(query)
    print(f"Router Decision: Intent={decision.intent}, Targets={decision.target_kbs}, Reason='{decision.reason}'")
    
    if decision.intent == "abstain":
        print("FAIL: Router is blocking this query.")
        sys.exit(1)

    # 2. Test Engine Initialization
    print("\n[TRACE 2] Initializing v11 Engine (1024-dim)...")
    engine = AlignedScholarBotEngine(verbose=True, print_kb_stats=True)
    
    # 3. Perform Retrieval Trace
    print(f"\n[TRACE 3] Performing RAG Retrieval for: '{query}'")
    # We call generate_response but let's see why it's failing
    response, confidence, meta = engine.generate_response(query)
    
    print(f"\n[RESULT] Status: {meta.get('status')}")
    print(f"[RESULT] Confidence: {confidence}")
    print(f"[RESULT] KB Source: {meta.get('source')}")
    print(f"[RESULT] References Count: {len(meta.get('references', []))}")
    print("\f\n[FINAL ANSWER PREVIEW]:")
    print(response[:800])
    
    if not meta.get("references"):
        print("\nFAIL: Retrieval failed to return references.")
        if "evidence_chunks" in meta:
            print(f"DEBUG: Evidence chunks count: {len(meta['evidence_chunks'])}")
            if meta['evidence_chunks']:
                print(f"DEBUG: First chunk text: {meta['evidence_chunks'][0].get('text', '')[:200]}")
    
except Exception as e:
    import traceback
    print(f"\n[ERROR] CRITICAL CRASH: {e}")
    traceback.print_exc()
