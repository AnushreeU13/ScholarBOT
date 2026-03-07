
import os
import sys
import json

# Setup path to import backend
sys.path.append(os.getcwd())
from aligned_backend import AlignedScholarBotEngine

def main():
    print("Initializing ScholarBOT v9 Engine...")
    # This will build BM25 indices
    engine = AlignedScholarBotEngine(verbose=True)
    
    query = "How is tuberculosis diagnosed according to WHO guidelines?"
    print(f"\n--- Testing Query: {query} ---")
    
    # generate_response
    response, conf, meta = engine.generate_response(query)
    
    print("\n--- RESPONSE ---")
    print(response)
    print(f"\nConfidence: {conf}")
    print("\n--- DEBUG INFO ---")
    # See if Hybrid search was used
    evidence = meta.get("evidence_chunks", [])
    print(f"Retrieved {len(evidence)} chunks.")
    
    # Check if Clinician Summary contains citations
    print("\nMetadata Trace:")
    print(f"Status: {meta.get('status')}")
    print(f"Source: {meta.get('source')}")

if __name__ == "__main__":
    main()
