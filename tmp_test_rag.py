
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from aligned_backend import AlignedScholarBotEngine

def test_v9():
    print("Testing ScholarBOT v9 Engine...")
    try:
        engine = AlignedScholarBotEngine(verbose=True)
        query = "What is Tuberculosis?"
        print(f"Query: {query}")
        response, confidence, metadata = engine.generate_response(query)
        print("\n--- RESPONSE ---")
        print(response)
        print(f"\nConfidence: {confidence}")
        print(f"Status: {metadata.get('status')}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_v9()
