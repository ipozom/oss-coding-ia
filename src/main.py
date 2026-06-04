import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to sys.path to allow absolute imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import run_agent

def main():
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("OPENAI_MODEL" if provider == "openai" else "GOOGLE_MODEL" if provider == "gemini" else "OLLAMA_MODEL", "unknown")
    
    print(f"OSS Coding IA Agent initialized with {provider} ({model})")
    print("Type 'exit' to quit.")
    
    while True:
        try:
            query = input("\nUser: ")
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            run_agent(query)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
