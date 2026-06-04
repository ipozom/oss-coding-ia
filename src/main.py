import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
required_env_vars = ["LLM_PROVIDER", "OPENAI_MODEL", "GOOGLE_MODEL", "OLLAMA_MODEL"]
missing_env_vars = [var for var in required_env_vars if os.getenv(var) is None]
if missing_env_vars:
    print(f"❌ Missing environment variables: {', '.join(missing_env_vars)}")
    sys.exit(1)
# Add the project root to sys.path to allow absolute imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import Agent

def main():
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv(
        "OPENAI_MODEL" if provider == "openai" else 
        "GOOGLE_MODEL" if provider == "gemini" else 
        "OLLAMA_MODEL", 
        "unknown"
    )
    
    print("=" * 50)
    print("🤖 OSS Coding IA Agent")
    print("=" * 50)
    print(f"📡 Provider: {provider}")
    print(f"🧠 Model: {model}")
    print("=" * 50)
    print("Type 'exit', 'quit', or 'q' to quit.")
    print()
    
    while True:
        try:
            query = input("💬 User: ")
            if query.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            if query.strip():
                print("\n" + "-" * 50)
                agent = Agent()
                try:
                    response = agent.run(query)
                    print(response)
                except Exception as e:
                    print(f"\n❌ Error processing your request: {e}\n")
                print("-" * 50 + "\n")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()