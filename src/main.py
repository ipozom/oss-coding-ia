import sys
import os

# Add the project root to sys.path to allow absolute imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import run_agent

def main():
    print("OSS Coding IA Agent initialized with Ollama (qwen2.5-coder:14b)")
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
