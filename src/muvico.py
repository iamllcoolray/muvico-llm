import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys

# Configuration
MODEL_PATH = "./movie-recommender-model"  # Update this to your model path

print("="*70)
print("🎬 MOVIE RECOMMENDER - Interactive CLI")
print("="*70)

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model():
    """Load the trained model"""
    print("\n🔄 Loading model...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Error: Model not found at {MODEL_PATH}")
        print("   Please train the model first or update MODEL_PATH")
        sys.exit(1)
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        model.eval()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ Model loaded on {device}")
        
        if device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        return model, tokenizer
    
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        sys.exit(1)

# ============================================================================
# GENERATE RESPONSE
# ============================================================================

def generate_recommendation(model, tokenizer, user_input, temperature=0.7, max_tokens=300):
    """Generate movie recommendations"""
    
    # Format prompt
    prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    # Clean up
    response = response.replace("<|endoftext|>", "").strip()
    
    return response

# ============================================================================
# CLI INTERFACE
# ============================================================================

def print_help():
    """Print help message"""
    print("\n📚 Commands:")
    print("   Type your movie preferences to get recommendations")
    print("   /help    - Show this help message")
    print("   /clear   - Clear screen")
    print("   /temp    - Change temperature (creativity)")
    print("   /tokens  - Change max response length")
    print("   /example - Show example queries")
    print("   /quit    - Exit the program")
    print()

def print_examples():
    """Print example queries"""
    print("\n💡 Example Queries:")
    print("   • I loved The Matrix and Inception. What should I watch?")
    print("   • Recommend movies like The Shawshank Redemption")
    print("   • I enjoy romantic comedies. Any suggestions?")
    print("   • What are some good sci-fi thrillers?")
    print("   • I'm in the mood for something like Pulp Fiction")
    print("   • Suggest family-friendly animated movies")
    print()

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def chat_loop(model, tokenizer):
    """Main chat loop"""
    
    temperature = 0.7
    max_tokens = 300
    
    print("\n" + "="*70)
    print("💬 Chat with the Movie Recommender!")
    print("="*70)
    print_help()
    
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_input = input("🎬 You: ").strip()
            
            # Handle empty input
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower()
                
                if command == '/quit' or command == '/exit' or command == '/q':
                    print("\n👋 Thanks for using Movie Recommender! Goodbye!")
                    break
                
                elif command == '/help' or command == '/h':
                    print_help()
                    continue
                
                elif command == '/clear' or command == '/cls':
                    clear_screen()
                    print("="*70)
                    print("🎬 MOVIE RECOMMENDER - Interactive CLI")
                    print("="*70)
                    continue
                
                elif command == '/example' or command == '/examples':
                    print_examples()
                    continue
                
                elif command.startswith('/temp'):
                    try:
                        parts = command.split()
                        if len(parts) > 1:
                            temp = float(parts[1])
                            if 0.1 <= temp <= 2.0:
                                temperature = temp
                                print(f"✅ Temperature set to {temperature}")
                            else:
                                print("❌ Temperature must be between 0.1 and 2.0")
                        else:
                            print(f"Current temperature: {temperature}")
                            print("Usage: /temp <value>  (e.g., /temp 0.8)")
                    except ValueError:
                        print("❌ Invalid temperature value")
                    continue
                
                elif command.startswith('/tokens'):
                    try:
                        parts = command.split()
                        if len(parts) > 1:
                            tokens = int(parts[1])
                            if 50 <= tokens <= 1000:
                                max_tokens = tokens
                                print(f"✅ Max tokens set to {max_tokens}")
                            else:
                                print("❌ Tokens must be between 50 and 1000")
                        else:
                            print(f"Current max tokens: {max_tokens}")
                            print("Usage: /tokens <value>  (e.g., /tokens 400)")
                    except ValueError:
                        print("❌ Invalid token value")
                    continue
                
                else:
                    print(f"❌ Unknown command: {command}")
                    print("   Type /help for available commands")
                    continue
            
            # Generate response
            print("\n🤖 Assistant: ", end="", flush=True)
            
            try:
                response = generate_recommendation(
                    model, 
                    tokenizer, 
                    user_input,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                print(response)
                print()
                
                # Save to history
                conversation_history.append({
                    'user': user_input,
                    'assistant': response
                })
                
            except Exception as e:
                print(f"\n❌ Error generating response: {e}")
                print("Please try again with a different query.\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Type /quit to exit or continue chatting.")
            continue
        
        except EOFError:
            print("\n\n👋 Goodbye!")
            break

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function"""
    
    # Load model
    model, tokenizer = load_model()
    
    # Start chat
    chat_loop(model, tokenizer)

if __name__ == "__main__":
    main()