import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🎬 MOVIE RECOMMENDER LLM - FINE-TUNING WITH LoRA")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model settings
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Change if you have more VRAM
OUTPUT_DIR = "./movie-recommender-model"
DATA_DIR = "./data"

# Training settings
NUM_TRAIN_SAMPLES = 5000  # Number of training examples to create
NUM_EPOCHS = 3
BATCH_SIZE = 4  # Adjust based on GPU (2 for 8GB, 4 for 12GB+)
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check GPU
if torch.cuda.is_available():
    print(f"\n✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("\n⚠️  No GPU detected - training will be very slow on CPU")

# ============================================================================
# STEP 1: DOWNLOAD MOVIELENS DATASET
# ============================================================================

def download_movielens():
    """Download MovieLens 1M dataset"""
    import urllib.request
    import zipfile
    
    ratings_file = os.path.join(DATA_DIR, "ratings.dat")
    movies_file = os.path.join(DATA_DIR, "movies.dat")
    
    if os.path.exists(ratings_file) and os.path.exists(movies_file):
        print("\n✅ Dataset already downloaded")
        return
    
    if not os.path.exists(os.path.join(DATA_DIR, "ml-1m")):
        print("\n📥 Downloading MovieLens 1M dataset...")
        url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        zip_path = os.path.join(DATA_DIR, "ml-1m.zip")
    
        urllib.request.urlretrieve(url, zip_path)
        
        print("📦 Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
    else:
        print("\n✅ Dataset already downloaded")
    
    # Move files
    import shutil
    for file in ['ratings.dat', 'movies.dat']:
        src = os.path.join(DATA_DIR, 'ml-1m', file)
        dst = os.path.join(DATA_DIR, file)
        if os.path.exists(src):
            shutil.move(src, dst)
    
    # Cleanup
    os.remove(zip_path)
    shutil.rmtree(os.path.join(DATA_DIR, 'ml-1m'))
    print("✅ Dataset ready!")

# ============================================================================
# STEP 2: LOAD DATA
# ============================================================================

def load_data():
    """Load MovieLens ratings and movies"""
    print("\n📊 Loading data...")
    
    ratings = pd.read_csv(
        os.path.join(DATA_DIR, 'ratings.dat'),
        sep='::',
        engine='python',
        names=['userId', 'movieId', 'rating', 'timestamp'],
        encoding='latin-1'
    )
    
    movies = pd.read_csv(
        os.path.join(DATA_DIR, 'movies.dat'),
        sep='::',
        engine='python',
        names=['movieId', 'title', 'genres'],
        encoding='latin-1'
    )
    
    print(f"   Ratings: {len(ratings):,}")
    print(f"   Movies: {len(movies):,}")
    print(f"   Users: {ratings['userId'].nunique():,}")
    
    return ratings, movies

# ============================================================================
# STEP 3: CREATE TRAINING EXAMPLES
# ============================================================================

def create_training_examples(ratings_df, movies_df, num_samples=5000):
    """Create conversational training examples"""
    print(f"\n🔨 Creating {num_samples} training examples...")
    
    training_data = []
    
    # Get active users
    user_counts = ratings_df['userId'].value_counts()
    active_users = user_counts[user_counts >= 20].index.tolist()
    
    # Sample users
    sampled_users = np.random.choice(
        active_users,
        min(num_samples, len(active_users)),
        replace=False
    )
    
    # Prompt templates
    templates = [
        "I loved {movies}. What should I watch next?",
        "I really enjoyed {movies}. Recommend similar movies.",
        "Based on {movies}, what would you suggest?",
        "I'm looking for movies like {movies}.",
        "My favorites are {movies}. What else would I like?",
    ]
    
    for user_id in sampled_users:
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        
        # Get highly rated movies
        liked = user_ratings[user_ratings['rating'] >= 4.0]
        loved = user_ratings[user_ratings['rating'] >= 4.5]
        
        if len(liked) >= 3 and len(loved) >= 2:
            # Get movie details
            liked_movies = movies_df[movies_df['movieId'].isin(liked['movieId'])]
            loved_movies = movies_df[movies_df['movieId'].isin(loved['movieId'])]
            
            # Sample for context
            context_movies = liked_movies.sample(min(np.random.randint(3, 6), len(liked_movies)))
            rec_movies = loved_movies.sample(min(np.random.randint(2, 5), len(loved_movies)))
            
            # Format context
            context_titles = context_movies['title'].tolist()
            if len(context_titles) > 1:
                context_str = ", ".join(context_titles[:-1]) + f" and {context_titles[-1]}"
            else:
                context_str = context_titles[0]
            
            # Create prompt
            prompt = np.random.choice(templates).format(movies=context_str)
            
            # Create response
            response = "Based on your preferences, here are my recommendations:\n\n"
            
            for idx, (_, movie) in enumerate(rec_movies.iterrows(), 1):
                title = movie['title']
                genres = movie['genres'].replace('|', ', ')
                
                # Find common genres
                genre_list = set(movie['genres'].split('|'))
                context_genres = set('|'.join(context_movies['genres']).split('|'))
                common_genres = genre_list & context_genres
                
                if common_genres:
                    reason = f"shares your love of {', '.join(list(common_genres)[:2])}"
                else:
                    reason = f"is a great {genres} film"
                
                response += f"{idx}. **{title}** - This {reason} and matches your taste.\n\n"
            
            # Format for training
            training_text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}<|endoftext|>"
            
            training_data.append({'text': training_text})
    
    print(f"✅ Created {len(training_data)} training examples")
    
    # Save sample
    sample_file = os.path.join(DATA_DIR, 'training_samples.json')
    with open(sample_file, 'w') as f:
        json.dump(training_data[:5], f, indent=2)
    print(f"💾 Saved 5 sample examples to: {sample_file}")
    
    return training_data

# ============================================================================
# STEP 4: SETUP MODEL
# ============================================================================

def setup_model():
    """Load and prepare model with LoRA"""
    print(f"\n🤖 Loading model: {MODEL_NAME}")
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=True,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    
    print("✅ Model loaded")
    
    return model, tokenizer

def configure_lora(model):
    """Configure LoRA parameters"""
    print("\n⚙️  Configuring LoRA...")
    
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print stats
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"   Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"   Total: {total:,}")
    
    return model

# ============================================================================
# STEP 5: TRAIN
# ============================================================================

def train_model(model, tokenizer, train_dataset, val_dataset):
    """Train the model"""
    print("\n🚀 Starting training...")
    
    from trl import SFTConfig, SFTTrainer
    
    # SFTConfig - minimal version
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=250,
        eval_steps=250,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        bf16=False,
        optim="paged_adamw_8bit",
        report_to="none",
    )
    
    # Minimal SFTTrainer - just the essentials
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Training info
    total_steps = len(train_dataset) * NUM_EPOCHS // (BATCH_SIZE * 4)
    print(f"\n📊 Training Configuration:")
    print(f"   Examples: {len(train_dataset)}")
    print(f"   Validation: {len(val_dataset)}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Steps: {total_steps}")
    print(f"   Estimated time: ~{total_steps * 2 / 60:.0f} minutes")
    
    # Train
    print("\n⏱️  Training started...\n")
    start = datetime.now()
    
    trainer.train()
    
    duration = (datetime.now() - start).total_seconds() / 60
    print(f"\n✅ Training complete in {duration:.1f} minutes!")
    
    return trainer

# ============================================================================
# STEP 6: SAVE & TEST
# ============================================================================

def save_model(trainer, tokenizer):
    """Save the trained model"""
    print("\n💾 Saving model...")
    
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save info
    info = {
        'model_name': MODEL_NAME,
        'trained_date': datetime.now().isoformat(),
        'epochs': NUM_EPOCHS,
        'samples': NUM_TRAIN_SAMPLES,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
    }
    
    with open(os.path.join(OUTPUT_DIR, 'training_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Saved to: {OUTPUT_DIR}")

def test_model(model, tokenizer):
    """Test the trained model"""
    print("\n🧪 Testing model...\n")
    
    model.eval()
    
    test_queries = [
        "I loved The Matrix, Inception, and Interstellar. What should I watch?",
        "Recommend movies like The Shawshank Redemption and The Godfather.",
        "I enjoy romantic comedies. Any suggestions?",
    ]
    
    for query in test_queries:
        print("="*70)
        print(f"Query: {query}")
        print("="*70)
        
        prompt = f"<|user|>\n{query}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("<|assistant|>")[-1].strip()
        
        print(f"\n{response}\n")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run complete training pipeline"""
    
    # Download data
    download_movielens()
    
    # Load data
    ratings, movies = load_data()
    
    # Create training examples
    train_data = create_training_examples(ratings, movies, NUM_TRAIN_SAMPLES)
    
    # Split train/val
    train_examples, val_examples = train_test_split(train_data, test_size=0.1, random_state=42)
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)
    
    # Setup model
    model, tokenizer = setup_model()
    model = configure_lora(model)
    
    # Train
    trainer = train_model(model, tokenizer, train_dataset, val_dataset)
    
    # Save
    save_model(trainer, tokenizer)
    
    # Test
    test_model(model, tokenizer)
    
    print("\n" + "="*70)
    print("✅ ALL DONE!")
    print("="*70)
    print(f"\nYour trained model is in: {OUTPUT_DIR}")
    print("\nTo use it later:")
    print("  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{OUTPUT_DIR}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_DIR}')")
    print()

if __name__ == "__main__":
    main()