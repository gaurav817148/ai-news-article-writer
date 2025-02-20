import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
@st.cache_resource  # Cache the model to avoid reloading every time
def load_model():
    try:
        model_name = "gpt2"  # You can also use "gpt2-medium" or "gpt2-large"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def generate_article(headline, max_length=200, temperature=0.7, top_k=40, top_p=0.9, no_repeat_ngram_size=3):
    tokenizer, model = load_model()
    if tokenizer is None or model is None:
        return "Model failed to load. Please check your internet connection or try again later."

    try:
        # Add a system prompt to guide the model
        prompt = f"Write a detailed and engaging news article about: {headline}\n\n"
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=no_repeat_ngram_size,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            do_sample=True,
        )
        article = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return article
    except Exception as e:
        return f"Error generating article: {e}"
    

# Streamlit app
st.title("AI News Article Writer")
st.write("Enter a headline below, and the AI will generate a news article for you!")

# Input headline
headline = st.text_input("Enter a headline:")

# Add sliders for user control
st.sidebar.header("Adjust Parameters")
max_length = st.sidebar.slider("Select article length (in words):", 50, 500, 200)
temperature = st.sidebar.slider("Select creativity (temperature):", 0.1, 1.0, 0.8)
top_k = st.sidebar.slider("Select top-k sampling:", 1, 100, 50)
top_p = st.sidebar.slider("Select top-p (nucleus sampling):", 0.1, 1.0, 0.95)
no_repeat_ngram_size = st.sidebar.slider("Select no-repeat n-gram size:", 1, 5, 2)

# Generate article button
if st.button("Generate Article"):
    if headline:
        with st.spinner("Generating article..."):
            article = generate_article(
                headline,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            st.write("### Generated Article:")
            st.write(article)
    else:
        st.warning("Please enter a headline.")