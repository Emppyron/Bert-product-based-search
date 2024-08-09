import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import ast
from sklearn.metrics.pairwise import cosine_similarity

# Load BERT model and tokenizer
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load your dataset (Ensure this points to your preprocessed DataFrame)
df = pd.read_csv('flipkart_com-ecommerce_sample.csv')

## data cleaning

df=df.dropna(subset=df.columns)
df=df.head(20)

# Function to parse product specifications
def parse_specifications(spec_string):
    spec_string = spec_string.replace("=>", ":")
    try:
        spec_dict = ast.literal_eval(spec_string)
        spec_list = spec_dict.get('product_specification', [])
        spec_str = " ".join([f"{item['key']}: {item['value']}" for item in spec_list if 'key' in item and 'value' in item])
    except (ValueError, SyntaxError):
        spec_str = ""
    return spec_str

print('h')
# Preprocess the dataset (you can skip this if already preprocessed)
df['parsed_specifications'] = df['product_specifications'].apply(parse_specifications)
df['combined_text'] = (df['product_name'] + " " + 
                       df['parsed_specifications'] + " " + df['product_category_tree'])

# Function to generate embeddings
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

# Generate embeddings (if not precomputed)
df['embedding'] = df['combined_text'].apply(lambda x: get_embedding(x, tokenizer, model))

# Function to search for products
def search_products(query, df, tokenizer, model, top_k=5):
    query_embedding = get_embedding(query, tokenizer, model)
    similarities = []
    for idx, product_embedding in enumerate(df['embedding']):
        similarity = cosine_similarity([query_embedding], [product_embedding]).item()
        similarities.append((df.iloc[idx], similarity))
    sorted_products = sorted(similarities, key=lambda x: x[1], reverse=True)
    top_products = sorted_products[:top_k]
    return top_products

# Streamlit UI
st.title("BERT-based Product Search")

# Query input
query = st.text_input("Enter your search query:")

if query:
    # Perform the search
    top_products = search_products(query, df, tokenizer, model)
    
    # Display the results
    st.subheader("Top Products")
    for product, similarity in top_products:
        image_urls = product['image']
        # Parse image_urls if it's a string representation of a list
        if isinstance(image_urls, str):
            try:
                image_urls = ast.literal_eval(image_urls)
            except (ValueError, SyntaxError):
                image_urls = [image_urls]

        st.write(f"**{product['product_name']}**")
        for url in image_urls:
            st.image(url, width=200)

        st.write(f"**Description**: {product['description']}")
        st.write(f"**Retail Price**: ${product['retail_price']}")
        st.write(f"**Discounted Price**: ${product['discounted_price']}")
        st.write(f"**Similarity Score**: {similarity:.4f}")
        st.write("---")  # Divider between products

# Run the app with streamlit run app.py
