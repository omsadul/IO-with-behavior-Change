from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the model and tokenizer
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embeddings(text_list):
    """
    Generate embeddings for a list of text items using RoBERTa.
    """
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def load_data(fndds_data_path):
    xls = pd.ExcelFile(fndds_data_path)
    fndds_data = pd.read_excel(xls, sheet_name='Food and Beverages', header=1)  # Ignore the first row and use the second row as header
    categories_data = pd.read_excel(xls, sheet_name='WWEIA Categories', header=None)
    categories_data.columns = ['WWEIA Category']
    return fndds_data, categories_data

def categorize_foods(food_descriptions, categories):
    food_embeddings = get_embeddings(food_descriptions)
    category_embeddings = get_embeddings(categories)
    similarities = cosine_similarity(food_embeddings, category_embeddings)
    category_indices = similarities.argmax(axis=1)
    categorized_foods = [categories[idx] for idx in category_indices]
    return categorized_foods

def main():
    fndds_data_path = r'E:\Prof Kimia Ghobadi\IO-with-behavior-Change\Smart categorization Models\data\2021-2023 FNDDS At A Glance - Foods and Beverages.xlsx'
    fndds_data, categories_data = load_data(fndds_data_path)
    # Print the columns of fndds_data to verify
    print("Columns in fndds_data:")
    print(fndds_data.columns)
    food_descriptions = fndds_data['Main food description'].dropna().astype(str).tolist()
    categories = categories_data['WWEIA Category'].dropna().astype(str).tolist()
    categorized_foods = categorize_foods(food_descriptions, categories)
    # Print the first 10 categorized foods for verification
    for food, category in zip(food_descriptions[:10], categorized_foods[:10]):
        print(f"Food: {food} -> Category: {category}")

if __name__ == "__main__":
    main()