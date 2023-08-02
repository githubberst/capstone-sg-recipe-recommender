# A very simple Flask Hello World app for you to get started with...

import datetime
import os
from flask import Flask, render_template, request
from pathlib import Path
from werkzeug.utils import secure_filename
import urllib.parse
THIS_FOLDER = Path(__file__).parent.resolve()
app = Flask(__name__)

### Sandra ipynb code
import pandas as pd
import re
import numpy as np
from ast import literal_eval
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import warnings
import nltk
from nltk.stem import WordNetLemmatizer
THIS_FOLDER = Path(__file__).parent.resolve()


nltk.download('wordnet')
nltk.download('omw-1.4')
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

warnings.filterwarnings('ignore')


def extract_ingredients(ingredients_list):
    # The regex pattern to split at numbers, fractions, and measurement units
    pattern = re.compile(r'\d+-\d+|\d+|\d+\/\d+|\bcup\b|\bkg\b|\bcube\b|\bcubes\b|\bof\b|\bknob\b|\btbs\b|\bfrom\b|\bwith\b|\btbsp\b|\btsp\b|\bml\b|\bg\b|\boz\b|\bteaspoon\b|\btablespoon\b|\blitres\b|\blitre\b|\bcups\b|\bpack\b|\bpacks\b|\bpinch\b|\bcan\b|\bslice\b|\bstalk\b|\bpiece\b|½|¼|⅛|¾|⅓', re.IGNORECASE)

    # Split the strings at the pattern, and remove leading/trailing whitespace from each split string
    ex_ingredients = []
    for ingredient in ingredients_list:
        ingredient = re.sub(pattern, '', ingredient).strip()
        ingredient = ingredient.lower()
        ingredient = re.sub(pattern, '', ingredient).strip()
        ex_ingredients.append(ingredient)
    
    return ex_ingredients

prep_diff_df = pd.read_csv(THIS_FOLDER / 'data/prep_diff.csv')
meatmen_df = pd.read_csv(THIS_FOLDER / 'data/meatmen_scraped_raw.csv')
meatmen_df = pd.merge(prep_diff_df, meatmen_df, on=['url','image'])
# check for missing values 
meatmen_df.difficulty.unique()
difficulty_mapping = {'super easy': 1, 'easy': 2, 'medium': 3, 'hard': 4}
meatmen_df['difficulty_scale'] = meatmen_df['difficulty'].map(difficulty_mapping)
empty_ingredients = meatmen_df[meatmen_df['ingredients'] == '[]']
empty_directions = meatmen_df[meatmen_df['directions'] == '[]']
meatmen_df = meatmen_df[meatmen_df['ingredients'] != '[]'] #filtering out recipes that contain empty lists 
meatmen_df = meatmen_df[meatmen_df['directions'] != '[]'] #filtering out recipes that contain empty directions 


ingredients_df = meatmen_df[['recipe_name','ingredients']]

ingredients_df['n_directions'] = meatmen_df['directions'].apply(literal_eval).apply(len)

# Convert the strings into lists - comment this after first run
ingredients_df['ingredients'] = ingredients_df['ingredients'].apply(literal_eval) 

# Apply the function to the 'ingredients' column
ingredients_df['ingredients_ex'] = ingredients_df['ingredients'].apply(extract_ingredients)
ingred_ex_df = ingredients_df[['recipe_name','ingredients_ex', 'n_directions']]
ingred_ex_df['n_all_ingredients'] = ingred_ex_df['ingredients_ex'].apply(len)



# Combine all of the ingredient names into a single list
all_ingredients = []
for sublist in ingred_ex_df['ingredients_ex']:
    for ingredient in sublist:
        all_ingredients.append(ingredient)
        
# Count the frequency of each ingredient
ingredient_counts = Counter(all_ingredients)

lemmatizer = WordNetLemmatizer()
# Define the replacements
replacements = {
    r'vermicelli':'bee hoon', 
    r'chicken breast':'chicken', 
    r'lean chicken':'chicken', 
    r'^dried shiitake mushroom$':'dried mushroom', 
    r'^red onion$':'onion', 
    r'^shallot$':'onion', 
    r'^small onion$':'onion', 
    r'^dark soy sauce$':'soy sauce', 
    r'^light soy sauce$':'soy sauce',
    r'^sweet soy sauce$':'soy sauce',
    r'^soya sauce$':'soy sauce',
    r'fried shallots': 'shallots',
    r'.*egg.*': 'egg',
    r'\bonion\b': 'onion',
    r'\bginger\b': 'ginger',
    r'\bgalangal?\b': 'ginger',
    r'\blemon\b': 'lemon',
    r'\bgarlic\b': 'garlic',
    r'\byogurt\b': 'yogurt',
    r'\bparsley\b': 'parsley',
    r'\bprawn\b': 'prawn', 
    r'\b(chil(?:i|ie|ies|li|lis|lies))\b': 'chilli', 
    r'\b(peppers?|peppercorn)\b': 'pepper', 
    r'\bsalt\b': 'salt', 
    r'\boil\b': 'oil', 
    r'\bvinegar\b': 'vinegar', 
    r'\bflour\b': 'flour',
    r'\bbutter\b': 'butter', 
    r'\bwater\b': 'water', 
    r'\btofu\b': 'tofu', 
    r'\bnoodles?\b': 'noodle', 
    r'chicken(?!\s(stock|broth|powder))':'chicken',
    r'pork(?!\s(stock|broth|powder))':'pork',
    r'beef(?!\s(stock|broth|powder))':'beef', 
    r'(?<!glutinous\s)rice(?!\s(stock|broth|vermicelli|beehoon|bee hoon|noodales|noodle|penne|crackers|risotto|glutinous))':'rice', 
    r'.*stock.*': 'stock',    
    r'.*peanut.*': 'peanut', 
    r'mushroom(?!\s(ball))':'mushroom', 
    r'.*lime.*': 'lime',
    r'milk(?!\s(coconut))':'milk', 
    r'carrot(?!\s(cake|juice))':'carrot', 
    r'.*sausage.*':'sausage', 
    r'(?<!sweet )potato(es)?':'potato', 
    r'barramundi|fish fillet|snapper|seabass|salmon':'fish',
    r'fish(?! (bones|powder|sauce|balls?))':'fish', 
    r'chicken broth':'chicken broth', 
    r'apple':'apple', 
    r'duck':'duck', 
    r'\btomato(es)?\b(?! paste)':'tomato', 
    r'orange':'orange', 
    r'coriander':'coriander', 
    r'(?<!sea )cucumber':'cucumber', 
    r'\bbanana(s)?\b(?! leaf|leaves)':'banana', 
    r'cabbage':'cabbage', 
    r'lamb':'lamb', 
    r'vegetables':'vegetables', 
    r'tau kwa':'tau kwa', 
    r'abalone':'abalone', 
    r'shallot(s)?':'shallot', 
    r'sweet potato(es)?':'sweet potato',
    r'\bsugars?\b': 'sugar',

}

# Define the descriptors
descriptors = ['grated', 'salt and pepper to taste', 'toasted', 'salt (to taste)', 'salt to taste', 'for deep frying',
               '(minced)', '(sliced)', 'solution', 'chopped', 'for frying', 'for deep-frying', 'for deep fry',
               '(shelled and deveined)', '(lightly beaten)', 'large', 'salt and white pepper to taste', '(bruised)',
               'pepper to taste', 'salt & pepper to taste', 'with', '(adjust to taste)', 'black pepper to taste',
               'fresh coriander for garnish', 'cracked', 'for garnish', 'fresh', 'for cooking', 'for blanching', '(adjust to preference)',
               'optional','quartered','washed', 'garnish', 'serve']

# Define the function to process ingredients
def process_ingredient(ingredient):
    
    # Lemmatization
    ingredient = lemmatizer.lemmatize(ingredient)
    
    # Standardize Ingredient Names
    ingredient = 'chicken breast' if ingredient == 'chicken breasts' else ingredient
    
    # Remove Descriptors and Units
    for descriptor in descriptors:
        ingredient = ingredient.replace(descriptor, '')
        
    # Remove units
    units = ['cloves', 'inch','stalks', 'bulb', 'sprigs', 'slices', '/']
    
    for unit in units:
        ingredient = ingredient.replace(unit, '')
    
    unit_r = [r'\bdash\b']
    
    for unit in unit_r:
        ingredient = re.sub(unit, '', ingredient)
    
    # Group Similar Ingredients
    for pattern, replacement in replacements.items():
        if re.search(pattern, ingredient, re.IGNORECASE):
            ingredient = replacement
   
    return ingredient.strip()  # remove leading and trailing whitespace

# Apply the function to the 'ingredients_ex' column
ingred_ex_df['ingredients_processed'] = ingred_ex_df['ingredients_ex'].apply(lambda x: [process_ingredient(ingredient) for ingredient in x])

# Now, we can define the condiments and herbs lists, including the renamed condiment-type ingredients
condiments = ['pepper', 'sugar', 'salt', 'oil', 'vinegar', 'flour', 'butter', 'water', 
              'oyster sauce', 'fish sauce', 'light soy sauce', 'dark soy sauce', 'palm sugar', 
              'white pepper powder', 'cooking oil', 'sesame oil', 'ketchup', 'rock sugar', 'cornstarch', 
              'rice flour', 'tapioca flour', 'hua tiao wine', 'shaoxing wine', 'plain flour', 
              'glutinous rice flour', 'baking soda', 'sake', 'mirin', 'sambal belacan', 'belacan',
              'dried chilli paste', 'white pepper', 'brown sugar', 'hoisin sauce', 'abalone sauce', 
              'white peppercorns', 'plum sauce', 'white vinegar', 'olive oil', 'unsalted butter', 
              'soy sauce', 'apple cider vinegar', 'chinese rice wine', 'black pepper', 'dijon mustard', 
              'kecap manis', 'black peppercorns', 'rice vinegar', 'black vinegar', 'cooking wine', 'rice wine', 'rice syrup', 
              'baking powder', 'vanilla extract', 'xo sauce', 'corn', 'tapioca starch', 'soya sauce', 'shortening',
              'sodium bicarbonate', 'boiling water', 'self raising flour', 'gula melaka','sauce','instant yeast','potato starch']

herbs = ['lemongrass', 'candlenut', 'turmeric powder', 'coriander powder', 'cumin powder', 
         'star anise', 'clove', 'cinnamon stick', 'spice powder', 'five spice powder', 
         'cinnamon', 'bay leaves', 'worcestershire sauce', 'thyme', 'mint leaves', 
         'kaffir lime leaves', 'coriander seeds', 'black fungus', 'huiji waist tonic']

# Function to extract key ingredients
def extract_key_ingredients(ingredients):
    key_ingredients = []
    for ingredient in ingredients:
        if not any(condiment in ingredient for condiment in condiments) and not any(herb in ingredient for herb in herbs):
            key_ingredients.append(ingredient)
    return key_ingredients

# Create col for 'key_ingredients'
ingred_ex_df['key_ingredients'] = ingred_ex_df['ingredients_processed'].apply(extract_key_ingredients)


def remove_empty_strings(ingredient_list):
    return [ingredient for ingredient in ingredient_list if ingredient]

ingred_ex_df['key_ingredients'] = ingred_ex_df['key_ingredients'].apply(remove_empty_strings)

# Adding new descriptors and units to be removed or handled
new_units = ['cm', 'bunch', 'pcs','medium','servings','gram', 'size', 'small', 'large', 'approx.','dried', '%','fresh', 'toasted', 'chopped', 'sliced', 'pieces','optional', 'preferably']

# Updating the list of descriptors and units
descriptors_units = descriptors + new_units

def process_ingredient_v2(ingredient):
    
    # Step 1: Remove Descriptors and Units
    for descriptor_unit in descriptors_units:
        ingredient = ingredient.replace(descriptor_unit, '')
    
    # Step 2: Remove content within brackets
    ingredient = re.sub(r'\([^)]*\)', '', ingredient)
    ingredient = re.sub(r'\(\)', '', ingredient)

    # Step 3: Standardize Ingredient Names
    ingredient = 'chicken breast' if ingredient == 'chicken breasts' else ingredient
    
    # Step 4: Group Similar Ingredients
    for pattern, replacement in replacements.items():
        if re.search(pattern, ingredient, re.IGNORECASE):
            ingredient = replacement
    
     
#     # Step 5: Remove Rare Ingredients
#     # We'll use the Counter class to count the ingredients and keep only those with count >= 5
#     ingredient_counts = Counter(all_ingredients)
#     ingredient = ingredient if ingredient_counts[ingredient] >= 5 else ''
    
    return ingredient.strip()  # remove leading and trailing whitespace

# Apply the function to the 'key_ingredients' column
ingred_ex_df['ingredients_processed'] = ingred_ex_df['ingredients_processed'].apply(lambda ingredients: [process_ingredient_v2(ingredient) for ingredient in ingredients])

ingred_ex_df['ingredients_processed'] = ingred_ex_df['ingredients_processed'].apply(remove_empty_strings)

# update 'key_ingredients'
ingred_ex_df['key_ingredients'] = ingred_ex_df['ingredients_processed'].apply(extract_key_ingredients)
ingred_ex_df['key_ingredients'] = ingred_ex_df['key_ingredients'].apply(remove_empty_strings)

# Flatten the 'key_ingredients' lists into a single list
all_key_ingredients = [ingredient for sublist in ingred_ex_df['key_ingredients'] for ingredient in sublist]

# double check that condiments like 'salt' and 'pepper' are not in the list. 
# Count the occurrences of 'salt' and 'pepper' in the list
salt_count = all_key_ingredients.count('salt')
pepper_count = all_key_ingredients.count('pepper')

# Get the unique key ingredients and their count
unique_key_ingredients = set(all_key_ingredients)
num_unique_key_ingredients = len(unique_key_ingredients)

unique_key_ingredients = pd.Series(list(unique_key_ingredients))  # Convert set to pandas Series

# List to store tuples of (search term, duplicates)
dupes_list = []

for i in unique_key_ingredients:
    # Escape special characters in the string
    i_escaped = re.escape(i)
    
    dupes = unique_key_ingredients[unique_key_ingredients.str.contains(i_escaped)]
    if len(dupes) > 1:
        # Add a tuple of (search term, duplicates) to the list
        dupes_list.append((i, dupes.unique().tolist()))

# Sort the list by the length of the duplicates in descending order
dupes_list.sort(key=lambda x: len(x[1]), reverse=True)


# Count the occurrences of each key ingredient
from collections import Counter
key_ingredient_counts = Counter(all_key_ingredients)



#Plot the occurrences of top 30 key ingredients after preprocessing 

import pandas as pd
import matplotlib.pyplot as plt

# Convert the counter to a DataFrame
key_ingredient_counts_df = pd.DataFrame.from_dict(key_ingredient_counts, orient='index').reset_index()
key_ingredient_counts_df = key_ingredient_counts_df.rename(columns={'index':'key_ingredient', 0:'count'})

# Sort the DataFrame by count
key_ingredient_counts_df = key_ingredient_counts_df.sort_values('count', ascending=False)


# treat multi-word ingredients as single terms
def multi2single_terms(ingredients):
    return [ingredient.replace(' ', '_') for ingredient in ingredients]

# Apply preprocessing to key ingredients data
ingred_ex_df['key_ingred_processed'] = ingred_ex_df['key_ingredients'].apply(multi2single_terms)

all_key_ingredients_pro = [ingredient for sublist in ingred_ex_df['key_ingred_processed'] for ingredient in sublist]
key_ingredient_counts_pro = Counter(all_key_ingredients_pro)
key_ingredient_counts_pro_df = pd.DataFrame.from_dict(key_ingredient_counts_pro, orient='index').reset_index()
key_ingredient_counts_pro_df = key_ingredient_counts_pro_df.rename(columns={'index':'key_ingredient', 0:'count'})
key_ingredient_counts_pro_df = key_ingredient_counts_pro_df.sort_values('count', ascending=False)

# e.g. "rice" and "glutinous_rice" will be treated as distinct ingredients

top_30_ingredients = key_ingredient_counts_pro_df['key_ingredient'][:30]

flask_disp_ingredients = {}

for index,item in top_30_ingredients.items():
     flask_disp_ingredients[item] = key_ingredient_counts_df['key_ingredient'][index]

flask_disp_ingredients["peanut"] = "peanut"

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a list of all the ingredients
all_key_ingredients = [' '.join(ingredients) for ingredients in ingred_ex_df['key_ingred_processed']]

# Initialize the TfidfVectorizer with the top 30 ingredients as the vocabulary
vectorizer = TfidfVectorizer(vocabulary=top_30_ingredients)

# Fit and transform the vectorizer on our corpus
tfidf_matrix = vectorizer.fit_transform(all_key_ingredients)


# Get the names of the features
features = vectorizer.get_feature_names_out()


# illustration of how the matrix looks like with respect to the ingredients 

# Convert the sparse matrix to a dense matrix
tfidf_matrix_dense = tfidf_matrix.todense()

# Convert the dense matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix_dense, columns=features)

def get_recommendations_with_score(ingredients, N):
    
    # Transform the ingredients into a TF-IDF vector
    ingredients_vector = vectorizer.transform([' '.join(ingredients)])

    # Compute the cosine similarity between the TFIDF of input ingredients and
    # TFIDF of the top 30 common key ingredients among the 674 recipes
    similarity_scores = cosine_similarity(ingredients_vector, tfidf_matrix)

    # Get the indices and scores of the top N recipes
    top_recipe_indices = np.argsort(similarity_scores[0])[-N:]
    top_recipe_scores = np.sort(similarity_scores[0])[-N:]

    # Sort the indices and scores in descending order
    top_recipe_indices = top_recipe_indices[::-1]
    top_recipe_scores = top_recipe_scores[::-1]

    # Get the top N recipe recommendations
    top_recipes = ingred_ex_df.iloc[top_recipe_indices]

    # Add the similarity scores to the DataFrame
    top_recipes['similarity_score'] = top_recipe_scores

    return top_recipes

# test_ingred = ['garlic','onion','pork']
test_ingred = ['garlic','onion','rice']

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score

# test
def get_recommendations_with_score_adjusted(ingredients, N):
    # Transform the ingredients into a TF-IDF vector
    ingredients_vector = vectorizer.transform([' '.join(ingredients)])

    # Compute the cosine similarity between the TFIDF of input ingredients and
    # TFIDF of the top 30 common key ingredients among the 674 recipes
    similarity_scores = cosine_similarity(ingredients_vector, tfidf_matrix)

    # Adjust the similarity scores favouring the least number of ingredients in each recipe
    adjusted_similarity_scores = similarity_scores[0] / (np.log1p(ingred_ex_df['n_all_ingredients'])) 

    # Get the indices and scores of the top N recipes
    top_recipe_indices = np.argsort(adjusted_similarity_scores)[-N:]
    top_recipe_scores = np.sort(adjusted_similarity_scores)[-N:]

    # Sort the indices and scores in descending order
    top_recipe_indices = top_recipe_indices[::-1]
    top_recipe_scores = top_recipe_scores[::-1]

    # Get the top N recipe recommendations
    top_recipes = ingred_ex_df.iloc[top_recipe_indices]

    # Add the similarity scores to the DataFrame
    top_recipes['similarity_score'] = top_recipe_scores
    
  
    return top_recipes

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import jaccard_score

# Initialize the MultiLabelBinarizer with the top 30 common ingredients
mlb = MultiLabelBinarizer(classes=top_30_ingredients)

# Fit and transform the 'key_ingredients' column
one_hot_encoded = mlb.fit_transform(ingred_ex_df['key_ingred_processed'])

# Convert the one-hot encoded matrix into a DataFrame
one_hot_df = pd.DataFrame(one_hot_encoded, columns=mlb.classes_)

def get_cosine_onehot(ingredients, N):
    # Create a binary vector for the input ingredients
    ingredients_vector = np.isin(mlb.classes_, ingredients).astype(int).reshape(1, -1)

    # Compute the cosine similarity between the TFIDF of input ingredients and
    # TFIDF of the top 30 common key ingredients among the 674 recipes
    similarity_scores = cosine_similarity(ingredients_vector, one_hot_encoded)[0]

    # Get the indices and scores of the top N recipes
    top_recipe_indices = np.argsort(similarity_scores)[-N:]
    top_recipe_scores = np.sort(similarity_scores)[-N:]

    # Sort the indices and scores in descending order
    top_recipe_indices = top_recipe_indices[::-1]
    top_recipe_scores = top_recipe_scores[::-1]

    # Get the top N recipe recommendations
    top_recipes = ingred_ex_df.iloc[top_recipe_indices]

    # Add the similarity scores to the DataFrame
    top_recipes['similarity_score'] = top_recipe_scores
      
    return top_recipes

def get_cosine_onehot_adjusted(ingredients, N):
    # Create a binary vector for the input ingredients
    ingredients_vector = np.isin(mlb.classes_, ingredients).astype(int).reshape(1, -1)

    # Compute the cosine similarity between the TFIDF of input ingredients and
    # TFIDF of the top 30 common key ingredients among the 674 recipes
    similarity_scores = cosine_similarity(ingredients_vector, one_hot_encoded)[0]

    # Adjust the similarity scores favouring the least number of ingredients in each recipe
    adjusted_similarity_scores = similarity_scores[0] / (np.log1p(ingred_ex_df['n_all_ingredients'])) 

    # Get the indices and scores of the top N recipes
    top_recipe_indices = np.argsort(adjusted_similarity_scores)[-N:]
    top_recipe_scores = np.sort(adjusted_similarity_scores)[-N:]

    # Sort the indices and scores in descending order
    top_recipe_indices = top_recipe_indices[::-1]
    top_recipe_scores = top_recipe_scores[::-1]

    # Get the top N recipe recommendations
    top_recipes = ingred_ex_df.iloc[top_recipe_indices]

    # Add the similarity scores to the DataFrame
    top_recipes['similarity_score'] = top_recipe_scores
      
    return top_recipes

def get_jaccard_rec(ingredients, N):
    # Create a binary vector for the input ingredients
    ingredients_vector = np.isin(mlb.classes_, ingredients).astype(int)

    # Compute the Jaccard similarity between the binary vector of input ingredients and
    # binary vectors of the key ingredients among the 674 recipes
    similarity_scores = [jaccard_score(ingredients_vector, row) for row in one_hot_encoded]

    # Adjust the similarity scores favouring the least number of ingredients in each recipe
    adjusted_similarity_scores = similarity_scores[0] 
    
    # Get the indices and scores of the top N recipes
    top_recipe_indices = np.argsort(similarity_scores)[-N:]
    top_recipe_scores = np.sort(similarity_scores)[-N:]

    # Sort the indices and scores in descending order
    top_recipe_indices = top_recipe_indices[::-1]
    top_recipe_scores = top_recipe_scores[::-1]

    # Get the top N recipe recommendations
    top_recipes = ingred_ex_df.iloc[top_recipe_indices]

    # Add the similarity scores to the DataFrame
    top_recipes['similarity_score'] = top_recipe_scores

    return top_recipes

def get_jaccard_rec_adj(ingredients, N):
    # Create a binary vector for the input ingredients
    ingredients_vector = np.isin(mlb.classes_, ingredients).astype(int)

    # Compute the Jaccard similarity between the binary vector of input ingredients and
    # binary vectors of the key ingredients among the 674 recipes
    similarity_scores = [jaccard_score(ingredients_vector, row) for row in one_hot_encoded]

    # Adjust the similarity scores favouring the least number of ingredients in each recipe
    adjusted_similarity_scores = similarity_scores[0] / (np.log1p(ingred_ex_df['n_all_ingredients'])) 
    
    # Get the indices and scores of the top N recipes
    top_recipe_indices = np.argsort(similarity_scores)[-N:]
    top_recipe_scores = np.sort(similarity_scores)[-N:]

    # Sort the indices and scores in descending order
    top_recipe_indices = top_recipe_indices[::-1]
    top_recipe_scores = top_recipe_scores[::-1]

    # Get the top N recipe recommendations
    top_recipes = ingred_ex_df.iloc[top_recipe_indices]

    # Add the similarity scores to the DataFrame
    top_recipes['similarity_score'] = top_recipe_scores

    return top_recipes

combined_df = pd.merge(meatmen_df, ingred_ex_df, on='recipe_name', how='inner')


def get_mod_jaccard_rec_adj(ingredients, N): #modifying the function to return based on recipe names so that the correct recipe is matched. 
    # Create a binary vector for the input ingredients
    ingredients_vector = np.isin(mlb.classes_, ingredients).astype(int)

    # Compute the Jaccard similarity between the binary vector of input ingredients and
    # binary vectors of the key ingredients among the 674 recipes
    similarity_scores = [jaccard_score(ingredients_vector, row) for row in one_hot_encoded]

    # Adjust the similarity scores favouring the least number of ingredients in each recipe
    adjusted_similarity_scores = similarity_scores[0] / (np.log1p(ingred_ex_df['n_all_ingredients'])) 
    
    # Get the indices and scores of the top N recipes
    top_recipe_indices = np.argsort(similarity_scores)[-N:]
    top_recipe_scores = np.sort(similarity_scores)[-N:]

    # Sort the indices and scores in descending order
    top_recipe_indices = top_recipe_indices[::-1]
    top_recipe_scores = top_recipe_scores[::-1]

    # Get the top N recipe names
    top_recipe_names = ingred_ex_df.iloc[top_recipe_indices]['recipe_name']

    # Create a DataFrame with the recipe names and similarity scores
    top_recipes = pd.DataFrame({
        'recipe_name': top_recipe_names,
        'similarity_score': top_recipe_scores
    })

    return top_recipes


def show_recommendations(ingredients, N):
    # Get the top N recipe recommendations
    top_recipes = get_mod_jaccard_rec_adj(ingredients, N)

    # Select the desired columns from the combined dataframe
    output = combined_df[combined_df['recipe_name'].isin(top_recipes['recipe_name'])][['recipe_name', 'image', 'ingredients', 'n_all_ingredients', 'difficulty', 'n_directions', 'prep_time', 'url']]

    # Replace the 'image' column with Image objects
    # output['image'] = output['image'].apply(lambda url: Image(url=url))

    # Set 'recipe_name' as the index for both DataFrames
    top_recipes.set_index('recipe_name', inplace=True)
    output.set_index('recipe_name', inplace=True)
    
    # Add the similarity scores to the DataFrame
    output['similarity_score'] = top_recipes['similarity_score']
    
    # Sort the DataFrame by 'similarity_score' in descending order
    output.sort_values(by='similarity_score', ascending=False, inplace=True)


    return output

def image2html(image_str):
    return "<img width=150 src='"+image_str+"' />"


def ingredients2html(ingr_list_string):
    ingr_list = literal_eval(ingr_list_string)
    returnstring = "<ul>"
    for item in ingr_list:
        returnstring = returnstring + "<li>"+item+"</li>"
    returnstring = returnstring + "</ul>"
    return returnstring

def preptime2mins(preptime_literal):
    time=0
    if(preptime_literal == 0):
        return pd.NA
    preptime_string = literal_eval(preptime_literal)[0]
    if not preptime_string:
        return pd.NA
    numbers = re.search(r'\ *(\d+)\ *',preptime_string)
    mins = re.search(r'(\d+)\ *min',preptime_string)
    hours = re.search(r'(\d+)\ *ho?u?r',preptime_string)
    if mins:
        time = time + int(mins.group(1))
    if hours:
        time = time + int(hours.group(1))*60
    if time == 0:
        if numbers:
            return int(numbers.group(1))
        return pd.NA
    else:
        return time

def formatname(recipeInfo):
    return '<a href="'+recipeInfo[2]+'">'+urllib.parse.unquote(recipeInfo[0])+'</a><br/><a href="'+recipeInfo[2]+'">'+recipeInfo[1]+'</a><br/>'

def cleanup_text(df):
    df['image'] = df['image'].apply(image2html)
    df['ingredients'] = df['ingredients'].apply(ingredients2html)
    df['Name of Recipe'] = df[['Name of Recipe','image','url']].apply(formatname,axis=1)
    return df

from roboflow import Roboflow
robo_rf = Roboflow(api_key="1QuR0PDlumqwc0YT70dT")
robo_project = robo_rf.workspace().project("ingredient-detection")
robo_model = robo_project.version(4).model

def image_prediction(filePath):

    # visualise prediction
    # model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

    # save prediction labels for recommendation logic
    prediction_json = robo_model.predict(filePath, confidence=40, overlap=30).json()
    return prediction_json

### End of Sandra ipynb code


@app.route('/')
def main():
    return render_template("index.jinja2", ingredients=flask_disp_ingredients)

@app.route('/image_detection',methods=['POST'])
def detect_image():
    f = request.files['file-select']
    s_name = secure_filename(f.filename)
    f_name = os.path.join(THIS_FOLDER / "uploads",s_name)
    f.save(f_name)
    results = image_prediction(f_name)
    os.remove(f_name)
    return results

@app.route('/results',methods=['POST'])
def results():

    
    form_ingredients = request.form.getlist('ingredients')
    recs = show_recommendations(form_ingredients, 10000)
    top_rec = show_recommendations(form_ingredients, 1)

    if 'difficulty' in request.form:
        diffrecs = pd.DataFrame()
        for difficulty in request.form.getlist('difficulty'):
            diffrecs = pd.concat([diffrecs,recs[recs['difficulty'] == difficulty]])
        recs = diffrecs
        recs.sort_values(by='similarity_score', ascending=False, inplace=True)

    recs = recs[recs['n_directions'] <= int(request.form['steps'])]
    recs = recs[recs['n_all_ingredients'] <= int(request.form['ingrno'])]
    recs['prep_time_integer'] = recs['prep_time'].fillna(0).apply(preptime2mins)
    recs = recs[recs['prep_time_integer'] <= int(request.form['prept'])]
    recs.sort_values(by='similarity_score', ascending=False, inplace=True)

    reco_range = int(request.form['reco_range'])
    recs["Name of Recipe"] = recs.index
    range_df = cleanup_text(recs.head(reco_range))
    print(range_df)
    display_df = range_df[['Name of Recipe','ingredients','n_all_ingredients','difficulty','n_directions','prep_time_integer']].copy()
    display_df = display_df.rename(columns={"ingredients": "Ingredients Required", "n_all_ingredients": "No of Ingredients","difficulty":"Difficulty","n_directions":"No of Steps","prep_time_integer":"Prep Time (mins)"})
    reccos = display_df.to_html(render_links=True,escape=False, index=False)
    return render_template("results.jinja2",data=reccos,form=request.form, ingredients=flask_disp_ingredients)

