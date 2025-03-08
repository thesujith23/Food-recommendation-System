from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Load the dataset
df = pd.read_csv('ds1.csv', encoding='ISO-8859-1')

# Define the numeric columns based on the dataset
numeric_columns = [
    'Calories', 'Carbohydrates (g)', 'Proteins (g)', 'Fats (g)', 'Fiber (g)',
    'Vitamin A (IU)', 'Vitamin C (mg)', 'Vitamin D (IU)', 'Vitamin E (mg)', 
    'Vitamin K (mcg)', 'Vitamin B1 (mg)', 'Vitamin B2 (mg)', 'Vitamin B3 (mg)', 
    'Vitamin B6 (mg)', 'Vitamin B9 (mcg)', 'Calcium (mg)', 'Iron (mg)', 
    'Magnesium (mg)', 'Phosphorus (mg)', 'Potassium (mg)', 'Sodium (mg)', 
    'Zinc (mg)', 'Omega-3 (mg)', 'Omega-6 (mg)', 'Sugar (g)'
]

# Preprocess the dataset
df = df.drop(columns=['Unnamed: 29'], errors='ignore')
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Scale the numeric columns
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Define the helper functions from your script
def preprocess_user_input(user_input):
    processed_input = {}
    if user_input['gender'] == 'male':
        processed_input['Calories'] = 2400 if user_input['age'] >= 18 and user_input['age'] < 50 else 2000
    else:
        processed_input['Calories'] = 2000 if user_input['age'] >= 18 and user_input['age'] < 50 else 1800

    health_condition = user_input.get('health_condition', 'default')
    nutrient_targets = {
        'weight loss': {
        'Calories': -500, 'Proteins (g)': 70, 'Fats (g)': 40, 'Carbohydrates (g)': 120, 'Fiber (g)': 30, 
        'Vitamin A (IU)': 3000, 'Vitamin C (mg)': 70, 'Vitamin D (IU)': 600, 'Calcium (mg)': 1000, 'Iron (mg)': 10
    },
    'weight gain': {
        'Calories': +500, 'Proteins (g)': 90, 'Fats (g)': 100, 'Carbohydrates (g)': 300, 'Fiber (g)': 25, 
        'Vitamin B1 (mg)': 1.2, 'Vitamin B2 (mg)': 1.3, 'Vitamin B3 (mg)': 16, 'Zinc (mg)': 11, 'Omega-3 (mg)': 1500
    },
    'diabetes': {
        'Carbohydrates (g)': 100, 'Proteins (g)': 80, 'Fats (g)': 50, 'Fiber (g)': 30, 'Sugar (g)': 5, 
        'Vitamin C (mg)': 85, 'Magnesium (mg)': 400, 'Potassium (mg)': 4700, 'Omega-3 (mg)': 1000
    },
    'hypertension': {
        'Carbohydrates (g)': 200, 'Proteins (g)': 60, 'Fats (g)': 40, 'Fiber (g)': 30, 'Sodium (mg)': 1500, 
        'Vitamin D (IU)': 600, 'Calcium (mg)': 1200, 'Potassium (mg)': 4700, 'Omega-3 (mg)': 1500
    },
    'high cholesterol': {
        'Fats (g)': 40, 'Fiber (g)': 35, 'Carbohydrates (g)': 180, 'Proteins (g)': 80, 'Omega-3 (mg)': 1200,
        'Vitamin E (mg)': 15, 'Magnesium (mg)': 400, 'Potassium (mg)': 4000
    },
    'digestive health': {
        'Fiber (g)': 35, 'Magnesium (mg)': 420, 'Vitamin B3 (mg)': 16, 'Vitamin B6 (mg)': 1.3, 
        'Vitamin B9 (mcg)': 400, 'Water (L)': 2.7  # Approximate water intake in liters for digestion
    },
    'bone health': {
        'Calcium (mg)': 1300, 'Vitamin D (IU)': 800, 'Vitamin K (mcg)': 120, 'Magnesium (mg)': 420, 
        'Phosphorus (mg)': 700
    },
    'heart health': {
        'Omega-3 (mg)': 1500, 'Fiber (g)': 30, 'Potassium (mg)': 4700, 'Vitamin E (mg)': 15, 
        'Magnesium (mg)': 400, 'Sodium (mg)': 1500, 'Vitamin C (mg)': 90
    },
    'immunity boosting': {
        'Vitamin C (mg)': 90, 'Vitamin D (IU)': 600, 'Vitamin A (IU)': 3000, 'Zinc (mg)': 11, 
        'Vitamin B6 (mg)': 1.3, 'Vitamin E (mg)': 15, 'Iron (mg)': 18
    },
    'skin health': {
        'Vitamin A (IU)': 3000, 'Vitamin C (mg)': 75, 'Vitamin E (mg)': 15, 'Omega-3 (mg)': 1000, 
        'Zinc (mg)': 8, 'Water (L)': 2.7  # Approximate water intake in liters for skin hydration
    }
    }
    
    for nutrient, value in nutrient_targets.get(health_condition, {}).items():
        processed_input[nutrient] = value + processed_input.get(nutrient, 0)
    return processed_input

def filter_data_based_on_diet(df, diet_preference):
    if diet_preference == 'veg':
        return df[df['Veg/Non-Veg'] == 'Vegetarian']
    elif diet_preference == 'non-veg':
        return df[df['Veg/Non-Veg'] == 'Non-Vegetarian']
    else:
        return df

def filter_data_based_on_gender(df, gender):
    if gender == 'male':
        return df[df['Male/Female'].str.lower() == 'male']
    elif gender == 'female':
        return df[df['Male/Female'].str.lower() == 'female']
    else:
        return df

def recommend_food(processed_input, food_df, numeric_columns, top_n=3, recommended_items=None):
    if food_df.empty:
        return ["No available food items for this category"]

    if recommended_items is None:
        recommended_items = set()

    input_vector = np.array([processed_input.get(col, 0) for col in numeric_columns]).reshape(1, -1)
    food_vectors = food_df[numeric_columns].values
    
    similarities = cosine_similarity(input_vector, food_vectors)[0]
    top_n_indices = similarities.argsort()[::-1]

    unique_recommendations = []
    for idx in top_n_indices:
        food_item = food_df.iloc[idx]['food items']
        if food_item not in recommended_items:
            unique_recommendations.append(food_item)
            recommended_items.add(food_item)
        if len(unique_recommendations) == top_n:
            break

    return unique_recommendations

def recommend_meals_for_day(processed_input, filtered_df, numeric_columns):
    meals = {}
    recommended_items = set()
    for meal_time in ['Breakfast', 'Lunch', 'Dinner']:
        meal_df = filtered_df[filtered_df['Category'].str.lower() == meal_time.lower()]
        meals[meal_time] = recommend_food(processed_input, meal_df, numeric_columns, recommended_items=recommended_items)
    return meals

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = request.form['gender']
        health_condition = request.form['health_condition']
        diet_preference = request.form['diet_preference']

        user_input = {
            'age': age,
            'gender': gender,
            'health_condition': health_condition,
            'diet_preference': diet_preference
        }

        processed_input = preprocess_user_input(user_input)
        filtered_df = filter_data_based_on_diet(df, user_input['diet_preference'])
        filtered_df = filter_data_based_on_gender(filtered_df, user_input['gender'])

        meals_for_day = recommend_meals_for_day(processed_input, filtered_df, numeric_columns)
        
        return render_template('index.html', meals=meals_for_day, show_results=True)

    return render_template('index.html', show_results=False)

if __name__ == '__main__':
    app.run(debug=True)
