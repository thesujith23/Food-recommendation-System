# 🍲 Food Recommendation System

A machine learning-based recommendation system that suggests food items to users based on their search queries. This project utilizes natural language processing and cosine similarity to find and rank the most relevant food items from a dataset.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

---

## 📋 Table of Contents

- [About The Project](#-about-the-project)
- [How It Works](#-how-it-works)
- [Built With](#-built-with)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Contact](#-contact)

---

## 📖 About The Project

Finding the right food can be a challenge. This project aims to simplify the process by providing a smart recommendation system. By inputting a food name, the system processes a large dataset of Indian food items and returns a list of the top 10 most similar dishes based on their names, ingredients, and descriptions.

The core of this system is a content-based filtering approach that calculates the similarity between food items to provide accurate and relevant suggestions.

---

## ⚙️ How It Works

The recommendation engine follows these steps:
1.  **Data Loading & Preprocessing:** The system loads the "IndianFoodDatasetCSV.csv" and cleans the data for consistency.
2.  **Feature Engineering:** It creates a combined "features" column from various attributes like `TranslatedRecipeName`, `Cuisine`, `Course`, etc.
3.  **Text Vectorization:** The text features are converted into a numerical format using `TfidfVectorizer`, which helps in understanding the importance of each word.
4.  **Similarity Calculation:** It computes the **cosine similarity** between the user's input and all the food items in the dataset.
5.  **Recommendation:** The system sorts the food items by their similarity scores and returns the top 10 matches.



---

## 💻 Built With

This project is built using core Python libraries for data science and machine learning.

- **Python:** The primary programming language.
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical operations.
- **Scikit-learn:** For implementing the TF-IDF Vectorizer and calculating cosine similarity.
- **Jupyter Notebook:** For developing and showcasing the model.

---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/thesujith23/Food-recommendation-System.git](https://github.com/thesujith23/Food-recommendation-System.git)
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd Food-recommendation-System
    ```
3.  **Install the required libraries:**
    ```bash
    pip install numpy pandas scikit-learn
    ```
4.  **Run the Jupyter Notebook:**
    Open the `Food Recommendation System.ipynb` file in Jupyter Notebook or JupyterLab to see the code and run the cells.

---

## Usage

To get a food recommendation, simply call the `recommend()` function with the name of a food item as a string.

**Example:**
In a code cell within the notebook, you can run:

```python
recommend('poha')
