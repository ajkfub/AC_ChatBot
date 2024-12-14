import numpy
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
spacy.cli.download("en_core_web_sm")
import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
data_directory = os.path.join(parent_dir, "data/enquiries_data")

# Step 2: Find optimal k using Silhouette Score
def find_optimal_k(X, max_k, plot):
    silhouette_scores = []
    k_values = range(1075, max_k + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
        # print(score)
        if score >0.5:
          k_values = range (1075,k+1)
          break

    if plot:
        # Plotting the Silhouette Scores
        plt.figure(figsize=(500, 500))
        plt.plot(k_values, silhouette_scores, marker='o')
        plt.title('Silhouette Scores for Different k')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.xticks(k_values)
        plt.grid()
        plt.show()

    # Get the optimal k
    optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
    # print(f"Optimal number of clusters (k): {optimal_k}")
    return optimal_k

# Step 3: Suggest similar questions based on user input
def suggest_similar_questions(vectorizer, user_question, enquiries, optimal_k, X):
    # Add the user question to the list of inquiries
    all_inquiries = enquiries + [user_question]

    # Vectorize all inquiries including the user question
    X_all = vectorizer.transform(all_inquiries)

    # Clustering with KMeans
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(X)  # Fit on the original data

    # Assign clusters to all inquiries
    clusters = kmeans.predict(X_all)

    # Find the cluster of the user question
    user_cluster = clusters[-1]

    # Extract questions from the same cluster, excluding the user question
    suggested_questions = [
        all_inquiries[i] for i in range(len(all_inquiries))
        if clusters[i] == user_cluster and i < len(enquiries)
    ][:3]  # Limit to 3 suggestions

    return suggested_questions

if __name__ == "__main__":

    # Load the English NLP model
    nlp = spacy.load("en_core_web_sm")

    # Step 1: Text Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')

    data_path = os.path.join(data_directory, "enquiries.csv")
    enquiries = list(pd.read_csv(data_path)["enquiries"])
    X = vectorizer.fit_transform(enquiries)

    # Find the optimal k
    optimal_k = find_optimal_k(X,1450, plot=False)

    # Example usage
    user_input = "What is EBITDA?"  # Replace with dynamic user input
    suggestions = suggest_similar_questions(vectorizer, user_input, enquiries, optimal_k, X)

    print(f"Suggested questions for '{user_input}':")
    for question in suggestions:
        print(f"- {question}")

    # Example usage
    user_input = "What is turnover?"  # Replace with dynamic user input
    suggestions = suggest_similar_questions(user_input, enquiries, optimal_k)

    print(f"Suggested questions for '{user_input}':")
    for question in suggestions:
        print(f"- {question}")






