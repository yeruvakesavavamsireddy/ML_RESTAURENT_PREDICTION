from sklearn.metrics.pairwise import cosine_similarity

def recommend_restaurants(df, index=0):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    similarity = cosine_similarity(numeric_df)
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return scores[1:6]