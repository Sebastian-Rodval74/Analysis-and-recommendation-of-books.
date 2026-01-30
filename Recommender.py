import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore, Style
df = pd.read_csv('books.csv')


vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['DESCRIPTION'].fillna(''))
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)



def recommend_books(title, df=df, cosine_sim=cosine_sim):
    df_lower = df['TITLE'].str.lower()
    title_lower = title.lower()
    if title_lower not in df_lower.values:
        return None
    
    idx = df_lower[df_lower == title_lower].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    max_score = sim_scores[0][1] if sim_scores else 1
    recommended_books = []
    for i, score in sim_scores:
        normalized_score = round((score / max_score) * 100, 2)
        if normalized_score >= 75:
            label = 'Very strong thematic match'
        elif normalized_score >= 50:
            label = 'Strong thematic overlap'
        elif normalized_score >= 30:
            label = 'Moderate similarity'
        else:
            label = 'Loose conceptual connection'

        recommended_books.append({
            'TITLE': df['TITLE'].iloc[i],
            'AUTHOR': df['AUTHOR'].iloc[i],
            'YEAR': df['YEAR'].iloc[i],
            'GENRE': df['GENRE'].iloc[i],
            'SCORE': normalized_score,
            'LABEL': label,
            'DESCRIPTION': df['DESCRIPTION'].iloc[i]
        })
    return recommended_books

user_input = input(Fore.RED + "\nEnter the title of your favorite book: " + Style.RESET_ALL)
recs = recommend_books(user_input)

if recs:
    print(f"\nBooks similar to  '{user_input}':\n")
    for i, book in enumerate(recs, 1):
        print(
    f"{i}. "
    f"{Fore.LIGHTMAGENTA_EX}{book['TITLE']}{Style.RESET_ALL} "
    f"({book['YEAR']}) - "
    f"{Fore.GREEN}{book['GENRE']}{Style.RESET_ALL} by "
    f"{Fore.BLUE}{book['AUTHOR']}{Style.RESET_ALL}\n"
    f"  {book['DESCRIPTION']}\n"
    f" \nSimilarity Score: {Fore.YELLOW}{book['SCORE']}%{Style.RESET_ALL}\n"
    f"Match Type: {Fore.CYAN}{book['LABEL']}{Style.RESET_ALL}\n"
)
else:
    print(f"Lo siento, el libro {user_input} no se encontró en la base de datos.")
