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
    recommended_books = [{'TITLE': df['TITLE'].iloc[i],
                          'AUTHOR': df['AUTHOR'].iloc[i],
                          'YEAR': df['YEAR'].iloc[i],
                          'GENRE': df['GENRE'].iloc[i],
                          'SCORE': round(score * 100, 2),
                          'DESCRIPTION': df['DESCRIPTION'].iloc[i]
                          }
                            for i, score in sim_scores]
    return recommended_books

user_input = input(Fore.RED + "\nIngresa el titulo de tu libro favorito: " + Style.RESET_ALL)
recs = recommend_books(user_input)

if recs:
    print(f"\nLibros similares a '{user_input}':\n")
    for i, book in enumerate(recs, 1):
        print(f"{i}. {Fore.LIGHTMAGENTA_EX + book['TITLE'] + Style.RESET_ALL}{f' ({book['YEAR']})'}{f'- {Fore.GREEN + book['GENRE'] + Style.RESET_ALL}'}{f' by {Fore.BLUE +book['AUTHOR'] + Style.RESET_ALL}'}{""}\n  {book['DESCRIPTION']}\n\n{'  Similarity Score: ' + Fore.YELLOW + str(book['SCORE']) + '%' + Style.RESET_ALL}\n")
else:
    print(f"Lo siento, el libro {user_input} no se encontró en la base de datos.")
