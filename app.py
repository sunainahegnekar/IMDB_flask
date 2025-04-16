from flask import Flask, render_template, request
import pandas as pd
from scipy import stats

app = Flask(__name__)

try:
    from flask import Flask, render_template, request
except ImportError as e:
    print("Error: Flask is not installed. Please install it using 'pip install flask'.")
    exit(1)

try:
    import pandas as pd
except ImportError as e:
    print("Error: pandas is not installed. Please install it using 'pip install pandas'.")
    exit(1)

try:
    from scipy import stats
except ImportError as e:
    print("Error: scipy is not installed. Please install it using 'pip install scipy'.")
    exit(1)

app = Flask(__name__)

# Load and clean the dataset
try:
    df = pd.read_csv("https://raw.githubusercontent.com/krishna-koly/IMDB_TOP_1000/main/imdb_top_1000.csv")
    df['runtime'] = df['Runtime'].str.extract(r'(\d+)').astype(float)
    df['gross'] = df['Gross'].str.replace(',', '', regex=True).astype(float)
    df.columns = df.columns.str.strip().str.lower()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

def top_movies_by_genre(genre, df, top_n=5, sort_order='descending'):
    try:
        df['genre'] = df['genre'].str.lower()
        filtered = df[df['genre'].str.split(', ').apply(lambda x: any(genre.lower() in g for g in x))]
        if filtered.empty:
            return None
        if sort_order == 'ascending':
            return filtered.sort_values(by='imdb_rating', ascending=True).head(top_n)[['series_title', 'genre', 'imdb_rating']]
        else:  # default to descending
            return filtered.sort_values(by='imdb_rating', ascending=False).head(top_n)[['series_title', 'genre', 'imdb_rating']]
    except Exception as e:
        print(f"Error in top_movies_by_genre: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/overview')
def overview():
    try:
        shape = df.shape
        columns = df.columns.tolist()
        dtypes = df.dtypes.to_dict()
        return render_template('overview.html', rows=shape[0], cols=shape[1], columns=columns, dtypes=dtypes)
    except Exception as e:
        return render_template('error.html', error=f"Error generating overview: {str(e)}")

@app.route('/statistics')
def statistics():
    try:
        stats = df.describe(include='all').T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']].to_dict()
        unique_values = {
            'series_title': df['series_title'].nunique(),
            'director': df['director'].nunique(),
            'genres': df['genre'].nunique()
        }
        return render_template('statistics.html', stats=stats, unique_values=unique_values)
    except Exception as e:
        return render_template('error.html', error=f"Error generating statistics: {str(e)}")

@app.route('/genre', methods=['GET', 'POST'])
def genre():
    movies = None
    genre_input = ''
    top_n = 5
    sort_order = 'descending'
    genre_options = ['Action', 'Comedy', 'Drama', 'Horror', 'Adventure']
    sort_options = ['descending', 'ascending']
    try:
        if request.method == 'POST':
            genre_input = request.form.get('genre', '').strip()
            try:
                top_n = int(request.form.get('top_n', 5))
                if top_n < 1 or top_n > 10:
                    top_n = 5
            except ValueError:
                top_n = 5
            sort_order = request.form.get('sort_order', 'descending')
            if genre_input in genre_options:
                movies = top_movies_by_genre(genre_input, df, top_n, sort_order)
        return render_template('genre.html', movies=movies, genre_input=genre_input, top_n=top_n, genre_options=genre_options, sort_order=sort_order, sort_options=sort_options)
    except Exception as e:
        return render_template('error.html', error=f"Error processing genre search: {str(e)}")

@app.route('/hypothesis')
def hypothesis():
    try:
        action_movies = df[df['genre'].str.contains('Action', case=False, na=False)]
        comedy_movies = df[df['genre'].str.contains('Comedy', case=False, na=False)]
        action_budgets = action_movies['gross'].dropna()
        comedy_budgets = comedy_movies['gross'].dropna()
        if len(action_budgets) < 2 or len(comedy_budgets) < 2:
            raise ValueError("Insufficient data for t-test")
        t_statistic, p_value = stats.ttest_ind(action_budgets, comedy_budgets, nan_policy='omit')
        result = {
            't_statistic': round(t_statistic, 2),
            'p_value': round(p_value, 3),
            'significant': p_value < 0.05
        }
        return render_template('hypothesis.html', result=result)
    except Exception as e:
        return render_template('hypothesis.html', result={
            't_statistic': 'N/A',
            'p_value': 'N/A',
            'significant': False,
            'error': str(e)
        })

@app.route('/charts')
def charts():
    return render_template('charts.html')

if __name__ == '__main__':
    app.run(debug=True)