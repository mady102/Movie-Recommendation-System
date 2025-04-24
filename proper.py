from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.spinner import Spinner
from kivy.core.window import Window

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Set app background color
Window.clearcolor = (0.1, 0.1, 0.1, 1)

# Load and preprocess dataset
df = pd.read_csv("tmdb_with_moods_enhanced_cleaned.csv")
df.fillna({'overview': '', 'genres': '', 'release_date': '', 'mood': ''}, inplace=True)
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
df['combined_features'] = df['overview'] + " " + df['genres'] + " " + df['mood']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# KNN Model
knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
knn_model.fit(tfidf_matrix)

# Unique moods for spinner
mood_options = sorted(list(df['mood'].unique()))

def recommend_movies_knn(year, mood):
    mood_text = mood.lower().strip()
    mood_vec = vectorizer.transform([mood_text])
    distances, indices = knn_model.kneighbors(mood_vec)

    filtered = []
    for idx in indices[0]:
        movie = df.iloc[idx]
        if movie['year'] == year or movie['year'] in [year - 1, year + 1]:
            result = f"{movie['title']} ({movie['year']})\nGenre: {movie['genres']}\nOverview: {movie['overview'][:300]}..."
            filtered.append(result)
    return filtered if filtered else ["⚠ No matching movies found."]

class StyledLabel(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = (1, 1, 1, 1)
        self.font_size = kwargs.get("font_size", 18)
        self.bold = True
        self.size_hint_y = None
        self.height = kwargs.get("height", 30)

class MovieRecommenderUI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=20, spacing=15, **kwargs)

        self.add_widget(StyledLabel(text='🎬 Movie Recommendation System', font_size=26, height=50))

        self.add_widget(StyledLabel(text='Enter Release Year:'))
        self.year_input = TextInput(
            multiline=False, size_hint_y=None, height=40,
            background_color=(0.2, 0.2, 0.2, 1), foreground_color=(1, 1, 1, 1),
            hint_text="e.g. 2015"
        )
        self.add_widget(self.year_input)

        self.add_widget(StyledLabel(text='Select Mood:'))
        self.mood_spinner = Spinner(
            text="Choose a mood",
            values=mood_options,
            size_hint_y=None,
            height=40,
            background_color=(0.3, 0.3, 0.3, 1),
            color=(1, 1, 1, 1)
        )
        self.add_widget(self.mood_spinner)

        self.recommend_button = Button(
            text='🎥 Recommend Movies', background_color=(1, 0.3, 0.4, 1),
            color=(1, 1, 1, 1), size_hint_y=None, height=50
        )
        self.recommend_button.bind(on_press=self.show_recommendations)
        self.add_widget(self.recommend_button)

        self.scroll_view = ScrollView(size_hint=(1, 1))
        self.results_box = GridLayout(cols=1, spacing=10, size_hint_y=None, padding=10)
        self.results_box.bind(minimum_height=self.results_box.setter('height'))
        self.scroll_view.add_widget(self.results_box)
        self.add_widget(self.scroll_view)

    def show_recommendations(self, _):
        self.results_box.clear_widgets()
        try:
            year = int(self.year_input.text.strip())
            mood = self.mood_spinner.text

            if mood == "Choose a mood":
                raise ValueError("No mood selected.")

            results = recommend_movies_knn(year, mood)
            for movie in results:
                label = StyledLabel(text=movie, font_size=16)
                label.text_size = (self.width - 40, None)
                label.bind(texture_size=label.setter('size'))
                self.results_box.add_widget(label)

        except ValueError:
            self.results_box.add_widget(StyledLabel(
                text="⚠ Please enter a valid year and select a mood.",
                font_size=16
            ))

class MovieApp(App):
    def build(self):
        return MovieRecommenderUI()

if __name__ == '__main__':
    MovieApp().run()
