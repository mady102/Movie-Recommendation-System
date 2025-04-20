from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.floatlayout import FloatLayout

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Set app background color
Window.clearcolor = (0.1, 0.1, 0.1, 1)

# Load and preprocess the dataset
df = pd.read_csv("tmdb_cleaned_dataset-use.csv")
df['overview'] = df['overview'].fillna('')
df['genres'] = df['genres'].fillna('')
df['release_date'] = df['release_date'].fillna('')
df['mood'] = df['mood'].fillna('')
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
df['combined_features'] = df['overview'] + " " + df['genres'] + " " + df['mood']

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# Recommendation function
def recommend_movies(input_year, input_mood, top_n=10):
    input_vec = vectorizer.transform([input_mood])
    cosine_sim = cosine_similarity(input_vec, tfidf_matrix).flatten()
    
    filtered_df = df[df['year'] == input_year]
    if filtered_df.empty:
        prev_df = df[df['year'] == input_year - 1]
        next_df = df[df['year'] == input_year + 1]
        extended_df = pd.concat([prev_df, next_df])
        if extended_df.empty:
            return ["No movies found in nearby years either."]
        year_indices = extended_df.index
        scores = cosine_sim[year_indices]
        top_indices = scores.argsort()[::-1][:top_n]
        return [f"{row['title']} ({row['year']}): {row['genres']}" for _, row in extended_df.iloc[top_indices].iterrows()]

    year_indices = filtered_df.index
    scores = cosine_sim[year_indices]
    top_indices = scores.argsort()[::-1][:top_n]
    return [f"{row['title']} ({row['year']}): {row['genres']}" for _, row in filtered_df.iloc[top_indices].iterrows()]

# Custom styled label
class StyledLabel(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = (1, 1, 1, 1)
        self.font_size = 20
        self.bold = True

class MovieRecommender(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=20, spacing=15, **kwargs)

        self.add_widget(StyledLabel(text='🎬 Movie Recommendation System', font_size=28))

        self.add_widget(StyledLabel(text='Enter the Year:'))
        self.year_input = TextInput(multiline=False, size_hint_y=None, height=40, background_color=(0.2, 0.2, 0.2, 1), foreground_color=(1, 1, 1, 1))
        self.add_widget(self.year_input)

        self.add_widget(StyledLabel(text='Enter your Mood:'))
        self.mood_input = TextInput(multiline=False, size_hint_y=None, height=40, background_color=(0.2, 0.2, 0.2, 1), foreground_color=(1, 1, 1, 1))
        self.add_widget(self.mood_input)

        self.submit_btn = Button(text='🎥 Recommend Movies', background_color=(1, 0, 0.3, 1), color=(1, 1, 1, 1), size_hint_y=None, height=50)
        self.submit_btn.bind(on_press=self.get_recommendations)
        self.add_widget(self.submit_btn)

        self.results_box = GridLayout(cols=1, spacing=10, size_hint_y=None, padding=[10])
        self.results_box.bind(minimum_height=self.results_box.setter('height'))

        self.scroll = ScrollView(size_hint=(1, 1))
        self.scroll.add_widget(self.results_box)
        self.add_widget(self.scroll)

    def get_recommendations(self, instance):
        self.results_box.clear_widgets()
        try:
            year = int(self.year_input.text)
            mood = self.mood_input.text.strip()
            results = recommend_movies(year, mood)
            for movie in results:
                self.results_box.add_widget(StyledLabel(text=movie, font_size=16))
        except ValueError:
            self.results_box.add_widget(StyledLabel(text="Please enter a valid year and mood.", font_size=16))

class MovieApp(App):
    def build(self):
        return MovieRecommender()

if __name__ == '__main__':
    MovieApp().run()
