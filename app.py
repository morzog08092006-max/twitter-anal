import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
import datetime
import numpy as np

# Налаштування сторінки
st.set_page_config(page_title="X (Twitter) Analyzer", layout="wide")

st.title("📊 Аналізатор акаунтів Twitter/X")

# --- БІЧНА ПАНЕЛЬ: ЗАВАНТАЖЕННЯ ДАНИХ ---
st.sidebar.header("Джерело даних")
data_source = st.sidebar.radio(
    "Оберіть метод отримання даних:", 
    ("Завантаження CSV", "Отримання через API")
)

        
@st.cache_data
def load_data_from_csv(file):
    df = pd.read_csv(file)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

def mock_api_data(username):
    # Імітація відповіді від API для розробки
    dates = pd.date_range(end=datetime.datetime.today(), periods=150)
    data = {
        'date': dates,
        'text': [f"Це тестовий твіт номер {i} від @{username}. Very good!" if i % 2 == 0 
                 else f"Сьогодні поганий день... Bad news {i}" for i in range(150)],
        'likes': np.random.randint(0, 500, size=150),
        'retweets': np.random.randint(0, 100, size=150),
        'replies': np.random.randint(0, 50, size=150)
    }
    return pd.DataFrame(data)

df = None

if data_source == "Завантаження CSV":
    uploaded_file = st.sidebar.file_uploader("Завантажте ваш CSV-звіт", type=["csv"])
    st.sidebar.markdown("*Формат CSV повинен містити колонки:* date, text, likes, retweets, replies")
    if uploaded_file is not None:
        try:
            df = load_data_from_csv(uploaded_file)
        except Exception as e:
            st.error(f"Помилка читання файлу: {e}")
else:
    username = st.sidebar.text_input("Введіть юзернейм (без @):", "developer")
    if st.sidebar.button("Отримати дані"):
        with st.spinner("Підключення до API..."):
            df = mock_api_data(username)

# --- ОСНОВНА ЧАСТИНА: АНАЛІЗ ТА ВІЗУАЛІЗАЦІЯ ---
if df is not None:
    st.success("Дані успішно завантажено!")
    
    # Перевірка наявності необхідних колонок
    required_cols = ['text', 'date']
    if not all(col in df.columns for col in required_cols):
        st.error(f"У файлі відсутні обов'язкові колонки. Потрібні як мінімум: {required_cols}")
    else:
        # Заповнення відсутніх колонок метрик нулями
        for col in ['likes', 'retweets', 'replies']:
            if col not in df.columns:
                df[col] = 0
                
        # 1. Обчислення активності та Engagement
        st.subheader("1. Активність користувача та Engagement")
        
        df['engagement'] = df['likes'] + df['retweets'] + df['replies']
        avg_engagement = df['engagement'].mean()
        total_tweets = len(df)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Загальна кількість твітів", total_tweets)
        col2.metric("Середній Engagement (на 1 твіт)", f"{avg_engagement:.1f}")
        col3.metric("Сумарна кількість взаємодій", df['engagement'].sum())
        
        st.divider()

        # 2. Динаміка публікацій (Візуалізація)
        st.subheader("2. Динаміка публікацій")
        
        # Групуємо твіти по днях
        df['date_only'] = df['date'].dt.date
        timeline_df = df.groupby('date_only').size().reset_index(name='tweets_count')
        
        fig_timeline = px.line(
            timeline_df, x='date_only', y='tweets_count', 
            title="Кількість публікацій з часом",
            labels={'date_only': 'Дата', 'tweets_count': 'Кількість публікацій'},
            markers=True
        )
        fig_timeline.update_traces(line_color='#1DA1F2') # Фірмовий колір Twitter
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        st.divider()

# 3. Аналіз тональності (Sentiment Analysis)
        st.subheader("3. Елементарний Sentiment Analysis")
        st.markdown("*Примітка: Базовий аналіз працює найкраще з англомовним текстом.*")
        
        def analyze_sentiment(text):
            analysis = TextBlob(str(text))
            # Визначаємо полярність (від -1.0 до 1.0)
            if analysis.sentiment.polarity > 0.05:
                return 'Позитивний'
            elif analysis.sentiment.polarity < -0.05:
                return 'Негативний'
            else:
                return 'Нейтральний'
        
        with st.spinner("Аналізуємо тональність текстів..."):
            df['sentiment'] = df['text'].apply(analyze_sentiment)
        
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Тональність', 'Кількість']
        
        col_pie, col_table = st.columns([1, 1])
        
        with col_pie:
            fig_pie = px.pie(
                sentiment_counts, 
                names='Тональність', 
                values='Кількість',
                color='Тональність',
                color_discrete_map={
                    'Позитивний': '#00CC96', 
                    'Нейтральний': '#636EFA', 
                    'Негативний': '#EF553B'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_table:
            st.dataframe(df[['date_only', 'text', 'sentiment']].head(15), use_container_width=True)

else:
    st.info("Будь ласка, завантажте CSV-файл або скористайтеся демо-API для початку аналізу.")