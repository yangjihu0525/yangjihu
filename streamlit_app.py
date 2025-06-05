import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import requests
from textblob import TextBlob

# 뉴스 API 키 (Streamlit Secrets에 등록 권장)
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY")

st.title("📊 AI 기반 종합 주가 분석 & 뉴스 감성 분석")

# 1. 종목 입력 및 주가 데이터 가져오기
ticker = st.text_input("종목 코드를 입력하세요 (예: AAPL, TSLA)", "AAPL")
data = yf.download(ticker, period="6mo")

# 2. 주가 시각화 및 기술적 지표
st.subheader("최근 6개월 주가")
st.line_chart(data["Close"])

st.subheader("기술적 지표 (단순 이동 평균)")
data["SMA20"] = data["Close"].rolling(window=20).mean()
data["SMA50"] = data["Close"].rolling(window=50).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], name="SMA 20"))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], name="SMA 50"))

st.plotly_chart(fig)

# 3. 뉴스 가져오기 & 감성 분석 함수
def fetch_news(ticker):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&"
        f"language=en&"
        f"sortBy=publishedAt&"
        f"apiKey={NEWS_API_KEY}"
    )
    response = requests.get(url)
    if response.status_code != 200:
        return []
    articles = response.json().get("articles", [])
    return articles

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # -1 ~ 1 (부정~긍정)

# 4. 뉴스 & 감성 분석 출력
st.subheader("📰 최신 뉴스 및 감성 분석")

if not NEWS_API_KEY:
    st.warning("뉴스API 키를 streamlit secrets에 NEWS_API_KEY로 등록해주세요.")
else:
    if ticker:
        news_list = fetch_news(ticker)
        if not news_list:
            st.write("뉴스를 불러올 수 없습니다.")
        else:
            positive, negative = 0, 0
            for article in news_list[:10]:
                title = article["title"]
                desc = article.get("description") or ""
                content = title + " " + desc
                polarity = analyze_sentiment(content)
                if polarity > 0.1:
                    positive += 1
                elif polarity < -0.1:
                    negative += 1
                st.markdown(f"**{title}**")
                st.write(f"감성 점수: {polarity:.2f}")
                st.write(article["url"])
                st.write("---")
            st.write(f"긍정 뉴스 수: {positive}, 부정 뉴스 수: {negative}")
