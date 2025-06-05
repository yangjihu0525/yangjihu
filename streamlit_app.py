import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("📊 AI 기반 종합 주가 분석 & 산업군 비교 & 단순 회귀 예측")

ticker = st.text_input("🔎 종목 코드 입력 (예: AAPL, TSLA, 005930.KS)", "AAPL")

if ticker:
    data = yf.download(ticker, period="6mo")
    st.subheader(f"💹 {ticker} 주가 차트 (6개월)")
    st.line_chart(data["Close"])

    # 기술적 지표 SMA20, SMA50
    data["SMA20"] = data["Close"].rolling(window=20).mean()
    data["SMA50"] = data["Close"].rolling(window=50).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], name="SMA20"))
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], name="SMA50"))
    st.plotly_chart(fig)

    # 산업군 주요 종목 비교 (간단 예시)
    def get_industry_peers(ticker):
        info = yf.Ticker(ticker).info
        sector = info.get('sector')
        peers = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'INTC'],
            'Consumer Cyclical': ['TSLA', 'NKE', 'SBUX', 'HD', 'MCD'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
            'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABBV', 'TMO'],
            'Communication Services': ['FB', 'GOOGL', 'NFLX', 'DIS', 'VZ'],
        }
        return peers.get(sector, [ticker])

    peers = get_industry_peers(ticker)
    st.subheader(f"🏭 산업군 ({yf.Ticker(ticker).info.get('sector', '정보없음')}) 주요 종목 비교")
    st.write(", ".join(peers))

    fig_peers = go.Figure()
    for peer in peers:
        try:
            peer_data = yf.download(peer, period="6mo")
            fig_peers.add_trace(go.Scatter(x=peer_data.index, y=peer_data['Close'], name=peer))
        except Exception as e:
            st.write(f"{peer} 데이터 불러오기 실패: {e}")
    st.plotly_chart(fig_peers)

    # 단순 회귀 기반 주가 예측
    st.subheader("🤖 단순 회귀 기반 주가 예측")

    data_close = data['Close'].values.reshape(-1, 1)
    X = np.arange(len(data_close)).reshape(-1, 1)  # 날짜 인덱스
    y = data_close

    model = LinearRegression()
    model.fit(X, y)

    future_days = 5
    X_pred = np.arange(len(data_close), len(data_close) + future_days).reshape(-1, 1)
    preds = model.predict(X_pred)

    for i, price in enumerate(preds):
        st.write(f"{i+1}일 후 예상 주가: {price[0]:.2f}")
