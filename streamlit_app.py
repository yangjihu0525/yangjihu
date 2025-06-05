import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("📊 AI 기반 종합 주가 분석 & 산업군 비교 & LSTM 예측")

# --- 종목 입력 ---
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

    # 산업군 주요 종목 비교
    def get_industry_peers(ticker):
        info = yf.Ticker(ticker).info
        sector = info.get('sector')
        peers = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'INTC'],
            'Consumer Cyclical': ['TSLA', 'NKE', 'SBUX', 'HD', 'MCD'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
            'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABBV', 'TMO'],
            'Communication Services': ['FB', 'GOOGL', 'NFLX', 'DIS', 'VZ'],
            # 필요시 더 추가 가능
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

    # LSTM 딥러닝 모델 예측
    st.subheader("🤖 LSTM 기반 단기 주가 예측")

    data_close = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(data_close)

    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    seq_length = 20
    X, y = create_sequences(scaled_close, seq_length)

    if len(X) > 0:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=16, verbose=0)

        # 최근 5일 예측
        last_seq = scaled_close[-seq_length:]
        preds = []
        current_seq = last_seq.reshape(1, seq_length, 1)
        for _ in range(5):
            pred = model.predict(current_seq)[0][0]
            preds.append(pred)
            current_seq = np.append(current_seq[:,1:,:],[[[pred]]],axis=1)

        preds_prices = scaler.inverse_transform(np.array(preds).reshape(-1,1))

        for i, price in enumerate(preds_prices):
            st.write(f"{i+1}일 후 예상 주가: {price[0]:.2f}")
    else:
        st.write("데이터가 충분하지 않아 LSTM 예측을 실행할 수 없습니다.")

