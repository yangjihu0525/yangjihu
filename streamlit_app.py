import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.title("📈 딥러닝 LSTM 주가 예측 앱")

# 1. 종목 코드 입력
ticker = st.text_input("종목 코드를 입력하세요 (예: AAPL)", "AAPL")

# 2. 데이터 불러오기 (최근 1년)
data = yf.download(ticker, period="1y")
if data.empty:
    st.error("데이터를 불러올 수 없습니다. 종목 코드를 확인하세요.")
    st.stop()

st.subheader("최근 1년 종가")
st.line_chart(data["Close"])

# 3. 데이터 전처리 (종가만, 0~1 사이로 정규화)
close_prices = data["Close"].values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_prices = scaler.fit_transform(close_prices)

# 4. 시계열 데이터셋 만들기 함수 (60일치 입력 → 다음 날 종가 예측)
def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X, Y = create_dataset(scaled_prices, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)  # LSTM 입력 형식 맞추기

# 5. LSTM 모델 구축 및 컴파일
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_step,1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. 모델 학습 안내 및 실행
st.write("⏳ 딥러닝 모델 학습 중입니다... (약 1~2분 소요)")
model.fit(X, Y, epochs=10, batch_size=32, verbose=0)
st.success("모델 학습 완료!")

# 7. 학습 데이터에 대한 예측 및 시각화
predicted = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(Y.reshape(-1,1))

fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index[time_step+1:], y=real_prices.flatten(), name="실제 종가"))
fig.add_trace(go.Scatter(x=data.index[time_step+1:], y=predicted_prices.flatten(), name="예측 종가"))
fig.update_layout(title=f"{ticker} 주가 예측 (학습 데이터)", xaxis_title="날짜", yaxis_title="가격(USD)")
st.plotly_chart(fig)

# 8. 미래 30일 예측
last_60_days = scaled_prices[-time_step:]
temp_input = last_60_days.flatten().tolist()

lst_output = []
n_steps = time_step
next_days = 30

for i in range(next_days):
    x_input = np.array(temp_input[-n_steps:])
    x_input = x_input.reshape(1, n_steps, 1)
    yhat = model.predict(x_input, verbose=0)[0][0]
    temp_input.append(yhat)
    lst_output.append(yhat)

future_predicted = scaler.inverse_transform(np.array(lst_output).reshape(-1,1))
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=next_days)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=data.index, y=data["Close"], name="실제 종가"))
fig2.add_trace(go.Scatter(x=future_dates, y=future_predicted.flatten(), name="미래 30일 예측"))
fig2.update_layout(title=f"{ticker} 미래 30일 주가 예측", xaxis_title="날짜", yaxis_title="가격(USD)")
st.plotly_chart(fig2)
