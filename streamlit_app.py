import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.title("📈 딥러닝 LSTM 주가 예측 앱")

# 종목 입력
ticker = st.text_input("종목 코드를 입력하세요 (예: AAPL)", "AAPL")

# 데이터 불러오기
data = yf.download(ticker, period="1y")
if data.empty:
    st.error("데이터를 불러올 수 없습니다.")
    st.stop()

st.subheader("종가 차트 (최근 1년)")
st.line_chart(data["Close"])

# 데이터 전처리 - 종가만 사용, 정규화
close_prices = data["Close"].values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_prices = scaler.fit_transform(close_prices)

# 시계열 데이터셋 생성 함수
def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X, Y = create_dataset(scaled_prices, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)  # LSTM 입력형태 맞춤

# LSTM 모델 구축
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step,1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

st.write("⏳ 딥러닝 모델 학습 중입니다. 잠시만 기다려 주세요...")

# 학습 (에포크 적당히 설정, 메모리나 시간 고려)
model.fit(X, Y, epochs=5, batch_size=32, verbose=0)

st.success("학습 완료!")

# 예측하기
train_predict = model.predict(X)
train_predict = scaler.inverse_transform(train_predict.reshape(-1,1))
Y_true = scaler.inverse_transform(Y.reshape(-1,1))

# 결과 시각화
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index[time_step+1:], y=Y_true.flatten(), name="실제 종가"))
fig.add_trace(go.Scatter(x=data.index[time_step+1:], y=train_predict.flatten(), name="예측 종가"))
fig.update_layout(title=f"{ticker} 주가 예측 vs 실제", xaxis_title="날짜", yaxis_title="가격(USD)")
st.plotly_chart(fig)

# 앞으로 30일 예측
last_60_days = scaled_prices[-time_step:]
temp_input = last_60_days.reshape(1,-1)[0].tolist()

lst_output = []
n_steps = time_step
next_days = 30

for i in range(next_days):
    if len(temp_input) > n_steps:
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, n_steps, 1)
        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        temp_input = temp_input[1:]
        lst_output.append(yhat[0][0])
    else:
        x_input = temp_input.reshape(1, n_steps, 1)
        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])

future_pred = scaler.inverse_transform(np.array(lst_output).reshape(-1,1))

future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=next_days)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=data.index, y=data["Close"], name="실제 종가"))
fig2.add_trace(go.Scatter(x=future_dates, y=future_pred.flatten(), name="미래 30일 예측"))
fig2.update_layout(title=f"{ticker} 미래 주가 예측 (30일)", xaxis_title="날짜", yaxis_title="가격(USD)")
st.plotly_chart(fig2)
