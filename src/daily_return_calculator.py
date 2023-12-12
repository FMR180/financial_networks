import yfinance as yf
import pandas as pd


list_stocks = ["BNP.PA", "ACA.PA", "ISP.MI", "DBK.DE", "UCG.MI", "INGA.AS"]
date_start = "2022-01-01"
date_end = "2022-12-31"
df = pd.DataFrame()

for i in list_stocks:
    stock = yf.Ticker(i)
    historical_data = stock.history(start=date_start, end=date_end)
    daily_returns = historical_data['Close'].pct_change().dropna()
    daily_returns.rename(i, inplace=True)
    if df.empty:
        df = daily_returns
    else:
        df = pd.merge(df, daily_returns, left_index=True, right_index=True)

print(df)
df.to_csv("C:/Users/Flo/Documents/Uni/Master/Masterarbeit/Working File/Arbeit2/EBAFiles/stock_returns.csv")
