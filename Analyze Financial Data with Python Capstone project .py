# 10-30-2023.py
# Young client wants a portfolio of stocks that will weight each stock in the portfolio and provide finance stats to evaluate
# Imports needed
from http import client
from tracemalloc import stop
from turtle import done
from venv import create
from matplotlib import axis
import matplotlib
import requests
import sys
import csv
import pandas_datareader.data as web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from datetime import datetime, timedelta
import seaborn as sns
import cvxopt as opt
import cvxopt.blas as blas
from cvxopt import solvers
from cvxopt import matrix as opt_matrix, solvers
from sklearn.metrics import mean_squared_error
from pandas_market_calendars import get_calendar
from pandas.tseries.offsets import BDay
from scipy.optimize import minimize
from sympy import false

# Define a function to check if a date is a trading day and determine trading day count
def is_trading_day(date, calendar):
    date = pd.to_datetime(date)
    return calendar.valid_days(start_date=date, end_date=date).shape[0] == 1

start_date = "2023-06-01"
# end_date = datetime.today().strftime("%Y-%m-%d")
end_date = "2023-09-01"
s_date = start_date
e_date = end_date

# Create a calendar for the NYSE which is the same for NASDAQ
nyse = get_calendar("XNYS")

# Find the last valid trading day before the specified end date
end_date = pd.to_datetime(end_date)
while not is_trading_day(end_date, nyse):
    end_date -= pd.DateOffset(days=1)

# Initialize a counter for trading days
trading_days_count = 0

# Convert the start and end dates to datetime objects
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

current_date = datetime.today()
current_date = pd.to_datetime(current_date)

# Loop through the dates and count trading days
current_date = start_date
while current_date <= end_date:
    if is_trading_day(current_date, nyse):
        trading_days_count += 1
    current_date += pd.DateOffset(days=1)

# print("Number of trading days on the NYSE:", trading_days_count)
# Convert start_date and end_date back to strings for rest of program
start_date = start_date.strftime(s_date)
end_date = end_date.strftime(e_date)

# Get Data Input
# def manage_tickers():
    # tickers = []
    # while True:
        # action = input("Enter 'add' to add a ticker, 'delete' to delete a ticker, or 'd' when you are finished:  ")

        # if action.lower() == 'add':
            # ticker = input("Enter one ticker symbol in upper caps:   ")
            # tickers.append(ticker)

        # elif action.lower() == 'delete':    
            # ticker = input("Enter the ticker symbol to delete:   ")
            # if ticker in tickers:
                # tickers.remove(ticker)
            # else:
                # print("Ticker not found.")

        # elif action.lower() == 'd':
            # return ', '.join(tickers)
        # else:
            # print("Invalid action. Please enter 'add', 'delete' or 'd'.")
# print(manage_tickers())

# Stock picks (INTC, VOO, ABML, LLY, GM, SHY)
token = "3315a436693679ce0f1f7a2167b278bed7850fe7"  
tickers = ["INTC", "PG", "IDYA", "FDX", "JPM", "MGA", "SHY", "BRK-B", "XOM", "STLA", "BKR", "MSCI", "TXRH"]
print("\n\nCurrent tickers include: ",tickers)
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Token {token}"
}

url = f"https://api.tiingo.com/tiingo/daily/prices"

params = {
    "tickers": ",".join(tickers),
    "startDate": start_date,
    "endDate": end_date
}

response = requests.get(url, headers=headers, params=params)
EOD_prices = pd.DataFrame()
if response.status_code == 200:
    ticker_data = response.json()

    data_list = []

    for ticker in tickers:
        # print(f"Ticker: {ticker}")
        for data in ticker_data:
            if data['ticker'] == ticker:
                for entry in data['priceData']:
                    date = entry['date']
                    open_price = entry['adjOpen']
                    high_price = entry['adjHigh']
                    low_price = entry['adjLow']
                    close_price = entry['adjClose']
                    volume = entry['adjVolume']
                    dividend_cash = entry.get('divCash', None)
                    split_factor = entry.get('splitFactor', None)

                    data_list.append({
                        'DATE': date,
                        'Ticker': ticker,
                        'Open': open_price,
                        'High': high_price,
                        'Low': low_price,
                        'Close': close_price,
                        'Volume': volume,
                        'Dividend Cash': dividend_cash,
                        'Split Factor': split_factor
                    })

    df = pd.DataFrame(data_list)
    df['DATE'] = pd.to_datetime(df['DATE'])
    EOD_prices = df[['DATE', 'Ticker', 'Close',]]
    EOD_prices.set_index('DATE', inplace=True, drop=True)
    # print(EOD_prices, "\n")
    
else:
    print("Error retrieving ticker data.")
# FRED data on S&P 500. Note Tiingo does NOT provide index data
# FRED data on dates includes the last date
date_format = "%Y-%m-%d"
date_obj = datetime.strptime(end_date, date_format)
end_date_FRED = date_obj - timedelta(days=1)
end_date_str = end_date_FRED.strftime(date_format)

# SP500 data from the FRED
SP500 = web.DataReader(['sp500'], 'fred', start_date, end_date_str)
df_SP = pd.DataFrame(SP500)
# print(df_SP)

if 'sp500' in df_SP.columns:
    df_SP.rename(columns={'sp500': 'Close'}, inplace=True)
    df_SP['Ticker'] = 'SP500'
    # df_SP.reset_index(drop=True,inplace=True)
    df_SP.rename(columns={'index': 'DATE'}, inplace=False)
    df_SP = df_SP[['Ticker', 'Close']]

# Concatenate date_df with df_SP
combined_df = pd.concat([EOD_prices, df_SP], ignore_index=False)

combined_df.reset_index(inplace=True)  # Reset index and drop the old index to combine
combined_df['DATE'] = pd.to_datetime(combined_df['DATE'])
# Create a pivot table with tickers as columns and 'DATE' as the index
pivot_df = combined_df.pivot(index='DATE', columns='Ticker', values='Close')
pivot_df.iloc[:, 1:] = pivot_df.iloc[:, 1:].round(2)

#  Clean data in df
pivot_prep = pivot_df.fillna(0)
# Eliminate all rows where values are zero
mask = (pivot_prep !=0).any(axis=1)
pivot_prep1 = pivot_prep[mask]
pivot_prep1 = pivot_prep1.replace(0.0, 0.000001)

#  Check combined_df for missing dates with data
combined_df_md = pivot_prep1
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
elapsed_days = (end_date - start_date).days

combined_df_md.index = pd.to_datetime(combined_df_md.index)

port_prices = combined_df_md.drop('SP500', axis=1) # To be used for optimization

custom_path5 = r'/Users/michaelkos/Documents/Finance/Python/Spreadsheets with Fin data/port_prices.csv'
port_prices.to_csv(custom_path5)

# Filter for NYSE weekday rows (Monday to Friday on NYSE minus holidays) within the specified date range
trading_days = nyse.schedule(start_date, end_date)

# pivot_prep.index = pd.to_datetime(pivot_prep.index) # For testing

# Filter your DataFrame to include only the rows that correspond to NYSE trading days
df_on_trading_days = combined_df_md.loc[combined_df_md.index.isin(trading_days.index)]

# Check for missing data in the filtered DataFrame
missing_data_on_trading_days = df_on_trading_days.isna().any().any()

if missing_data_on_trading_days:
    print("There is missing data on NYSE trading days.")
    # Identify where there is missing data
    missing_data = combined_df_md.isna()
    # Get the ticker and date where data is missing
    missing_data_locations = [(date, ticker) for date, row in missing_data.iterrows() for ticker, is_missing in row.items() if is_missing]
    # Print the locations of missing data
    for location in missing_data_locations:
        print(f"Missing data at date {location[0]} for ticker {location[1]}")
        print("This missing data will use the first valid observation in the column to function as compensation.")
        # Replace 0.000001 with NaN
        combined_df_md.replace(0.000001, np.nan, inplace=True)
        # Get the first valid value in each column
        first_valid = combined_df_md.apply(lambda col: col.dropna().iloc[0])
        # Fill NaN values with the first valid value in their column
        combined_df_md.fillna(first_valid, inplace=True)
        combined_df_md.to_csv(custom_path5)
else:
    print("No missing data on NYSE trading days.")

# Check if the modified DataFrame is different from the original
if not (pd.read_csv(custom_path5)).equals(pd.read_csv(custom_path5)):
    print("The DataFrame has been modified.")

pivot_return = combined_df_md.pct_change()   # This is daily returns including date and NaN
pivot_return = pivot_return.fillna(0) # Remove NaN and insert a 0

# Create 2 dataframes, 1 without SP500 and one with only SP500, now there will be 3 dataframes.
port = pivot_return.drop('SP500', axis=1)
# print(port)
SP500_df = pivot_return[['SP500']]
# print(f"\nRETURNS for port\n\n{port}")
# print(f"\nRETURNS for SP500\n\n{SP500_df}")

custom_path1 = r'/Users/michaelkos/Documents/Finance/Python/Spreadsheets with Fin data/port.csv'
port.to_csv(custom_path1)


# Plot closing values for each ticker, fig.1
# Lines will be discontinuous so interpolation step is needed
# Resample the DataFrame with daily frequency
daily_df = combined_df_md.resample('D').asfreq()
# Fill missing values using linear interpolation
daily_df = combined_df_md.interpolate(method='linear')
# Create a figure and axis
sns.set_style('whitegrid')
plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
ax = plt.gca()
# Iterate through unique tickers and plot their data on the same graph
# unique_tickers = combined_df['Ticker'].unique()
for ticker in daily_df.columns:
    ax.plot(daily_df.index, daily_df[ticker], label=ticker)

# Set labels and title for adjusted closing prices
plt.xlabel("Date")
plt.ylabel("Adjusted Closing Price")
plt.title("Adjusted Closing Prices")
plt.legend()
plt.grid(True)
# plt.show()    
plt.clf

# Plot Daily ROR, fig. 2
# Create a figure and axis
sns.set_style('whitegrid')
plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
ax = plt.gca()
# Iterate through unique tickers and plot their data on the same graph
unique_tickers = pivot_return.columns.unique()
for ticker in pivot_return.columns:
    ax.plot(pivot_return.index, pivot_return[ticker], label=ticker)

# Set labels and title
plt.xlabel("Date")
plt.ylabel("'Daily Return'")
plt.title("Daily Return")
plt.legend()
plt.grid(True)
# plt.show()     

# Calculate Mean daily ROR for each ticker then portfolio then SP500
mean_daily_ror = pivot_return.sum() / (trading_days_count - 2)
# print(mean_daily_ror)

# Iterate through unique tickers and print mean daily ROR
for ticker in unique_tickers:
    ticker_data = pivot_return.columns == ticker
# for ticker in unique_tickers:
    # print(f"Mean Daily ROR for {ticker}: {mean_daily_ror[ticker]}")

# print mean daily ROR for portfolio with equal weighted tickers
tickers_port = port.columns  
# Calculate the daily returns for the portfolio but could just take mean_daily_ror & drop SP500
portfolio_returns = port[tickers_port].sum(axis=1)  # Mean ROR for port each day. 
# First value is NaN convert to 0.
portfolio_returns = portfolio_returns.fillna(0)
# print(portfolio_returns)

tickers_SP500 = SP500_df.columns
SP500_returns = SP500_df[tickers_SP500].sum(axis=1) 
# print("\n",SP500_returns)

# Calculate port mean daily returns for each ticker = expected returns
portfolio_returns = port.sum() / (trading_days_count - 2)

num_of_tickers = len(port.columns)
# portfolio_expected_returns = portfolio_returns.sum() / (trading_days_count - 2) / num_of_tickers
weights = 1 / num_of_tickers 
portfolio_expected_returns = (portfolio_returns.sum()) * weights * 1

# portfolio_expected_returns_annualized_c = (((portfolio_expected_returns + 1) ** 252) -1) * 100   # This is percent annualized return compounded
# portfolio_expected_returns_annualized_s = portfolio_expected_returns * 252 * 100 # This is non compounded simple percent

# Get mean ROR for SP500
SP500_m = SP500_df.sum() / (trading_days_count - 2)  # Not usefull

# Calculate the mean daily ROR for the equally weighted portfolio
SP500_mean_return = SP500_returns.sum() / (trading_days_count - 2) 
# print(SP500_mean_return)

# Annualized returns compounded daily versus simple
portfolio_expected_returns_annualized_c = ((1 +portfolio_expected_returns) **252) - 1 # This is compounded
portfolio_expected_returns_annualized_s = portfolio_expected_returns * 252 # This is simple

# Bar plot mean_daily_ror which is the same as expected returns, fig. 3
fig, ax = plt.subplots(figsize=(12, 6))
legend_labels = unique_tickers.tolist() + ['Portfolio'] * 1
bars = plt.bar(unique_tickers.tolist() + ['Portfolio'], mean_daily_ror.tolist() + [portfolio_expected_returns], color=["green", "orange", "blue", "red", "purple", "brown", "gray", "yellow", "pink","cyan", "magenta"])
plt.xlabel('Ticker')
plt.ylabel('Mean Daily ROR')
plt.title('Mean Daily ROR for Each Stock and Portfolio')
plt.xticks(rotation=45)  # Rotate labels by 45 degrees
plt.legend(bars, legend_labels)
# plt.show()

# Plot portfolio ticker correlation matrix, fig. 4
corr_matrix = port.corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.4f')
plt.title('Correlation Matrix Heatmap for Initial Portfolio')
# plt.show()
# print(corr_matrix)

cov_matrix = port.cov()   # Not using an annualized number or I would multiply by 252
# print(cov_matrix)

# Variance of Daily ROR for each ticker and Potfolio with equally weighted tickers
variance = pivot_return.var() # * 252**0.5 * 100 # Annualized percent Give var for all tickers
# num_of_tickers = len(port.columns)

weights_port = np.array([weights] * num_of_tickers) # need a np array for variance calculation
port_tickers_var = port.var() # * 252**0.5 * 100  # Give var for each ticker in port for variance of daily ROR

variance_port = np.dot(weights_port.T, np.dot(cov_matrix, weights_port))

variance_SP = 1 * SP500_df.var().sum() # * 252**0.5 * 100
# For annualized variance, multiply by 252**0.5 * 100 percent
# print("Variance\n", variance)
# print("Port tickers variance\n", port_tickers_var)
# print("Variance for Portfolio\n", variance_port)
# print("Variance for SP500\n", variance_SP)

# Plot variance, fig. 5
# Combine the variances for tickers and portfolio
variance_values = variance.tolist() + [variance_port]

# Create labels for tickers and portfolio
ticker_labels = unique_tickers.tolist()
portfolio_label = ['Portfolio']
ticker_labels_and_portfolio = ticker_labels + portfolio_label

fig, ax = plt.subplots(figsize=(12, 6))
plt.bar(ticker_labels_and_portfolio, variance_values, color = ["green", "orange", "blue", "red", "purple", "brown", "gray", "yellow", "pink", "cyan", "magenta"])
plt.xlabel('Ticker')
plt.ylabel('Variance')
plt.title('Variance ROR for Each Stock and Portfolio')
plt.xticks(rotation=45)  # Rotate labels by 45 degrees
plt.legend(bars, legend_labels)
# plt.show()

# Sharpe ratios not rolling and rolling
# Define the risk-free rate = 3% and annualization factor
risk_free_rate = 0.03 / 252
# risk_free_rate =  (((1 + 0.03)**(1/252)) - 1)  # This is daily risk free rate for calcs daily
annualization_factor_std_dev = 252 ** 0.5  # sqrt(252) for daily std_dev, for 252 trading days a year
# annualization_factor_ror = ((daily_ror + 1)**252 -1)
# Define the rolling window size in days
rolling_window = 2 # Most use 60 days with daily data and 36 months with monthly data
# Portfolio weights are specified as equal as already defined

# Calculate the mean and standard deviation of returns for the rolling window sharpe values
port_ret = port.mean(axis=1)
port_rolling_mean = port_ret.rolling(window=rolling_window).mean()
port_rolling_std = port_ret.rolling(window=rolling_window).std() #Where is the cov data, Hmmm. supicious
SP500_rolling_mean = SP500_df.rolling(window=rolling_window).mean()
SP500_rolling_std = SP500_df.rolling(window=rolling_window).std()

# Calculate the excess returns over the risk-free rate for rolling sharpe
portfolio_xs_returns = port_rolling_mean - risk_free_rate
SP500_excess_returns = SP500_rolling_mean - risk_free_rate

# Calculate the rolling Sharpe ratio
port_rolling_sharpe = ((portfolio_xs_returns / port_rolling_std) * annualization_factor_std_dev).dropna()
SP500_rolling_sharpe = ((SP500_excess_returns / SP500_rolling_std) * annualization_factor_std_dev).dropna()

# Graph the rolling sharpe for portfolio and SP500, fig. 6
# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))
# Plot the rolling Sharpe ratios for the portfolio and SP500
ax.plot(port_rolling_sharpe.index, port_rolling_sharpe, label='Portfolio Rolling Sharpe', color='blue')
ax.plot(SP500_rolling_sharpe.index, SP500_rolling_sharpe, label='SP500 Rolling Sharpe', color='red')

# Set labels and title
plt.xlabel("Date")
plt.ylabel("'Rolling Sharpe Ratio'")
plt.title("Daily Rolling Sharpe Ratio for Portfolio and SP500")
plt.legend()
# plt.show()   

# Create a heatmap, fig. 7
cov_matrix1 = cov_matrix * 10000
plt.figure(figsize=(10, 7))
sns.heatmap(cov_matrix1, annot=True, cmap='coolwarm', fmt='.4f')
plt.title('Covariance Matrix Heatmap for Initial Portfolio')
# plt.show()

weights_port = np.array([weights] * num_of_tickers) # need a np array for std dev calculation
# print("weights_port:", weights_port)

# Standard Deviation of Daily ROR for each ticker, portfolio and SP500
std_dev = pivot_return.std() * annualization_factor_std_dev * 100 # for all tickers including SP500 * 100 for percent
# std_dev_port_tickers = port.std() * annualization_factor_std_dev * 100 # std_dev for all port tickers
# std_dev_port = weights * port.std().sum() *** This is incorrect!!
std_dev_port = (np.sqrt(np.dot(weights_port.T, np.dot(cov_matrix, weights_port)))) * (annualization_factor_std_dev) * 100
std_dev_port1 = (np.sqrt(np.dot(weights_port.T, np.dot(cov_matrix, weights_port)))) 
std_dev_SP500 = SP500_returns.std() * annualization_factor_std_dev * 100
std_dev_SP5001 = SP500_returns.std() 

# Plot std_dev, fig. 8
std_values = std_dev.tolist() + [std_dev_port]
ticker_labels = ticker_labels_and_portfolio  # Exclude the portfolio label
# ticker_labels.append("Portfolio")  # Add a label for the portfolio
fig, ax = plt.subplots(figsize=(12, 6))
plt.bar(ticker_labels, std_values, color=["green", "orange", "blue", "red", "purple", "brown", "gray", "yellow", "pink", "cyan", "magenta"])
plt.xlabel('Ticker')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation for Each Stock and Portfolio')
plt.xticks(rotation=45)  # Rotate labels by 45 degrees
plt.legend(bars, legend_labels)
# plt.show()

# Calculate the Sharpe ratio for the portfolio & SP500
portfolio_xs_returns = (((portfolio_expected_returns + 1)**252 - 1)) - (risk_free_rate * 252)
# portfolio_expected_returns_annualized_c = (portfolio_expected_returns * annualization_factor_std_dev * 100) - (risk_free_rate * 252)  # 100 to convert to percent
SP500_xs_r = (((SP500_mean_return + 1)**252 - 1)) - (risk_free_rate * 252)    
SP500_xs_r_a = (((SP500_mean_return + 1)**252 - 1)) - (risk_free_rate * 252)    # 100 to convert to percent
portfolio_sharpe = (portfolio_xs_returns / std_dev_port) * 100
SP500_sharpe_ratio = (SP500_xs_r / std_dev_SP500) * 100

# print("SP500 Returns: " ,SP500_returns)
print("\nLast day of data Entry:   ", end_date)
print("\nTrading Days Count", trading_days_count)
print("\nElapsed Days: ", elapsed_days)
print("\nPortfolio Daily Mean Returns:", portfolio_expected_returns)
# print("\n\Portfolio Returns:", portfolio_returns)
# print("\nSP500 Daily Mean Returns", SP500_mean_return)
# print("\nSP500 Daily Mean Returns_m", SP500_m)
print("\nSP500 Expected Returns Annualized", SP500_xs_r_a)
print("\nPortfolio Expected Returns Annualized", portfolio_xs_returns)
print(f"\nExpected Returns Portfolio Annualized:{portfolio_expected_returns_annualized_c:.6f}")
print(f"\nDaily Risk Free Rate {risk_free_rate:.6f}")
print(f"\nPortfolio Sharpe Ratio: {portfolio_sharpe:.4f}")
print(f"\nSP500 Sharpe Ratio:  {SP500_sharpe_ratio:.4f}")
print(f"\nStandard Deviation Portfolio:  {std_dev_port:.4f}")
print(f"\nStandard deviation SP500:  {std_dev_SP500:.4f}")

# Optimal Portfolio Weighting
# weights to weights_o and portfolio_expected_returnsto expected_results_o for optimized portfolio

# Polynomial fitting requires a lot of data to work with. If less than 60 trading days will use scipy's optimization

# Standard Quadratic Optimization
def return_portfolios(expected_returns, cov_matrix):
    port_returns = []
    port_volatility = []
    stock_weights = []

    selected = list(expected_returns.index)
    num_assets = len(selected) 
    num_portfolios = 500
    
    for single_portfolio in range(num_portfolios):
        weights_o = np.random.random(num_assets)
        weights_o /= np.sum(weights_o)
        returns = np.dot(weights_o, expected_returns)
        volatility = np.sqrt(np.dot(weights_o.T, np.dot(cov_matrix, weights_o)))
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights_o)
        
    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility}
    
    for counter,symbol in enumerate(selected):
        portfolio[symbol +' Weight'] = [Weight[counter] for Weight in stock_weights]
    
    df = pd.DataFrame(portfolio)
    
    column_order = ['Returns', 'Volatility'] + [stock+' Weight' for stock in selected]
    df = df[column_order]
    return df
   

portfolio_daily_ROR = pd.read_csv(custom_path1)
# Remove rows with missing values (NaN)
portfolio_daily_ROR = portfolio_daily_ROR.iloc[1:]
# print(portfolio_daily_ROR)
# Assuming you have identified the non-numeric columns that need to be removed
non_numeric_columns = ["DATE"] # Add column names that are non-numeric

# Remove non-numeric columns
portfolio_daily_ROR = portfolio_daily_ROR.drop(columns=non_numeric_columns)

# Convert remaining columns to numeric data types
portfolio_daily_ROR = portfolio_daily_ROR.apply(pd.to_numeric, errors='coerce')

def optimal_portfolio(portfolio_daily_ROR):
    N = 2000 # This is the number of iterations
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)] # mus determines the range of expected portfolio_daily_ROR you are considering
    n = portfolio_daily_ROR.shape[1]
    S = opt.matrix(np.cov(portfolio_daily_ROR, rowvar=False))  # Covariance matrix
    pbar = opt.matrix(np.mean(portfolio_daily_ROR, axis=0))  # Expected portfolio_daily_ROR

    # Define constraints (e.g., individual asset weight limits)
    G = -opt.matrix(np.eye(n))  # Negative identity matrix for weights <= 1
    h = opt.matrix(0.0, (n, 1))  # All weights >= 0

    # Add the lower bound constraint for weights (>= 0)
    lb = opt_matrix(0.0, (n, 1))

    # Define constraint to ensure the sum of weights equals 1
    max_asset_weight = 0.8
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(max_asset_weight)

    solvers.options['show_progress'] = False
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b, options={'show_progress': False})['x']
                  for mu in mus]

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [opt.solvers.qp(opt.matrix(mu * S), -pbar, G, h, A, b, lb=lb)['x'] for mu in mus]

    # Calculate portfolio returns and risks
    optimal_portfolio_returns = [blas.dot(pbar, x) for x in portfolios]
    optimal_portfolio_risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]

    # Calculate the Sharpe ratio for each portfolio
    sharpe_ratios = [r / d for r, d in zip(optimal_portfolio_returns, optimal_portfolio_risks)]

    # Find the portfolio with the highest Sharpe ratio
    max_sharpe_ratio_index = sharpe_ratios.index(max(sharpe_ratios))

    # Get the optimal portfolio weights
    optimal_weights = portfolios[max_sharpe_ratio_index]
    return optimal_weights, optimal_portfolio_returns, optimal_portfolio_risks

    # Calculate the 2nd-degree polynomial of the frontier curve
    # m1 = np.polyfit(portfolio_daily_ROR, optimal_portfolio_risks, 2)
    # x1 = -m1[1] / (2 * m1[0])  # Calculate the vertex (minimum risk point)

    # Calculate the optimal portfolio weights at the minimum risk point
    # wt = opt.solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b, lb=lb)['x']
    # returoptimal_portfolio_list_lengthnp.asarray(wt), optimal_portfolio_returns, optimal_portfolio_risks

# Run the optimal portfolio function
weights_optimal, optimal_portfolio_returns, optimal_portfolio_risks = optimal_portfolio(portfolio_daily_ROR)

# Normalize the weights
total_weight = sum(weights_optimal)
normalized_weights = [w / total_weight for w in weights_optimal]

optimal_portfolio_list = list(portfolio_daily_ROR.columns[1:])
optimal_portfolio_list_length = len(optimal_portfolio_list)
weights_list = [round(normalized_weights[i], 6) for i in range(optimal_portfolio_list_length)]

optimum_portfolio1 = {'Asset': optimal_portfolio_list, 'Weight': weights_list}
optimum_portfolio_df = pd.DataFrame(optimum_portfolio1)

optimum_portfolio = optimum_portfolio_df[optimum_portfolio_df['Weight'] > 0.010]
# Set 'Asset' as the index in optimum_optimum
optimum_portfolio.set_index('Asset', inplace=True)
# print(optimum_optimum)

tickers_to_include = optimum_portfolio.index.tolist()
# Filter the portfolio prices DataFrame based on the tickers
portfolio_prices_filtered = port_prices[tickers_to_include]
portfolio_prices_filtered = portfolio_prices_filtered.tail(1)
# portfolio_prices_filteredfolio_prices.reset_index(drop=True, inplace=True)

# Merge the two DataFrames on the 'Asset' column
portfolio_optimum_with_price = optimum_portfolio.merge(portfolio_prices_filtered, left_index=True, right_index=True, how='left')
# Reset the index to have 'Asset' as a regular column again if needed
portfolio_optimum_with_price.reset_index(inplace=True)
portfolio_optimum_with_price['Prices'] = portfolio_prices_filtered.iloc[0].values
portfolio_optimum_with_price = portfolio_optimum_with_price[['Asset', 'Weight', 'Prices']]

# stock_data = port_prices
# returns_daily = stock_data[selected].pct_change()
portfolio_expected_returns= portfolio_daily_ROR.sum() / (trading_days_count - 2)
cov_portfolio = portfolio_daily_ROR.cov()

# port_ret, mean_portfolio_return, covariance_matrix
random_portfolios = return_portfolios(portfolio_expected_returns, cov_portfolio) 
weights_o, returns, risks = optimal_portfolio(portfolio_daily_ROR[1:])

# Create optimal Portfolio and determine, Expected returns, volatility, Sharpe ratio
# Create a list of assets and corresponding weights
portfolio_optimum_with_price['Percent %'] = portfolio_optimum_with_price['Weight'] * 100 # Create 'Percent %' column as a copy of 'Weight'
portfolio_optimum_with_price = portfolio_optimum_with_price[['Asset', 'Percent %','Prices']]  # Select the desired columns
# print(portfolio_optimum_with_price)

# Find the maximum Sharpe ratio (for the optimal portfolio)
# sharpe_ratios = np.array(returns) / np.array(risks)
# optimal_index = np.argmax(sharpe_ratios)

# Get the optimal portfolio return and volatility
# optimal_return = returns[optimal_index] 
# optimal_volatility = risks[optimal_index] 

portfolio_value = 100000
# New Allocation
portfolio_optimum_with_price['Allocation in $'] = portfolio_optimum_with_price['Percent %'] * portfolio_value / 100
allocation = portfolio_optimum_with_price[['Asset', 'Percent %', 'Prices', 'Allocation in $']]
# print(allocation) #.to_string(index=False))

# Find Share number
portfolio_optimum_with_price['Share Number1'] = portfolio_optimum_with_price['Allocation in $'] / portfolio_optimum_with_price['Prices']
portfolio_optimum_with_price['Share Number'] = np.floor(portfolio_optimum_with_price['Share Number1'])
share_number = portfolio_optimum_with_price[['Asset', 'Percent %', 'Share Number']]
# print(share_number) 

# Find Residual Cash
portfolio_optimum_with_price['Cost'] = portfolio_optimum_with_price['Share Number'] * portfolio_optimum_with_price['Prices']
portfolio_optimum_with_price_cost = portfolio_optimum_with_price[['Asset', 'Share Number', 'Prices', 'Cost']]
# print(portfolio_optimum_with_price_cost)  #.to_string(index=False))

residual_cash = round(100000 - (portfolio_optimum_with_price['Cost'].sum()),2)
# print(residual_cash)

optimized_portfolio_mean_daily_ror1 = mean_daily_ror[tickers_to_include]
op_weights = optimum_portfolio['Weight'].values
# Need to convert series to a df
optimized_portfolio_mean_daily_ror_df = pd.DataFrame({'Mean ROR': optimized_portfolio_mean_daily_ror1})
optimized_portfolio_mean_daily_ror_df.index.name = 'Asset'
optimized_portfolio_mean_daily_ror_df = allocation.merge(optimized_portfolio_mean_daily_ror_df,on='Asset', how='left')
optimized_portfolio_mean_daily_ror_df['OPMDROR'] = (optimized_portfolio_mean_daily_ror_df['Percent %'] * optimized_portfolio_mean_daily_ror_df['Mean ROR'])
optimized_expected_returns = optimized_portfolio_mean_daily_ror_df['OPMDROR'].sum() / optimal_portfolio_list_length
optimized_expected_returns1 = optimized_portfolio_mean_daily_ror_df['OPMDROR'].sum() / optimal_portfolio_list_length / 5
optimized_expected_returns_a = ((1 + optimized_expected_returns) **252) -1 # This is compounded

# Std dev for optimized oprtfolio need to get optimal_portfolio_ROR then cov then std dev
optimal_portfolio_ROR = portfolio_daily_ROR[tickers_to_include]
optimal_portfolio_cov = optimal_portfolio_ROR.cov()
optimal_portfolio_std_dev = round(np.sqrt(np.dot(op_weights.T, np.dot(optimal_portfolio_cov, op_weights))),8) * annualization_factor_std_dev * 100
optimal_portfolio_std_dev1 = round(np.sqrt(np.dot(op_weights.T, np.dot(optimal_portfolio_cov, op_weights))),8)   # Daily
# Optimal Portfolio Sharpe
optimized_xs_returns = optimized_expected_returns_a - risk_free_rate * 252
optimized_xs_returns1 = optimized_xs_returns / 100
optimal_portfolio_sharpe = optimized_xs_returns / optimal_portfolio_std_dev * 100

# Plots Efficient Frontier fig. 9
plt.style.use('seaborn')
plt.figure(figsize=(10, 7))
random_portfolios.plot.scatter(x='Volatility', y='Returns', c='b', fontsize=12)

# Plot the optimal portfolio point
plt.scatter(optimal_portfolio_std_dev1, optimized_expected_returns1, c='r', s=200, label='Optimal Portfolio')
plt.scatter(std_dev_SP5001, SP500_mean_return, c='black', s=200, label='SP500')
try:
    plt.plot(risks, returns, 'y-o') 
except:
  pass
plt.xlim(min(risks), max(risks))
# plt.ylim(min(returns), max(returns))

plt.ylabel('Expected Returns',fontsize=14)
plt.xlabel('Volatility (Std. Deviation)',fontsize=14)
plt.title('Efficient Frontier', fontsize=24)
plt.legend()
# plt.show()

# print(f"\nMean Estimated Returns for Initial Portfolio Annualized:{portfolio_expected_returns_annualized_c}")
print(f"\nMean Estimated Returns for Optimized Portfolio: {optimized_expected_returns:.6f}") 
print(f"\nEstimated Returns for Optimized Portfolio Annualized: {optimized_expected_returns_a:.2f}") 
print(f"\nInitial Portfolio Sharpe Ratio: {portfolio_sharpe.item():.4f}")
print(f"\nOptimal Portfolio Sharpe Ratio: {optimal_portfolio_sharpe:.4f}")
print(f"\nSP500 Sharpe Ratio:  {SP500_sharpe_ratio:.4f}")
# print(f"\nOptimal Portfolio Covariance Matrix: {optimal_portfolio_cov}")
print("\nOptimal Portfolio Std Dev\n", optimal_portfolio_std_dev, optimal_portfolio_std_dev1)
print("\nOptimal Portfolio\n", optimum_portfolio)
print(f"\nPortfolio Vaue: ${portfolio_value}.00")
print(f"\nResidual Cash: ${residual_cash}")
print("\nShare Number:\n", share_number)

# Create a heatmap, fig. 9
cov_matrix2 = optimal_portfolio_cov * 100000000
plt.figure(figsize=(10, 7))
sns.heatmap(cov_matrix2, annot=True, cmap='coolwarm', fmt='.4f')
plt.title('Covariance Matrix Heatmap for Optimized Portfolio')
plt.show()
