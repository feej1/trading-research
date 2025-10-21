
import pandas as pd
import stockUtil as stockClient
import datetime
import argparse
from math import log, sqrt, exp, erf
import numpy
import warnings


## TODO account for stock splits; current solution only works for stocks without splits in timeframe of interest

if __name__ == "__main__":

    # Ignore only FutureWarnings from a specific module (e.g., pandas)
    warnings.filterwarnings('ignore', category=FutureWarning)

    parser = argparse.ArgumentParser(description='Generate CSV of stock options data with calculated fields.')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    parser.add_argument('startDate', type=lambda d: datetime.datetime.strptime(d, '%Y-%m-%d').date(), help='Start date for options data (YYYY-MM-DD)')
    parser.add_argument('endDate', type=lambda d: datetime.datetime.strptime(d, '%Y-%m-%d').date(), help='End date for options data (YYYY-MM-DD)')
    parser.add_argument('shortestContract', type=int, help='The longest contract to include (in days)')
    parser.add_argument('longestContract', type=int, help='The shortest contract to include (in days)')
    parser.add_argument('--output', type=str, help='Output CSV file name')
    
    args = parser.parse_args()
    
    tkr = args.ticker
    startDate = args.startDate
    endDate = args.endDate
    shortestContract = args.shortestContract
    longestContract = args.longestContract
    output_file = args.output

    stockPrices_df = stockClient.getStockTimeSeriesDaily(tkr, dataAfter=str(startDate - datetime.timedelta(days=longestContract + 1)), dataBefore=str(endDate + datetime.timedelta(days=longestContract + 1)))
    print(f"Stock prices retrieved: {len(stockPrices_df)} entries")

    yields_df = stockClient.getYieldRate(yieldsAfter=str(startDate - datetime.timedelta(days=1)), yieldsBefore=str(endDate + datetime.timedelta(days=1)))
    print(f"Yield data points retrieved {len(yields_df)}")

    for date in stockPrices_df.index:

        # Only process dates within the specified range, extra are retrieved to calculate volatility
        if date.date() < startDate or date.date() > endDate:
            continue

        datObj = pd.to_datetime(date)
        print(f"Processing date: {datObj.date()}")

        options_df = stockClient.getStockOptions(tkr, str(datObj.date()), expireBefore=str(datObj + datetime.timedelta(days=longestContract)), expireAfter=str(datObj + datetime.timedelta(days=shortestContract)), type="call")
        print(f"Options retrieved: {len(options_df)} entries")

        vol = stockClient.getStockVolatility(stockPrices_df, str(datObj.date()), sampleSize=7, period=252) # trading days in a year
        print(f"Annualized Volatility: {vol}")

        riskFreeRate = yields_df.loc[str(datObj.date())]

        #  Using options and calculate midpoint of bid and ask, black scholes price and if option was profitable and add to result df
        results_df = pd.DataFrame(columns=['date', 'expiration', 'contractLength', 'type', 'strike', 'bid', 'ask', 'midpoint', 'blackScholesPrice', 'stockPriceAtExpiration', 'profitable', 'return'])

        for index, row in options_df.iterrows():
            # print(f"Processing option: {row.to_dict().keys()}")
            # print(f"Processing index: {index}")

            strike = float(row['strike'])
            bid = float(row['bid'])
            ask = float(row['ask'])
            expiration = row['expiration']
            option_type = row['type']
            contractLength = (expiration - datObj).days
  
            midpoint = (bid + ask) / 2.0

            time_to_expiration = (expiration - datObj).days / 365.0  # in years
            S = float(stockPrices_df.loc[str(datObj.date())].to_dict()['1. open'])  # Stock price at option purchase date
            K = strike
            r = 0.05  # Risk-free interest rate, assumed constant for simplicity
            sigma = vol / 100.0  # Convert percentage to decimal

            # use numpy for normal cdf instead of scipy
            norm_cdf = lambda x: 0.5 * (1.0 + erf(x / numpy.sqrt(2.0)))

            # Black-Scholes formula for call option price
            d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * time_to_expiration) / (sigma * sqrt(time_to_expiration))
            d2 = d1 - sigma * sqrt(time_to_expiration)
            blackScholesPrice = S * norm_cdf(d1) - K * exp(-r * time_to_expiration) * norm_cdf(d2)
            
            # Stock price at expiration
            if str(expiration.date()) in stockPrices_df.index:
                stockPriceAtExpiration = float(stockPrices_df.loc[str(expiration.date())].to_dict()['1. open'])
            else:
                # If expiration date is not a trading day, use the most recent trading day before expiration
                # future_dates = stockPrices_df[stockPrices_df.index < str(expiration.date()) and stockPrices_df.index > str(datObj.date())]
                future_dates = stockPrices_df[stockPrices_df.index < str(expiration.date())]
                future_dates = future_dates[future_dates.index > str(datObj.date())]
                if not future_dates.empty:
                    stockPriceAtExpiration = float(future_dates.iloc[-1].to_dict()['1. open'])
                else:
                    print(f"Could not find valid price of stock at expiration for option {row['contractID']}")
                    continue


            

            intrinsic_value = max(0, stockPriceAtExpiration - K)
            profitable = intrinsic_value > midpoint
            if profitable:
                return_pct = ((intrinsic_value - midpoint) / midpoint * 100) 
            else:
                # assumes trader will take still trade to avoid full loss even stock didn't make it to a profitable price
                (max(-midpoint, intrinsic_value) / midpoint) * 100 


            newRow_df = pd.DataFrame([{
                'date': datObj.date(),
                'expiration': expiration.date(),
                'contractLength': contractLength,
                'type': option_type,
                'strike': strike,
                'bid': bid,
                'ask': ask,
                'midpoint': midpoint,
                'blackScholesPrice': blackScholesPrice,
                'stockPriceAtExpiration': stockPriceAtExpiration,
                'profitable': profitable,
                'return': return_pct, ## when options is not profitable return is the minimum of either the cost of the option or still excercising the option
            }])
            results_df = pd.concat([results_df, newRow_df], ignore_index=True) 

            
    if not output_file:
        output_file = f"{tkr}_{startDate.isoformat()}_{endDate.isoformat()}_values.csv"
    results_df.to_csv(output_file, index=False)



    

    


