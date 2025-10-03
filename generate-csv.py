
import requests
import pandas as pd
import stockUtil as stockClient
import datetime
import argparse




## TODO account for stock splits; current solution only works for stocks without splits in timeframe of interest

if __name__ == "__main__":

    stockData = stockClient.getStockTimeSeriesDaily("AAPL", dataBefore=datetime.date(2023, 6, 1), dataAfter=datetime.date(2023, 5, 1))
    vol = stockClient.getStockVolatility(stockData, datetime.date(2023, 6, 25), sampleSize=7, period=252) # trading days in a year
    print(f"Annualized Volatility: {vol}")
    exit(0)

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

    stockPrices_df = stockClient.getStockTimeSeriesDaily(tkr, dataBefore=startDate + datetime.timedelta(days=1), dataAfter=startDate - datetime.timedelta(days=1))

    currDate = startDate
    while currDate <= endDate:

        print(f"Processing date: {currDate}")


        if currDate in stockPrices_df.index:
            options_df = stockClient.getStockOptions(tkr, startDate, expireBefore=currDate + datetime.timedelta(days=longestContract), expireAfter=currDate + datetime.timedelta(days=shortestContract))
    



        currDate += datetime.timedelta(days=1)



