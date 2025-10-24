
import pandas as pd
import stockUtil as stockClient
import datetime
import argparse
from math import log, sqrt, exp, erf
import numpy
import warnings
import asyncio


## Periodically save data to the csv as to not lose everything if the run fails

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

    # Concurrent processing of dates using asyncio.to_thread for blocking stockUtil calls.
    def process_date_sync(date):
        """Process one date synchronously and return a DataFrame of rows for that date."""
        rows = []
        # Only process dates within the specified range
        if date.date() < startDate or date.date() > endDate:
            return pd.DataFrame(rows)

        datObj = pd.to_datetime(date)
        print(f"Processing date: {datObj.date()}")

        # Blocking calls: run in thread when scheduled
        options_df = stockClient.getStockOptions(tkr, str(datObj.date()), expireBefore=str(datObj + datetime.timedelta(days=longestContract)), expireAfter=str(datObj + datetime.timedelta(days=shortestContract)), type="call")
        print(f"Options retrieved: {len(options_df)} entries")

        vol = stockClient.getStockVolatility(stockPrices_df, str(datObj.date()), sampleSize=7, period=252) # trading days in a year
        print(f"Annualized Volatility: {vol}")

        # Safe lookup for risk-free rate (fallback to 0.0)
        try:
            maybe = yields_df.loc[str(datObj.date())]
            # pick first numeric value if row/series
            if hasattr(maybe, '__iter__'):
                raw = pd.to_numeric(pd.Series(maybe).iloc[0], errors='coerce')
            else:
                raw = pd.to_numeric(maybe, errors='coerce')
            riskFreeRate = float(raw) if not pd.isna(raw) else 0.0
        except Exception:
            riskFreeRate = 0.0

        if riskFreeRate > 1:
            riskFreeRate = riskFreeRate / 100.0

        for _, row in options_df.iterrows():
            try:
                strike = float(row['strike'])
                bid = float(row['bid'])
                ask = float(row['ask'])
                expiration = row['expiration']
                option_type = row['type']
            except Exception:
                continue

            contractLength = (expiration - datObj).days
            midpoint = (bid + ask) / 2.0

            if midpoint < 0.1:
                continue

            # purchase price S
            try:
                S = float(stockPrices_df.loc[str(datObj.date())].to_dict()['1. open'])
            except Exception:
                continue

            # expiry price (use most recent prior trading day)
            try:
                if str(expiration.date()) in stockPrices_df.index:
                    stockPriceAtExpiration = float(stockPrices_df.loc[str(expiration.date())].to_dict()['1. open'])
                else:
                    future_dates = stockPrices_df[stockPrices_df.index < str(expiration.date())]
                    future_dates = future_dates[future_dates.index > str(datObj.date())]
                    if not future_dates.empty:
                        stockPriceAtExpiration = float(future_dates.iloc[-1].to_dict()['1. open'])
                    else:
                        continue
            except Exception:
                continue

            time_to_expiration = (expiration - datObj).days / 365.0
            K = strike
            r = riskFreeRate
            sigma = vol / 100.0

            norm_cdf = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))

            if time_to_expiration > 0 and sigma > 0:
                d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * time_to_expiration) / (sigma * sqrt(time_to_expiration))
                d2 = d1 - sigma * sqrt(time_to_expiration)
                blackScholesPrice = S * norm_cdf(d1) - K * exp(-r * time_to_expiration) * norm_cdf(d2)
            else:
                blackScholesPrice = max(0.0, S - K)

            intrinsic_value = max(0.0, stockPriceAtExpiration - K)
            profitable = intrinsic_value > midpoint
            if midpoint > 0:
                if profitable:
                    return_pct = ((intrinsic_value - midpoint) / midpoint * 100.0)
                else:
                    return_pct = (max(-midpoint, intrinsic_value) / midpoint) * 100.0
            else:
                return_pct = 0.0

            rows.append({
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
                'return': return_pct,
            })

        return pd.DataFrame(rows)

    async def run_all_dates():
        """Run processing for all dates concurrently.

        Returns a tuple (results_df, errors) where errors is a list of Exception
        objects encountered while processing individual dates. We use
        return_exceptions=True so one failure doesn't kill all tasks and we can
        still collect partial results.
        """
        tasks = []
        for date in stockPrices_df.index:
            if date.date() < startDate or date.date() > endDate:
                continue
            tasks.append(asyncio.create_task(asyncio.to_thread(process_date_sync, date)))

        results = []
        errors = []
        if tasks:
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            for item in completed:
                if isinstance(item, Exception):
                    errors.append(item)
                else:
                    if isinstance(item, pd.DataFrame) and not item.empty:
                        results.append(item)

        if results:
            return pd.concat(results, ignore_index=True), errors
        else:
            empty = pd.DataFrame(columns=['date', 'expiration', 'contractLength', 'type', 'strike', 'bid', 'ask', 'midpoint', 'blackScholesPrice', 'stockPriceAtExpiration', 'profitable', 'return'])
            return empty, errors

    # Run concurrent processing and capture exceptions per-task so we can save
    # partial results if something goes wrong partway through.
    results_df = pd.DataFrame()
    errors = []
    try:
        results_df, errors = asyncio.run(run_all_dates())
    except Exception as e:
        # If something goes wrong at the event loop level, capture the
        # exception and fall back to whatever was collected (likely empty).
        errors = [e]

    # Default output file when no --output provided
    if not output_file:
        output_file = f"{tkr}_{startDate.isoformat()}_{endDate.isoformat()}_values.csv"

    # If there were errors but we have partial results, name the file using
    # the last processed date (per your request). Include ticker for clarity
    # and mark it as partial in the filename.
    if errors and not results_df.empty:
        try:
            last_date = pd.to_datetime(results_df['date']).max().date().isoformat()
            output_file = f"{tkr}_{startDate.isoformat()}_{last_date}_values_partial.csv"
        except Exception:
            # Fallback to default partial name if we can't compute last_date
            output_file = f"{tkr}_{startDate.isoformat()}_{endDate.isoformat()}_values_partial.csv"

    # Write the results (partial or complete) to CSV
    try:
        results_df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Failed to write CSV to {output_file}: {e}")

    # Report status
    if errors:
        print(f"Finished with {len(errors)} error(s); partial results saved to {output_file}")
    else:
        print(f"All done — results saved to {output_file}")



    

    


