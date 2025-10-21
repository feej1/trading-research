import requests
import pandas as pd
import random
import string


def RandomKey(length: int) -> str:
    letters_and_digits = string.ascii_letters + string.digits
    key = ''.join(random.choice(letters_and_digits) for i in range(length))
    return key.upper()

def RandomIpAddress() -> str:
    return ".".join(str(random.randint(0, 255)) for _ in range(4))

def getStockVolatility(timeSeriesData: pd.DataFrame, date: str, sampleSize: int, period: int) -> float:
    # Calculate daily returns
    if 'daily_return' not in timeSeriesData.columns:
        timeSeriesData = addDailyReturns(timeSeriesData)

    # Use only the most recent 'sampleSize' days for volatility calculation
    filteredData = timeSeriesData[timeSeriesData.index < pd.to_datetime(date)].tail(sampleSize)

    # Calculate annualized volatility
    volatility = filteredData['daily_return'].std() * (period ** 0.5)  # Assuming 252 trading days in a year

    return volatility

def addDailyReturns(timeSeriesData: pd.DataFrame) -> pd.DataFrame:
    timeSeriesData['daily_return'] = timeSeriesData['4. close'].astype(float).pct_change()
    timeSeriesData['daily_return'] = timeSeriesData['daily_return'] * 100
    return timeSeriesData

def getStockTimeSeriesDaily(tkr: str, dataBefore: str = None, dataAfter: str = None) -> pd.DataFrame:
    options = {
        "function": "TIME_SERIES_DAILY",
        "symbol": tkr,
        "outputsize": "full"
    }

    data = makeApiRquest(options, dataKey="Time Series (Daily)")

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data).T  # Transpose to have dates as rows
    df.index = pd.to_datetime(df.index)  # Convert index to datetime
    df = df.sort_index()  # Sort by date

    print(f"Retrieved {len(df)} daily time series entries for {tkr}")

    if dataBefore is not None:
        df = df[df.index < dataBefore]
    if dataAfter is not None:
        df = df[df.index > dataAfter]

    return df

def makeApiRquest(options: dict, dataKey: str = "data") -> dict:

    retry = 5
    response = None
    while retry > 0:

        api_key = RandomKey(16)

        print(f'Making api request for {options["function"]}')

        url = f"https://www.alphavantage.co/query?apikey={api_key}"

        for key, value in options.items():
            url += f"&{key}={value}"

        headers = {'X-Forwarded-For': RandomIpAddress()} # used to bypass rate limiting
        response = requests.get(url, headers=headers)

        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        if api_key in response.text :
            print("Rate limit hit, retrying...")
            print(f"key: {api_key}, ip: {headers['X-Forwarded-For']}")
            print(f"Response: {response.text}")
            
            retry -= 1

        else: 
            # Deserialize the JSON response into a Python dictionary or list
            try:
                result = response.json()[dataKey]
                return result
            except Exception as e:
                print(f"Error parsing response. Data was {response.text}")
                break

def getStockSplits(tkr: str) -> pd.DataFrame:
    options = {
        "function": "SPLITS",
        "symbol": tkr,
    }

    data = makeApiRquest(options)

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data)

    print(f"Retrieved {len(df)} splits for {tkr}")

    df['effective_date'] = pd.to_datetime(df['effective_date'])
    df = df.sort_values(by='effective_date')

    return df

def getStockOptions(tkr: str, date: str, expireBefore: str = None, expireAfter: str = None, type: str = None) -> pd.DataFrame:


    options = {
        "function": "HISTORICAL_OPTIONS",
        "symbol": tkr,
        "date": date
    }

    data = makeApiRquest(options)

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data)

    if len(df) < 1:
        raise Exception("ERROR: Retreived 0 options")

    print(f"Retrieved {len(df)} options for {tkr} on {date}")

    df['expiration'] = pd.to_datetime(df['expiration'])

    if expireBefore is not None:
        df = df[df['expiration'] < expireBefore]
    if expireAfter is not None:
        df = df[df['expiration'] > expireAfter]
    if type is not None:
        df = df[df['type'] == type]
    
    return df


def getYieldRate(maturity = "3month", interval = "daily", yieldsBefore: str = None, yieldsAfter: str = None) -> pd.DataFrame:


    options = {
        "function": "TREASURY_YIELD",
        "interval": interval,
        "maturity": maturity
    }

    data = makeApiRquest(options)

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data)

    print(f"Retrieved {len(df)} yields for {maturity}")

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    if yieldsBefore is not None:
        df = df[df.index < yieldsBefore]
    if yieldsAfter is not None:
        df = df[df.index > yieldsAfter]
    
    return df

