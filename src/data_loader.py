import yfinance as yf
import pandas as pd
import numpy as np

def fetch_sector_returns(start_date="2015-01-01", end_date="2023-12-31"):
    """
    Downloads historical data for the 11 GICS Sector SPDR ETFs 
    and calculates daily returns.
    """
    tickers = ["XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]
    
    print(f"Downloading data for {len(tickers)} sector ETFs...")
    
    # Download the raw data
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # Handle the yfinance version difference dynamically
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        prices = data['Close']
    
    # Calculate daily percentage change (returns) and drop the first row (NaN)
    returns = np.log(prices / prices.shift(1)).dropna()
    
    print("Data successfully downloaded and returns calculated!")
    return returns

def fetch_tbill_rates(start_date="2015-01-01", end_date="2023-12-31"):
    """
    Downloads the 3-month US Treasury Bill rate from FRED via yfinance (^IRX).
    Returns a monthly Series of decimal monthly rates (e.g. 0.0043 for one month).
    ^IRX is the 13-week T-bill yield, quoted as annualised %.
    """
    print("Downloading 3-month T-bill rates (^IRX from FRED)...")
    
    tbill_raw = yf.download("^IRX", start=start_date, end=end_date, progress=False)
    
    # ^IRX is quoted as annualised percentage (e.g. 5.25 means 5.25% p.a.)
    # Convert: annualised % → decimal → monthly
    # yfinance returns multi-level columns for single tickers — flatten to 1D Series
    if isinstance(tbill_raw.columns, pd.MultiIndex):
        tbill_prices = tbill_raw['Close']['^IRX']
    else:
        tbill_prices = tbill_raw['Close'] if 'Close' in tbill_raw.columns else tbill_raw['Adj Close']
    
    # Ensure it's a flat float Series
    tbill_prices = tbill_prices.squeeze().astype(float)
    
    tbill_annual = tbill_prices / 100                  # 5.25 → 0.0525
    tbill_monthly = (1 + tbill_annual) ** (1/12) - 1  # annualised → monthly

    tbill_monthly = tbill_monthly.resample('ME').last()
    tbill_monthly.name = 'TBill_Monthly'
    
    print(f"T-bill shape: {tbill_monthly.shape}, dtype: {tbill_monthly.dtype}")
    print(f"Sample:\n{tbill_monthly.head(3)}")
    print("T-bill data downloaded successfully.")
    return tbill_monthly

if __name__ == "__main__":
    test_returns = fetch_sector_returns(start_date="2023-01-01", end_date="2023-12-31")
    print("\nFirst 5 rows of our daily returns data:")
    print(test_returns.head())
