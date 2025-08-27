import yfinance as yf
import pandas as pd
from datetime import datetime

def download_msci_eafe_data():
    # MSCI EAFE Index ticker symbol
    ticker = "EFA"  # iShares MSCI EAFE ETF as proxy for MSCI EAFE Index
    
    # Define date range
    start_date = "1995-01-01"
    end_date = "2025-05-31"
    
    try:
        # Download data
        print(f"Downloading MSCI EAFE Index data from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print("No data downloaded. Please check the ticker symbol and date range.")
            return None
        
        # Calculate total return (using adjusted close for dividends reinvestment)
        total_return_data = data[['Adj Close']].copy()
        total_return_data.columns = ['Total_Return_Index']
        
        # Display info about the downloaded data
        print(f"Downloaded {len(total_return_data)} trading days of data")
        print(f"Data range: {total_return_data.index[0].strftime('%Y-%m-%d')} to {total_return_data.index[-1].strftime('%Y-%m-%d')}")
        print("\nFirst 5 rows:")
        print(total_return_data.head())
        print("\nLast 5 rows:")
        print(total_return_data.tail())
        
        # Save to CSV
        output_file = "msci_eafe_total_return.csv"
        total_return_data.to_csv(output_file)
        print(f"\nData saved to {output_file}")
        
        return total_return_data
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

if __name__ == "__main__":
    data = download_msci_eafe_data()