import pandas as pd
import numpy as np
import cryptocompare
# from datetime import datetime, timedelta
import datetime
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings('ignore')

print("Starting crypto trading strategy script...")

class DataFetcher:
    def __init__(self, api_key=None):
        """
        Initialize DataFetcher with optional CryptoCompare API key
        Set your API key using: cryptocompare.cryptocompare._set_api_key_parameter(your_key)
        """
        if api_key:
            cryptocompare.cryptocompare._set_api_key_parameter('8d83c0d2fa1d6ba508a0c512987c8340867a1f8f2403e73d033aa880a6341f2e')
    
    def fetch_crypto_data_15min(self, symbol, days_back=12, verbose=True):
        """
        Fetch 15-minute candle data using CryptoCompare API
        Note: CryptoCompare minute endpoint requires aggregation parameter for 15min intervals
        """
        # Extract base currency from symbol (e.g., 'BTCUSDT' -> 'BTC')
        base_currency = symbol.replace('USDT', '').replace('BUSD', '')
        quote_currency = 'USDT'
        
        # Calculate total minutes needed
        minutes_needed = days_back * 24 * 60
        # For 15-min intervals, divide by 15
        candles_needed = minutes_needed // 15
        
        # CryptoCompare limit per call (max 2000)
        limit_per_call = 2000
        all_data = []
        
        # Calculate how many API calls needed
        num_calls = (candles_needed // limit_per_call) + 1
        
        # Start from current time and go backwards
        to_timestamp = int(time.time())
        
        for call_num in range(num_calls):
            try:
                # Fetch minute data with aggregate=15 for 15-minute intervals
                data = cryptocompare.get_historical_price_minute(
                    base_currency, 
                    currency=quote_currency, 
                    limit=min(limit_per_call, candles_needed - len(all_data)),
                    toTs=to_timestamp,
                    exchange='binance'
                )
                
                if not data:
                    if verbose:
                        print(f"No data returned for {symbol} at timestamp {to_timestamp}")
                    break
                
                # Add to results
                all_data = data + all_data  # Prepend older data
                
                if verbose:
                    print(f"Fetched {len(data)} candles for {symbol}, total: {len(all_data)}")
                
                # Check if we got all needed data
                if len(all_data) >= candles_needed or len(data) < limit_per_call:
                    break
                
                # Update timestamp for next batch (go further back in time)
                to_timestamp = data[0]['time'] - 1
                
                # Rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                if verbose:
                    print(f"Error fetching {symbol} data: {e}")
                break
        
        if not all_data:
            if verbose:
                print(f"No data available from CryptoCompare for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Process the data
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('datetime', inplace=True)
        
        # Keep only necessary columns and rename to match your format
        # CryptoCompare uses 'volumeto' for quote volume
        df = df[['open', 'high', 'low', 'close', 'volumeto']].copy()
        df.rename(columns={'volumeto': 'volume'}, inplace=True)
        
        # Resample to 15-minute intervals if needed
        df = df.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Calculate token volume
        df['token_volume'] = df['volume'] / df['close']
        
        # Filter to exact days_back period
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_back)
        df = df[df.index >= cutoff_date]
        
        if verbose:
            print(f"Final dataset: {len(df)} rows for {symbol}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df

def calculate_relative_strength(data_dict, tickers, rs_length):
    """Calculate percentage gains for relative strength chart (like Pine Script)"""
    print(f"Calculating relative strength with {rs_length} period lookback...")
    try:
        # Find common date range across all cryptocurrencies
        common_start = max([data_dict[ticker].index[0] for ticker in tickers])
        common_end = min([data_dict[ticker].index[-1] for ticker in tickers])
        
        # Align all data to common timeframe
        aligned_data = {}
        for ticker in tickers:
            data = data_dict[ticker]
            # Filter to common date range
            data_filtered = data[(data.index >= common_start) & (data.index <= common_end)]
            aligned_data[ticker] = data_filtered
        
        # Calculate percentage gains: ((current - past) / past) * 100
        rs_data = pd.DataFrame(index=aligned_data[tickers[0]].index)
        
        for ticker in tickers:
            close_prices = aligned_data[ticker]['close']
            
            # Calculate past prices (shifted by rs_length)
            past_prices = close_prices.shift(rs_length)
            
            # Calculate percentage gain: ((current - past) / past) * 100
            pct_gain = ((close_prices - past_prices) / past_prices) * 100
            
            # Only show values after we have enough historical data
            pct_gain = pct_gain.where(close_prices.index >= (common_start + pd.Timedelta(minutes=15*rs_length)), np.nan)
            
            rs_data[f'{ticker}_gain'] = pct_gain
        
        print(f"RS Date range: {rs_data.index[0]} to {rs_data.index[-1]}")
        return rs_data
    except Exception as e:
        print(f"Error in calculate_relative_strength: {e}")
        return pd.DataFrame()

def checkhl(data_back, data_forward, hl):
    """Check for high/low pivot points"""
    try:
        if hl == 'high' or hl == 'High':
            ref = data_back[len(data_back)-1]
            for i in range(len(data_back)-1):
                if ref < data_back[i]:
                    return 0
            for i in range(len(data_forward)):
                if ref <= data_forward[i]:
                    return 0
            return 1
        if hl == 'low' or hl == 'Low':
            ref = data_back[len(data_back)-1]
            for i in range(len(data_back)-1):
                if ref > data_back[i]:
                    return 0
            for i in range(len(data_forward)):
                if ref >= data_forward[i]:
                    return 0
            return 1
    except Exception as e:
        print(f"Error in checkhl: {e}")
        return 0

def pivot(osc, LBL, LBR, highlow):
    """Find pivot points in a data series"""
    try:
        left = []
        right = []
        pivots = []
        for i in range(len(osc)):
            pivots.append(np.nan)
            if i < LBL + 1:
                left.append(osc[i])
            if i > LBL:
                right.append(osc[i])
            if i > LBL + LBR:
                left.append(right[0])
                left.pop(0)
                right.pop(0)
                if checkhl(left, right, highlow):
                    pivots[i - LBR] = osc[i - LBR]
        return pivots
    except Exception as e:
        print(f"Error in pivot function: {e}")
        return [np.nan] * len(osc)

def process_crypto_strategy(data, ticker, high_length, stock_ema_length):
    """Process trading strategy for a single cryptocurrency"""
    print(f"\nProcessing strategy for {ticker}...")
    try:
        # Calculate Pivots
        pivots_high = pivot(data['close'], high_length, high_length, 'high')
        pivots_low = pivot(data['close'], high_length, high_length, 'low')

        data_reset = data.reset_index()

        pivot_high_indexes = data_reset.index[data_reset['close'] == pivots_high].tolist()
        pivot_low_indexes = data_reset.index[data_reset['close'] == pivots_low].tolist()

        pivot_high_indexes.insert(0, 0)
        pivot_low_indexes.insert(0, 0)
        pivot_high_indexes.append(len(data_reset))
        pivot_low_indexes.append(len(data_reset))

        print(f"{ticker} - High pivots found: {len(pivot_high_indexes)-2}")
        print(f"{ticker} - Low pivots found: {len(pivot_low_indexes)-2}")
        
        # Calculate Anchored VWAP for Long Segments
        concatenated_segments = []
        for i in range(len(pivot_high_indexes) - 1):
            if i == 0:
                start_idx = pivot_high_indexes[i]
                end_idx = pivot_high_indexes[i + 1]-1
                segment_data = data_reset.loc[start_idx:end_idx].copy()
                segment_data['cum_vol'] = np.nan
                segment_data['cum_vol_price'] = np.nan
                segment_data['anchored_vwap_long'] = np.nan
                segment_data['anchor_pivot_high_idx'] = np.nan
            elif i == len(pivot_high_indexes) - 1:
                start_idx = pivot_high_indexes[i]-1
                end_idx = pivot_high_indexes[i + 1]
                segment_data = data_reset.loc[start_idx:end_idx].copy()
                segment_data['cum_vol'] = segment_data['volume'].cumsum()
                segment_data['cum_vol_price'] = segment_data['volume'] * segment_data['close']
                segment_data['cum_vol_price'] = segment_data['cum_vol_price'].cumsum()
                segment_data['anchored_vwap_long'] = segment_data['cum_vol_price'] / segment_data['cum_vol']
                segment_data['anchor_pivot_high_idx'] = start_idx
            else:
                start_idx = pivot_high_indexes[i]-1
                end_idx = pivot_high_indexes[i + 1]-1
                segment_data = data_reset.loc[start_idx:end_idx].copy()
                segment_data['cum_vol'] = segment_data['volume'].cumsum()
                segment_data['cum_vol_price'] = segment_data['volume'] * segment_data['close']
                segment_data['cum_vol_price'] = segment_data['cum_vol_price'].cumsum()
                segment_data['anchored_vwap_long'] = segment_data['cum_vol_price'] / segment_data['cum_vol']
                segment_data['anchor_pivot_high_idx'] = start_idx

            rolling_std = []
            for j in range(len(segment_data)):
                window_size = j + 1
                std = segment_data['close'].iloc[max(0, j - window_size + 1):j + 1].std()
                rolling_std.append(std)
            segment_data['anchored_vwap_plus_std'] = segment_data['anchored_vwap_long'] + (1.2 * np.array(rolling_std))
            concatenated_segments.append(segment_data)

        long_data = pd.concat(concatenated_segments)
        
        # Calculate Anchored VWAP for Short Segments
        concatenated_segments = []
        for i in range(len(pivot_low_indexes) - 1):
            if i == 0:
                start_idx = pivot_low_indexes[i]
                end_idx = pivot_low_indexes[i + 1]-1
                segment_data = data_reset.loc[start_idx:end_idx].copy()
                segment_data['cum_vol'] = np.nan
                segment_data['cum_vol_price'] = np.nan
                segment_data['anchored_vwap_short'] = np.nan
                segment_data['anchor_pivot_low_idx'] = np.nan
            elif i == len(pivot_low_indexes) - 1:
                start_idx = pivot_low_indexes[i]-1
                end_idx = pivot_low_indexes[i + 1]
                segment_data = data_reset.loc[start_idx:end_idx].copy()
                segment_data['cum_vol'] = segment_data['volume'].cumsum()
                segment_data['cum_vol_price'] = segment_data['volume'] * segment_data['close']
                segment_data['cum_vol_price'] = segment_data['cum_vol_price'].cumsum()
                segment_data['anchored_vwap_short'] = segment_data['cum_vol_price'] / segment_data['cum_vol']
                segment_data['anchor_pivot_low_idx'] = start_idx
            else:
                start_idx = pivot_low_indexes[i]-1
                end_idx = pivot_low_indexes[i + 1]-1
                segment_data = data_reset.loc[start_idx:end_idx].copy()
                segment_data['cum_vol'] = segment_data['volume'].cumsum()
                segment_data['cum_vol_price'] = segment_data['volume'] * segment_data['close']
                segment_data['cum_vol_price'] = segment_data['cum_vol_price'].cumsum()
                segment_data['anchored_vwap_short'] = segment_data['cum_vol_price'] / segment_data['cum_vol']
                segment_data['anchor_pivot_low_idx'] = start_idx

            rolling_std = []
            for j in range(len(segment_data)):
                window_size = j + 1
                std = segment_data['close'].iloc[max(0, j - window_size + 1):j + 1].std()
                rolling_std.append(std)
            segment_data['anchored_vwap_minus_std'] = segment_data['anchored_vwap_short'] - (1.2 * np.array(rolling_std))
            concatenated_segments.append(segment_data)

        short_data = pd.concat(concatenated_segments)
        
        # Merge Data and Add Indicators
        data = pd.merge(long_data, short_data[['datetime', 'anchored_vwap_short', 'anchored_vwap_minus_std', 'anchor_pivot_low_idx']], on='datetime', how='left')
        data = data.drop(columns=['cum_vol', 'cum_vol_price'])
        data = data.drop_duplicates(subset='datetime', keep='first')
        data = data.set_index('datetime')

        # Add EMA and VWMA
        data['200EMA'] = data['close'].ewm(span=stock_ema_length, adjust=False).mean()
        data['200EMA'].iloc[:stock_ema_length] = np.nan
        data['200VWMA'] = (
            data['close'].mul(data['token_volume']).rolling(window=stock_ema_length).sum() /
            data['token_volume'].rolling(window=stock_ema_length).sum()
        )
        data['200VWMA'].iloc[:stock_ema_length] = np.nan
        data['LTV'] = data['close'].iloc[-1]
        
        return data, pivot_high_indexes, pivot_low_indexes, pivots_high, pivots_low
    except Exception as e:
        print(f"Error in process_crypto_strategy for {ticker}: {e}")
        return None, [], [], [], []

def generate_signals(data, pivot_high_indexes, pivot_low_indexes, ticker, rs_data=None):
    """Generate strict buy/sell signals for a cryptocurrency"""
    print(f"Generating signals for {ticker}...")
    try:
        def get_anchor_pivot_idx(row, anchor_col):
            try:
                idx = int(row[anchor_col])
                return idx
            except:
                return None

        data_reset = data.reset_index()
        buy_indexes = []
        sell_indexes = []
        is_plotable = np.zeros(len(data), dtype=int)
        rs_data_used = rs_data if rs_data is not None else globals().get('rs_data', None)

        # --- STRICT BUY SIGNALS ---
        for idx in range(len(pivot_high_indexes)-1):
            pivot_idx = pivot_high_indexes[idx]
            next_pivot_idx = pivot_high_indexes[idx+1]
            if next_pivot_idx <= pivot_idx:
                continue
            segment = data_reset.loc[pivot_idx:next_pivot_idx-1].copy()
            if segment.empty:
                continue
            for i in range(pivot_idx+4, next_pivot_idx):
                above_both_ma = (data_reset.loc[i, 'close'] > data_reset.loc[i, '200EMA']) and (data_reset.loc[i, 'close'] > data_reset.loc[i, '200VWMA'])
                crossover_long = (
                    (data_reset.loc[i, 'close'] > data_reset.loc[i, 'anchored_vwap_plus_std']) and
                    (data_reset.loc[i, 'open'] < data_reset.loc[i, 'anchored_vwap_plus_std']) and
                    above_both_ma
                )
                cross_gap_long = (
                    (data_reset.loc[i, 'close'] > data_reset.loc[i, 'anchored_vwap_plus_std']) and
                    (data_reset.loc[i, 'open'] > data_reset.loc[i, 'anchored_vwap_plus_std']) and
                    (data_reset.loc[i-1, 'close'] < data_reset.loc[i, 'anchored_vwap_plus_std']) and
                    above_both_ma
                )
                anchor_high_idx = get_anchor_pivot_idx(data_reset.loc[i], 'anchor_pivot_high_idx')
                anchor_high_valid = False
                if anchor_high_idx is not None and not np.isnan(anchor_high_idx):
                    anchor_row = data_reset.loc[anchor_high_idx]
                    anchor_high_valid = (
                        anchor_row['close'] > anchor_row['200EMA'] and anchor_row['close'] > anchor_row['200VWMA']
                    )
                crossed_below = False
                for j in range(pivot_idx, i):
                    if (data_reset.loc[j, 'close'] < data_reset.loc[j, '200EMA']) or (data_reset.loc[j, 'close'] < data_reset.loc[j, '200VWMA']):
                        crossed_below = True
                        break
                # # Relative strength validation for BTCUSDT buy
                # rs_ok = True
                # if ticker == 'BTCUSDT':
                #     rs_ok = False
                #     current_time = data_reset.loc[i, 'datetime']
                #     if (rs_data_used is not None) and (current_time in rs_data_used.index):
                #         try:
                #             btc_gain = rs_data_used.loc[current_time, 'BTCUSDT_gain']
                #             eth_gain = rs_data_used.loc[current_time, 'ETHUSDT_gain']
                #             sol_gain = rs_data_used.loc[current_time, 'SOLUSDT_gain']
                #             rs_ok = (btc_gain >= eth_gain) and (btc_gain >= sol_gain)
                #         except Exception as e:
                #             print(f"Error in RS validation for BUY {ticker}: {e}")
                #             rs_ok = False

                # # Relative strength validation for BTCUSDT, ETHUSDT, and SOLUSDT
                # rs_ok = True
                # current_time = data_reset.loc[i, 'datetime']
    
                # if ticker == 'BTCUSDT':
                #     rs_ok = False
                #     if (rs_data_used is not None) and (current_time in rs_data_used.index):
                #         try:
                #             btc_gain = rs_data_used.loc[current_time, 'BTCUSDT_gain']
                #             eth_gain = rs_data_used.loc[current_time, 'ETHUSDT_gain']
                #             sol_gain = rs_data_used.loc[current_time, 'SOLUSDT_gain']
                #             rs_ok = (btc_gain >= eth_gain) and (btc_gain >= sol_gain)
                #         except Exception:
                #             rs_ok = False
    
                # elif ticker == 'ETHUSDT':
                #     rs_ok = False
                #     if (rs_data_used is not None) and (current_time in rs_data_used.index):
                #         try:
                #             btc_gain = rs_data_used.loc[current_time, 'BTCUSDT_gain']
                #             eth_gain = rs_data_used.loc[current_time, 'ETHUSDT_gain']
                #             sol_gain = rs_data_used.loc[current_time, 'SOLUSDT_gain']
                #             rs_ok = (eth_gain >= btc_gain) and (eth_gain >= sol_gain)
                #         except Exception:
                #             rs_ok = False
    
                # elif ticker == 'SOLUSDT':
                #     rs_ok = False
                #     if (rs_data_used is not None) and (current_time in rs_data_used.index):
                #         try:
                #             btc_gain = rs_data_used.loc[current_time, 'BTCUSDT_gain']
                #             eth_gain = rs_data_used.loc[current_time, 'ETHUSDT_gain']
                #             sol_gain = rs_data_used.loc[current_time, 'SOLUSDT_gain']
                #             rs_ok = (sol_gain >= btc_gain) and (sol_gain >= eth_gain)
                #         except Exception:
                #             rs_ok = False
                            
                # if (crossover_long or cross_gap_long) and anchor_high_valid and crossed_below and rs_ok:
                #     buy_indexes.append(data_reset.loc[i, 'datetime'])
                #     is_plotable[i] = 1
                #     break

                # Define ticker mapping for validation
                TRACKED_TICKERS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']

                # Relative strength validation
                rs_ok = True
                current_time = data_reset.loc[i, 'datetime']

                if ticker in TRACKED_TICKERS and rs_data_used is not None and current_time in rs_data_used.index:
                    try:
                        # Extract all gains at once
                        gains = {t: rs_data_used.loc[current_time, f'{t}_gain'] for t in TRACKED_TICKERS}
                        
                        # Check if current ticker has the highest gain
                        current_gain = gains[ticker]
                        rs_ok = all(current_gain >= gain for t, gain in gains.items() if t != ticker)
                    except Exception:
                        rs_ok = False
                elif ticker in TRACKED_TICKERS:
                    rs_ok = False

                if (crossover_long or cross_gap_long) and anchor_high_valid and crossed_below and rs_ok:
                    buy_indexes.append(data_reset.loc[i, 'datetime'])
                    is_plotable[i] = 1
                    break

        # --- STRICT SELL SIGNALS ---
        for idx in range(len(pivot_low_indexes)-1):
            pivot_idx = pivot_low_indexes[idx]
            next_pivot_idx = pivot_low_indexes[idx+1]
            if next_pivot_idx <= pivot_idx:
                continue
            segment = data_reset.loc[pivot_idx:next_pivot_idx-1].copy()
            if segment.empty:
                continue
            for i in range(pivot_idx+4, next_pivot_idx):
                below_both_ma = (data_reset.loc[i, 'close'] < data_reset.loc[i, '200EMA']) and (data_reset.loc[i, 'close'] < data_reset.loc[i, '200VWMA'])
                crossover_short = (
                    (data_reset.loc[i, 'close'] < data_reset.loc[i, 'anchored_vwap_minus_std']) and
                    (data_reset.loc[i, 'open'] > data_reset.loc[i, 'anchored_vwap_minus_std']) and
                    below_both_ma
                )
                cross_gap_short = (
                    (data_reset.loc[i, 'close'] < data_reset.loc[i, 'anchored_vwap_minus_std']) and
                    (data_reset.loc[i, 'open'] < data_reset.loc[i, 'anchored_vwap_minus_std']) and
                    (data_reset.loc[i-1, 'close'] > data_reset.loc[i, 'anchored_vwap_minus_std']) and
                    below_both_ma
                )
                anchor_low_idx = get_anchor_pivot_idx(data_reset.loc[i], 'anchor_pivot_low_idx')
                anchor_low_valid = False
                if anchor_low_idx is not None and not np.isnan(anchor_low_idx):
                    anchor_row = data_reset.loc[anchor_low_idx]
                    anchor_low_valid = (
                        anchor_row['close'] < anchor_row['200EMA'] and anchor_row['close'] < anchor_row['200VWMA']
                    )
                crossed_above = False
                for j in range(pivot_idx, i):
                    if (data_reset.loc[j, 'close'] > data_reset.loc[j, '200EMA']) or (data_reset.loc[j, 'close'] > data_reset.loc[j, '200VWMA']):
                        crossed_above = True
                        break
                # # Relative strength validation for BTCUSDT sell
                # rs_ok = True
                # if ticker == 'BTCUSDT':
                #     rs_ok = False
                #     current_time = data_reset.loc[i, 'datetime']
                #     if (rs_data_used is not None) and (current_time in rs_data_used.index):
                #         try:
                #             btc_gain = rs_data_used.loc[current_time, 'BTCUSDT_gain']
                #             eth_gain = rs_data_used.loc[current_time, 'ETHUSDT_gain']
                #             sol_gain = rs_data_used.loc[current_time, 'SOLUSDT_gain']
                #             rs_ok = (btc_gain <= eth_gain) and (btc_gain <= sol_gain)
                #         except Exception as e:
                #             print(f"Error in RS validation for SELL {ticker}: {e}")
                #             rs_ok = False

                # # Relative strength validation for BTCUSDT, ETHUSDT, and SOLUSDT
                # rs_ok = True
                # current_time = data_reset.loc[i, 'datetime']
    
                # if ticker == 'BTCUSDT':
                #     rs_ok = False
                #     if (rs_data_used is not None) and (current_time in rs_data_used.index):
                #         try:
                #             btc_gain = rs_data_used.loc[current_time, 'BTCUSDT_gain']
                #             eth_gain = rs_data_used.loc[current_time, 'ETHUSDT_gain']
                #             sol_gain = rs_data_used.loc[current_time, 'SOLUSDT_gain']
                #             rs_ok = (btc_gain <= eth_gain) and (btc_gain <= sol_gain)
                #         except Exception:
                #             rs_ok = False
    
                # elif ticker == 'ETHUSDT':
                #     rs_ok = False
                #     if (rs_data_used is not None) and (current_time in rs_data_used.index):
                #         try:
                #             btc_gain = rs_data_used.loc[current_time, 'BTCUSDT_gain']
                #             eth_gain = rs_data_used.loc[current_time, 'ETHUSDT_gain']
                #             sol_gain = rs_data_used.loc[current_time, 'SOLUSDT_gain']
                #             rs_ok = (eth_gain <= btc_gain) and (eth_gain <= sol_gain)
                #         except Exception:
                #             rs_ok = False
    
                # elif ticker == 'SOLUSDT':
                #     rs_ok = False
                #     if (rs_data_used is not None) and (current_time in rs_data_used.index):
                #         try:
                #             btc_gain = rs_data_used.loc[current_time, 'BTCUSDT_gain']
                #             eth_gain = rs_data_used.loc[current_time, 'ETHUSDT_gain']
                #             sol_gain = rs_data_used.loc[current_time, 'SOLUSDT_gain']
                #             rs_ok = (sol_gain <= btc_gain) and (sol_gain <= eth_gain)
                #         except Exception:
                #             rs_ok = False
                            
                # if (crossover_short or cross_gap_short) and anchor_low_valid and crossed_above and rs_ok:
                #     sell_indexes.append(data_reset.loc[i, 'datetime'])
                #     is_plotable[i] = 1
                #     break

                # Define tracked tickers
                TRACKED_TICKERS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']

                # Relative strength validation for short signals
                rs_ok = True
                current_time = data_reset.loc[i, 'datetime']

                if ticker in TRACKED_TICKERS and rs_data_used is not None and current_time in rs_data_used.index:
                    try:
                        # Extract all gains at once
                        gains = {t: rs_data_used.loc[current_time, f'{t}_gain'] for t in TRACKED_TICKERS}
                        
                        # Check if current ticker has the LOWEST gain (weakness for short signal)
                        current_gain = gains[ticker]
                        rs_ok = all(current_gain <= gain for t, gain in gains.items() if t != ticker)
                    except Exception:
                        rs_ok = False
                elif ticker in TRACKED_TICKERS:
                    rs_ok = False

                if (crossover_short or cross_gap_short) and anchor_low_valid and crossed_above and rs_ok:
                    sell_indexes.append(data_reset.loc[i, 'datetime'])
                    is_plotable[i] = 1
                    break

        print(f"{ticker} - Strict Buy signals: {len(buy_indexes)}")
        print(f"{ticker} - Strict Sell signals: {len(sell_indexes)}")
        
        return buy_indexes, sell_indexes, is_plotable
    except Exception as e:
        print(f"Error in generate_signals for {ticker}: {e}")
        return [], [], []

def create_signals_dataframe(tickers, signals_data, processed_data):
    """Create a consolidated DataFrame of all signals"""
    print("Creating consolidated signals dataframe...")
    try:
        signal_records = []

        for ticker in tickers:
            # Process buy signals
            for timestamp in signals_data[ticker]['buy_indexes']:
                try:
                    # Find the corresponding data row
                    data_row = processed_data[ticker].loc[timestamp]
                    
                    signal_record = {
                        'crypto': ticker,
                        'timestamp': timestamp,
                        'open': data_row['open'],
                        'high': data_row['high'],
                        'low': data_row['low'],
                        'close': data_row['close'],
                        'volume': data_row['volume'],
                        'signal_type': 'BUY'
                    }
                    signal_records.append(signal_record)
                except Exception as e:
                    print(f"Error processing buy signal for {ticker} at {timestamp}: {e}")
            
            # Process sell signals
            for timestamp in signals_data[ticker]['sell_indexes']:
                try:
                    # Find the corresponding data row
                    data_row = processed_data[ticker].loc[timestamp]
                    
                    signal_record = {
                        'crypto': ticker,
                        'timestamp': timestamp,
                        'open': data_row['open'],
                        'high': data_row['high'],
                        'low': data_row['low'],
                        'close': data_row['close'],
                        'volume': data_row['volume'],
                        'signal_type': 'SELL',
                    }
                    signal_records.append(signal_record)
                except Exception as e:
                    print(f"Error processing sell signal for {ticker} at {timestamp}: {e}")

        # Create DataFrame from records
        signals_df = pd.DataFrame(signal_records)

        # Sort by timestamp
        signals_df = signals_df.sort_values('timestamp')
        
        print(f"Created signals dataframe with {len(signals_df)} rows")
        return signals_df
    except Exception as e:
        print(f"Error in create_signals_dataframe: {e}")
        return pd.DataFrame()

def send_signal_email(signals_df):
    """Send email notification for recent signals (within last 20 minutes)"""
    print("Checking for recent signals to send email...")
    try:
        # Set current time
        # current_time = datetime.datetime(2025, 10, 1, 8, 36, 0)
        current_time = datetime.datetime.now()
        
        # Filter signals from the last 20 minutes
        time_threshold = current_time - datetime.timedelta(minutes=30)
        recent_signals = signals_df[pd.to_datetime(signals_df['timestamp']) > time_threshold]
        
        if len(recent_signals) > 0:
            print(f"Found {len(recent_signals)} recent signals, preparing email...")
            # Email configuration
            sender_email = "asusrog1650@gmail.com"  # Replace with your Gmail
            receiver_email = "asusrog1650@gmail.com"  # Replace with recipient email
            password = "fbcsuqwthwtjwgmw"  # Use an app password for Gmail
            
            # Create message
            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = receiver_email
            message["Subject"] = "Crypto Trading Signal Alert"
            
            # Create email body with signals
            body = "Recent Trading Signals:\n\n"
            
            for _, row in recent_signals.iterrows():
                # Add IST time (UTC+5:30)
                ist_time = pd.to_datetime(row['timestamp']) + datetime.timedelta(hours=5, minutes=30)
                body += f"Crypto: {row['crypto']}\n"
                body += f"Signal: {row['signal_type']}\n"
                body += f"Time (IST): {ist_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                body += f"Price: {row['close']}\n"
                body += "------------------------\n"
            
            message.attach(MIMEText(body, "plain"))
            
            try:
                # Connect to Gmail SMTP server
                server = smtplib.SMTP("smtp.gmail.com", 587)
                server.starttls()
                server.login(sender_email, password)
                
                # Send email
                text = message.as_string()
                server.sendmail(sender_email, receiver_email, text)
                print(f"Email alert sent for {len(recent_signals)} recent signals")
                
                # Close connection
                server.quit()
            except Exception as e:
                print(f"Error sending email: {e}")
        else:
            print("No recent signals in the last 20 minutes")
    except Exception as e:
        print(f"Error in send_signal_email: {e}")

def main():
    try:
        print("Starting main execution...")

        # Initialize the DataFetcher
        fetcher = DataFetcher()  # Optional: add your CryptoCompare API key

        # Parameters
        tickers = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT','BNBUSDT','XRPUSDT']
        high_length = 50
        stock_ema_length = 200
        days_back = 100  # Now actively used for fetching data
        rs_length = 100  # Length for relative strength calculation

        crypto_data = {}

        for ticker in tickers:
            print(f"\nFetching data for {ticker}...")
            try:
                data = fetcher.fetch_crypto_data_15min(ticker, days_back=days_back, verbose=True)
                
                if data.empty:
                    print(f"Warning: No data returned for {ticker}")
                    continue
                    
                crypto_data[ticker] = data
                print(f"{ticker} - Data shape: {data.shape}")
                print(f"{ticker} - Date range: {data.index[0]} to {data.index[-1]}")
                print(f"{ticker} - Columns: {list(data.columns)}")
                
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                continue

        print("\nAll cryptocurrency data fetched successfully!")

        # Calculate relative strength data
        rs_data = calculate_relative_strength(crypto_data, tickers, rs_length)

        # Process all cryptocurrencies
        processed_data = {}
        strategy_results = {}

        for ticker in tickers:
            result = process_crypto_strategy(crypto_data[ticker], ticker, high_length, stock_ema_length)
            if result[0] is not None:
                data, pivot_high_indexes, pivot_low_indexes, pivots_high, pivots_low = result
                processed_data[ticker] = data
                strategy_results[ticker] = {
                    'pivot_high_indexes': pivot_high_indexes,
                    'pivot_low_indexes': pivot_low_indexes,
                    'pivots_high': pivots_high,
                    'pivots_low': pivots_low
                }
            else:
                print(f"Failed to process strategy for {ticker}, skipping...")

        if len(processed_data) < len(tickers):
            print("Error: Could not process strategy for all tickers")
            return

        print("\nAll cryptocurrencies processed successfully!")

        # Generate signals for all cryptocurrencies
        signals_data = {}
        for ticker in tickers:
            buy_indexes, sell_indexes, is_plotable = generate_signals(
                processed_data[ticker], 
                strategy_results[ticker]['pivot_high_indexes'],
                strategy_results[ticker]['pivot_low_indexes'],
                ticker,
                rs_data
            )
            signals_data[ticker] = {
                'buy_indexes': buy_indexes,
                'sell_indexes': sell_indexes,
                'is_plotable': is_plotable
            }

        print("\nSignals generated for all cryptocurrencies!")

        # Create consolidated signals dataframe
        signals_df = create_signals_dataframe(tickers, signals_data, processed_data)
        
        # Send email for recent signals
        send_signal_email(signals_df)
        
        print("Script execution completed successfully!")

        # print(signals_df.tail(10))
        
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
