import streamlit as st
import yfinance as yf
import pandas as pd
import urllib.request
import io
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
from streamlit.runtime.scriptrunner import get_script_run_ctx
import json
import os

PRESETS_FILE = "filter_presets.json"
PERSIST_PRESETS_KEY = "_presets_cache"

def load_presets():
    if PERSIST_PRESETS_KEY in st.session_state:
        return st.session_state[PERSIST_PRESETS_KEY]
    if not os.path.exists(PRESETS_FILE):
        return {}
    with open(PRESETS_FILE) as f:
        presets = json.load(f)
    st.session_state[PERSIST_PRESETS_KEY] = presets
    return presets

def save_presets(presets):
    with open(PRESETS_FILE, "w") as f:
        json.dump(presets, f, indent=2)
    st.session_state[PERSIST_PRESETS_KEY] = presets

# =====================================================================
# CACHED DATA FETCHING LAYER
# =====================================================================

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CONTEXT = ssl._create_unverified_context()

def get_sp500_tickers():
    """Acquires S&P 500 companies list with ticker, name, and sector from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36'
    }
    
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, context=_SSL_CONTEXT) as response:
        html_content = response.read().decode('utf-8')
    
    html_stream = io.StringIO(html_content)
    table = pd.read_html(html_stream)
    df = table[0] 
    
    sp500_data = {}
    for _, row in df.iterrows():
        clean_ticker = str(row['Symbol']).replace('.', '-')
        sp500_data[clean_ticker] = {
            "name": str(row['Security']) if pd.notna(row.get('Security')) else "Unknown",
            "sector": str(row['GICS Sector']) if pd.notna(row.get('GICS Sector')) else "Unknown"
        }
        
    return sp500_data


@st.cache_data(ttl=3600)  # Cache bulk pricing data for 1 hour
def download_technical_data(tickers):
    """Downloads 1-year daily historical pricing timelines for all tickers."""
    price_df = yf.download(tickers, period="1y", group_by='ticker', progress=False, threads=True)
    if not isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = pd.MultiIndex.from_product([price_df.columns, ['']])
    return price_df


@st.cache_data(ttl=86400)  # Cache financials statements for 24 hours
def get_cached_financials(ticker):
    """Downloads yearly financial statements for a specific ticker."""
    ticker_obj = yf.Ticker(ticker)
    return ticker_obj.get_financials(freq="yearly")


# =====================================================================
# CORE WORKER & SCREENING LOGIC
# =====================================================================

def process_ticker_fundamentals(
    ticker, current_price, smas, ath_price, ath_needed_pct,
    atr_pct,
    fund_mode, fund_rate_target, fund_years_target, company_name, sector
):
    """Worker function focusing strictly on fundamental statements validation for pre-filtered candidates."""
    try:
        if fund_mode != "Disabled":
            financials_df = get_cached_financials(ticker)
            if financials_df.empty:
                return None

            # Optimization: Direct index contains check instead of list comprehension
            keyword = "Revenue" if fund_mode == "Revenue Growth" else "Net Income"
            matching_rows = financials_df.index[financials_df.index.str.contains(keyword, na=False)]
            
            if matching_rows.empty:
                return None
            
            # Fetch row and calculate growth YoY
            statement_series = financials_df.loc[matching_rows[0]].dropna().iloc[::-1]
            if len(statement_series) < fund_years_target + 1:
                return None

            pct_changes = statement_series.pct_change().dropna() * 100
            target_years_checked = pct_changes.tail(fund_years_target)

            if not (target_years_checked >= fund_rate_target).all():
                return None

        return {
            "ticker": ticker, 
            "name": company_name, 
            "sector": sector,
            "price": current_price,
            "sma50": smas[50], 
            "sma100": smas[100], 
            "sma150": smas[150], 
            "sma200": smas[200],
            "ath": ath_price, 
            "ath_needed": ath_needed_pct,
            "atr_pct": atr_pct
        }
    except Exception:
        import sys
        print(f"[WARN] Fundamental processing failed for {ticker}", file=sys.stderr)
        return None


def run_scanner(
    tickers,
    tickers_dict,
    sma_selections,
    alignment_enabled,
    min_ath_filter,
    max_ath_filter,
    growth_sma_period,
    is_above_mode,
    min_sma_growth_filter,
    max_sma_growth_filter,
    min_atr_filter,
    max_atr_filter,
    fund_mode,
    fund_rate_target,
    fund_years_target,
    sma_slope,
    sma_slope_period
):
    """Core scanning orchestrator applying vectorized technical masks followed by multithreaded fundamentals checks."""
    # Step 1: Bulk download technical pricing timelines
    price_df = download_technical_data(tickers)
    
    if price_df.empty:
        raise Exception("No market data acquired. Check connection.")

    # Vectorized SMA and Price extraction
    close_prices = price_df.xs('Close', level=1, axis=1)
    high_prices = price_df.xs('High', level=1, axis=1)
    low_prices = price_df.xs('Low', level=1, axis=1)
    current_prices = close_prices.iloc[-1]
    ath_prices = close_prices.max()
    sma_periods = [50, 100, 150, 200]
    smas_matrix = {p: close_prices.tail(p).mean() for p in sma_periods}
    
    # Vectorized ATR% computation (14-period)
    prev_close = close_prices.shift(1)
    hl = (high_prices - low_prices).abs()
    hpc = (high_prices - prev_close).abs()
    lpc = (low_prices - prev_close).abs()
    tr = hl.where(hl >= hpc, hpc)
    tr = tr.where(tr >= lpc, lpc)
    atr = tr.tail(14).mean()
    atr_pct_vals = (atr / current_prices) * 100
    
    # Vectorized return columns (1D, 5D, 1M, 6M, YTD, 1Y)
    dates = close_prices.index
    current_year = dates[-1].year
    prev_year_mask = dates.year < current_year
    ret_1d = safe_ret(close_prices, 1)
    ret_5d = safe_ret(close_prices, 5)
    ret_1m = safe_ret(close_prices, 21)
    ret_6m = safe_ret(close_prices, 126)
    ret_1y = safe_ret(close_prices, 252)
    if prev_year_mask.any():
        ytd_start = close_prices.loc[prev_year_mask].iloc[-1]
        ret_ytd = ((close_prices.iloc[-1] - ytd_start) / ytd_start) * 100
    else:
        ret_ytd = safe_ret(close_prices, len(close_prices) - 1)
    
    # --- Vectorized Technical Filtering Stage ---
    passed_mask = current_prices.notna() & ath_prices.notna()
    
    # 1. Price Above SMA filters
    for p in sma_periods:
        if sma_selections[p]:
            passed_mask &= (current_prices > smas_matrix[p])
    
    # 2. SMA Alignment filter (50 > 100 > 150 > 200)
    if alignment_enabled:
        passed_mask &= (smas_matrix[50] > smas_matrix[100])
        passed_mask &= (smas_matrix[100] > smas_matrix[150])
        passed_mask &= (smas_matrix[150] > smas_matrix[200])
    
    # 3. ATH Distance filters
    ath_dist_pct = ((ath_prices - current_prices) / current_prices) * 100
    if min_ath_filter is not None:
        passed_mask &= (ath_dist_pct >= min_ath_filter)
    if max_ath_filter is not None:
        passed_mask &= (ath_dist_pct <= max_ath_filter)
    
    # 4. Target SMA Position filters
    target_sma_vals = smas_matrix[growth_sma_period]
    sma_pos_pct = ((current_prices - target_sma_vals) / target_sma_vals) * 100
    has_growth_limit = min_sma_growth_filter is not None or max_sma_growth_filter is not None
    if has_growth_limit:
        if is_above_mode:
            passed_mask &= (current_prices > target_sma_vals)
        else:
            passed_mask &= (current_prices < target_sma_vals)
        if min_sma_growth_filter is not None:
            passed_mask &= (sma_pos_pct >= min_sma_growth_filter)
        if max_sma_growth_filter is not None:
            passed_mask &= (sma_pos_pct <= max_sma_growth_filter)
    
    # 5. ATR% Volatility filters
    if min_atr_filter is not None:
        passed_mask &= (atr_pct_vals >= min_atr_filter)
    if max_atr_filter is not None:
        passed_mask &= (atr_pct_vals <= max_atr_filter)
    
    # 6. SMA Slope filter (Rising/Falling)
    if sma_slope != "Disabled":
        slope_sma = smas_matrix[sma_slope_period]
        sma_5ago = close_prices.tail(sma_slope_period + 5).head(sma_slope_period).mean()
        if sma_slope == "Rising":
            passed_mask &= (slope_sma > sma_5ago)
        else:
            passed_mask &= (slope_sma < sma_5ago)

    # Identify candidates that passed ALL technical filters
    candidates = passed_mask[passed_mask].index.tolist()
    
    # Prepare data for those candidates
    pre_calculated_data = []
    for ticker in candidates:
        try:
            s_dict = {p: float(smas_matrix[p][ticker]) for p in sma_periods}
            ticker_info = tickers_dict.get(ticker, {})
            pre_calculated_data.append((
                ticker, float(current_prices[ticker]), s_dict, 
                float(ath_prices[ticker]), float(ath_dist_pct[ticker]),
                float(atr_pct_vals[ticker]),
                ticker_info.get("name", "Unknown"),
                ticker_info.get("sector", "Unknown"),
                float(ret_1d[ticker]), float(ret_5d[ticker]),
                float(ret_1m[ticker]), float(ret_6m[ticker]),
                float(ret_ytd[ticker]), float(ret_1y[ticker]),
            ))
        except Exception: 
            continue

    # Step 2: High-concurrency processing for candidates
    matches = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        for data in pre_calculated_data:
            t, p, s, ath, ath_n, atr_v, name, sector, r1d, r5d, r1m, r6m, rytd, r1y = data
            fut = executor.submit(
                process_ticker_fundamentals, 
                t, p, s, ath, ath_n, atr_v,
                fund_mode, fund_rate_target, fund_years_target, 
                name, sector
            )
            futures.append((fut, r1d, r5d, r1m, r6m, rytd, r1y))
        for fut, r1d, r5d, r1m, r6m, rytd, r1y in futures:
            res = fut.result()
            if res is not None:
                res.update({"1D": r1d, "5D": r5d, "1M": r1m, "6M": r6m, "YTD": rytd, "1Y": r1y})
                matches.append(res)

    raw_results = sorted(matches, key=lambda x: x['ticker'])
    return raw_results, len(pre_calculated_data)


# =====================================================================
# UI HELPER FUNCTIONS
# =====================================================================

def parse_numeric_filter(val, field_name):
    """Validates and parses numeric inputs, returning None if empty."""
    if not val or not val.strip():
        return None
    try:
        return float(val.strip())
    except ValueError:
        raise ValueError(f"Invalid entry for '{field_name}': '{val}'")


def build_numeric_display_df(raw_results, display_mode):
    if not raw_results:
        return pd.DataFrame()
    
    rows = []
    is_percentage_mode = (display_mode == "Percentage Distance (%)")
    
    for item in raw_results:
        ticker = item.get("ticker")
        if not ticker or (isinstance(ticker, float) and pd.isna(ticker)):
            continue
        price = item.get("price")
        name = item.get("name")
        ath = item.get("ath")
        ath_dist = item.get("ath_needed")
        if price is None or name is None or ath is None or ath_dist is None:
            continue
        sector = item.get("sector", "Unknown")
        atr_pct = item.get("atr_pct", None)
        ret_1d = item.get("1D", None)
        ret_5d = item.get("5D", None)
        ret_1m = item.get("1M", None)
        ret_6m = item.get("6M", None)
        ret_ytd = item.get("YTD", None)
        ret_1y = item.get("1Y", None)
        
        row = {
            "Ticker": ticker,
            "Company Name": name,
            "Sector": sector,
            "Price": price,
        }
        
        if atr_pct is not None:
            row["ATR%"] = atr_pct
        
        for label, val in [("1D", ret_1d), ("5D", ret_5d), ("1M", ret_1m),
                           ("6M", ret_6m), ("YTD", ret_ytd), ("1Y", ret_1y)]:
            if val is not None:
                row[label] = val
        
        for p in [50, 100, 150, 200]:
            sma_val = item[f"sma{p}"]
            if is_percentage_mode:
                pct_dist = ((price - sma_val) / sma_val) * 100
                row[f"{p} SMA Dist"] = pct_dist
            else:
                row[f"{p} SMA"] = sma_val
                
        if is_percentage_mode:
            row["ATH Dist"] = ath_dist
        else:
            row["ATH"] = ath
        
        rows.append(row)
        
    return pd.DataFrame(rows)


LOGICAL_COLUMNS = [
    "Ticker", "Company Name", "Sector", "Price", "ATR%", "1D", "5D", "1M", "6M", "YTD", "1Y",
    "SMA50", "SMA100", "SMA150", "SMA200", "ATH"
]

def resolve_visible_columns(selected, is_pct_mode):
    suffix = " SMA Dist" if is_pct_mode else " SMA"
    mapping = {
        "Ticker": ["Ticker"], "Company Name": ["Company Name"], "Sector": ["Sector"],
        "Price": ["Price"], "ATR%": ["ATR%"],
        "1D": ["1D"], "5D": ["5D"], "1M": ["1M"], "6M": ["6M"], "YTD": ["YTD"], "1Y": ["1Y"],
        "SMA50": [f"50{suffix}"], "SMA100": [f"100{suffix}"],
        "SMA150": [f"150{suffix}"], "SMA200": [f"200{suffix}"],
    }
    if is_pct_mode:
        mapping["ATH"] = ["ATH Dist"]
    else:
        mapping["ATH"] = ["ATH"]
    result = []
    for col in selected:
        result.extend(mapping.get(col, [col]))
    return result


def safe_ret(close_prices, lag):
    n = len(close_prices)
    if n < 2:
        return 0.0
    lag = min(lag, n - 1)
    return ((close_prices.iloc[-1] - close_prices.iloc[-1 - lag]) / close_prices.iloc[-1 - lag]) * 100


# =====================================================================
# MAIN STREAMLIT APPLICATION FUNCTION
# =====================================================================

def run_streamlit_app():
    """Builds and displays the core Streamlit interface."""
    st.set_page_config(
        page_title="S&P 500 Stock Scanner",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS Injector for Dark, High-Intensity Compact Aesthetic
    st.markdown("""
    <style>
        /* Main application container background */
        .stApp {
            background-color: #131722 !important;
            color: #d1d4dc !important;
        }
        
        /* Tight compact layout padding adjustment */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 0.5rem !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            max-width: 100% !important;
        }
        
        /* Reduce all vertical gaps */
        div[data-testid="stVerticalBlock"] > div {
            margin-bottom: 0.25rem !important;
        }
        hr {
            margin-top: 0.3rem !important;
            margin-bottom: 0.3rem !important;
        }
        
        /* Compact scan button in top-right */
        div.row-widget.stButton > button {
            font-size: 12px !important;
            padding: 0.2rem 0.6rem !important;
            width: auto !important;
            min-width: unset !important;
        }
        
        /* Compact expander headers */
        details > summary {
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            padding: 0.15rem 0.3rem !important;
        }
        .streamlit-expanderHeader {
            font-size: 0.8rem !important;
            padding: 0.15rem 0.3rem !important;
        }
        .streamlit-expanderContent {
            padding: 0.2rem 0.1rem !important;
        }
        div[data-testid="stExpander"] {
            border: none !important;
        }
        
        /* Compact checkboxes and inputs inside expanders */
        div[data-testid="stExpander"] label {
            font-size: 0.75rem !important;
        }
        div[data-testid="stExpander"] input {
            font-size: 0.75rem !important;
        }
        div[data-testid="stExpander"] div[data-baseweb="select"] > div {
            min-height: 1.5rem !important;
        }
        
        /* Custom typography and brand coloring */
        h1, h2, h3, h4, h5, h6 {
            color: #2962ff !important;
            font-family: 'Inter', sans-serif !important;
            margin-top: 0.2rem !important;
            margin-bottom: 0.2rem !important;
            font-weight: 700 !important;
        }
        
        /* Input and selectboxes styled cards */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            background-color: #2a2e39 !important;
            border-color: #363a45 !important;
            color: #d1d4dc !important;
        }
        
        input {
            color: #d1d4dc !important;
        }
        
        /* High-Visibility Run Scan Button */
        .stButton > button {
            background-color: #089981 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 4px !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: bold !important;
            font-size: 14px !important;
            width: 100% !important;
            transition: background-color 0.2s ease;
        }
        .stButton > button:hover {
            background-color: #06c2a2 !important;
            color: #ffffff !important;
            border: none !important;
        }
        
        /* Metric values formatting */
        div[data-testid="stMetricValue"] {
            font-size: 24px !important;
            color: #089981 !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #787b86 !important;
            font-size: 12px !important;
        }
        
        /* Compact margins between components */
        .element-container {
            margin-bottom: 0.4rem !important;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #131722;
        }
        ::-webkit-scrollbar-thumb {
            background: #363a45;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #2962ff;
        }
        button[id^="preset_box_"] {
            font-size: 0.75rem !important;
            padding: 0.15rem 0.4rem !important;
            min-height: unset !important;
            line-height: 1.3 !important;
        }
        button[kind="primary"] {
            background: #1a3a6a !important;
            border-color: #2962ff !important;
        }
        div[data-testid="stHorizontalBlock"] button[kind="primary"] {
            background: #f23645 !important; color: white !important;
            border: none !important; border-radius: 6px !important; font-size: 0.85rem !important;
            height: 32px !important; line-height: 1 !important;
        }
        div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            background: #089981 !important; color: white !important;
            border: none !important; border-radius: 6px !important; font-size: 0.85rem !important;
            height: 32px !important; line-height: 1 !important;
        }
        div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            background: #089981 !important; color: white !important;
            border: none !important;
        }
        /* Secondary Neutral Button */
        div.stButton:nth-child(2) > button {
            background-color: #2a2e39 !important;
            color: #d1d4dc !important;
            border: 1px solid #363a45 !important;
        }
        div.stButton:nth-child(2) > button:hover {
            background-color: #363a45 !important;
            color: #ffffff !important;
        }
        div[data-testid="stPopoverBody"] div[data-testid="stHorizontalBlock"] button[kind="primary"] {
            background: #1a3a6a !important; border-color: #2962ff !important;
            border-radius: initial !important; font-size: initial !important;
        }
        div[data-testid="stPopoverBody"] div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
            background: initial !important; color: initial !important;
            border-radius: initial !important; font-size: initial !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize Core Application State
    if "sp500_data" not in st.session_state or "sp500_tickers" not in st.session_state:
        try:
            sp500_dict = get_sp500_tickers()
            st.session_state.sp500_data = sp500_dict
            st.session_state.sp500_tickers = list(sp500_dict.keys())
        except Exception as e:
            st.error(f"Failed to load S&P 500 Stock Tickers list: {e}")
            st.stop()
    
    if "visible_columns" not in st.session_state:
        st.session_state.visible_columns = LOGICAL_COLUMNS[:]

    # Initialize state before UI blocks
    can_run_scan = True
    validation_error = None

    # Show deferred toast before any widgets render
    toast_msg = st.session_state.pop("_preset_toast", None)
    if toast_msg:
        st.toast(toast_msg)

    # Apply pending reset before any widgets render
    if st.session_state.pop("_reset_filters", False):
        st.session_state.sma50_chk = False
        st.session_state.sma100_chk = False
        st.session_state.sma150_chk = False
        st.session_state.sma200_chk = False
        st.session_state.alignment_chk = False
        st.session_state.min_atr_pct = ""
        st.session_state.max_atr_pct = ""
        st.session_state.growth_target_sma = "50"
        st.session_state.sma_direction = "Above (Price > SMA)"
        st.session_state.min_sma_growth = ""
        st.session_state.max_sma_growth = ""
        st.session_state.sma_slope = "Disabled"
        st.session_state.sma_slope_period = "50"
        st.session_state.fund_mode = "Disabled"
        st.session_state.fund_rate = ""
        st.session_state.fund_years = "1"
        st.session_state.min_ath = ""
        st.session_state.max_ath = ""
        st.session_state.display_mode = "Absolute Prices ($)"
        st.session_state._load_preset_name = None
        st.session_state._active_preset = ""
        st.session_state.scan_complete = False
        st.session_state.pop("raw_results", None)
    
    # Apply pending preset load before any widgets render
    if "_load_preset_name" in st.session_state and st.session_state._load_preset_name:
        preset_name = st.session_state._load_preset_name
        st.session_state._load_preset_name = None
        presets = load_presets()
        if preset_name in presets:
            for k, v in presets[preset_name].items():
                if not k.startswith("_"):
                    try:
                        st.session_state[k] = v
                    except Exception:
                        pass

    # Auto-load favorite preset on first run
    if "_favorite_loaded" not in st.session_state:
        st.session_state._favorite_loaded = True
        presets = load_presets()
        fav = presets.get("_favorite")
        if fav and fav in presets:
            for k, v in presets[fav].items():
                if not k.startswith("_"):
                    try:
                        st.session_state[k] = v
                    except Exception:
                        pass
            st.session_state._active_preset = fav
    
    # --- TITLE ---
    st.markdown(
        "<h2 style='text-align: left; margin-bottom: 0; color: #2962ff;'>Trendik</h2>"
        "<p style='text-align: left; font-size: 0.85rem; color: #787b86; margin-top: 0;'>Find the perfect stocks for you.</p>",
        unsafe_allow_html=True
    )

    # --- HORIZONTAL FILTER PANE ---
    filters_expanded = not st.session_state.get("filters_minimized", False)
    
    # --- PRESETS ---
    all_presets = load_presets()
    preset_names = [k for k in all_presets if not k.startswith("_")]
    fav = all_presets.get("_favorite", "")
    active = st.session_state.get("_active_preset", "")
    
    preshow, prespacer = st.columns([1, 5])
    with preshow:
        with st.popover("💾 Presets", use_container_width=False):
            presets = load_presets()
            pcol1, pcol2 = st.columns([2, 1])
            with pcol1:
                preset_name = st.text_input("Preset name", key="preset_name", label_visibility="collapsed", placeholder="Name your preset...")
            with pcol2:
                if st.button("💾 Save", use_container_width=True):
                    if preset_name.strip():
                        presets[preset_name.strip()] = {
                            "sma50_chk": st.session_state.sma50_chk,
                            "sma100_chk": st.session_state.sma100_chk,
                            "sma150_chk": st.session_state.sma150_chk,
                            "sma200_chk": st.session_state.sma200_chk,
                            "alignment_chk": st.session_state.alignment_chk,
                            "min_atr_pct": st.session_state.min_atr_pct,
                            "max_atr_pct": st.session_state.max_atr_pct,
                            "growth_target_sma": st.session_state.growth_target_sma,
                            "sma_direction": st.session_state.sma_direction,
                            "min_sma_growth": st.session_state.min_sma_growth,
                            "max_sma_growth": st.session_state.max_sma_growth,
                            "fund_mode": st.session_state.fund_mode,
                            "fund_rate": st.session_state.fund_rate,
                            "fund_years": st.session_state.fund_years,
                            "min_ath": st.session_state.min_ath,
                            "max_ath": st.session_state.max_ath,
                            "display_mode": st.session_state.display_mode,
                            "sma_slope": st.session_state.get("sma_slope", "Disabled"),
                            "sma_slope_period": st.session_state.get("sma_slope_period", "50"),
                            "visible_columns": st.session_state.visible_columns,
                        }
                        save_presets(presets)
                        st.session_state._preset_toast = f"✅ Saved '{preset_name.strip()}'"
                        st.rerun()

    if preset_names:
        n = len(preset_names)
        if n < 7:
            cols = [1] * n + [7 - n]
        else:
            cols = [1] * n
        pcols = st.columns(cols)
        for i, pname in enumerate(preset_names):
            with pcols[i]:
                is_selected = (pname == active)
                is_fav = (pname == fav)
                star = "⭐ " if is_fav else "☆ "
                label = star + pname
                if st.button(label, use_container_width=True, key=f"preset_box_{pname}", type="primary" if is_selected else "secondary"):
                    if not is_selected:
                        st.session_state._load_preset_name = pname
                        st.session_state._active_preset = pname
                    else:
                        st.session_state._active_preset = ""
                    st.rerun()

        if active and active in all_presets:
            is_fav = (active == fav)
            acol1, acol2, acol3, acol4 = st.columns([1, 1, 1, 4])
            with acol1:
                if st.button("☆ Set Favorite" if not is_fav else "⭐ Unfavorite", use_container_width=True, key="set_fav_btn"):
                    presets = load_presets()
                    presets["_favorite"] = active if not is_fav else ""
                    save_presets(presets)
                    st.rerun()
            with acol2:
                if st.button("🔄 Update", use_container_width=True, key="update_btn"):
                    presets = load_presets()
                    presets[active] = {
                        "sma50_chk": st.session_state.sma50_chk,
                        "sma100_chk": st.session_state.sma100_chk,
                        "sma150_chk": st.session_state.sma150_chk,
                        "sma200_chk": st.session_state.sma200_chk,
                        "alignment_chk": st.session_state.alignment_chk,
                        "min_atr_pct": st.session_state.min_atr_pct,
                        "max_atr_pct": st.session_state.max_atr_pct,
                        "growth_target_sma": st.session_state.growth_target_sma,
                        "sma_direction": st.session_state.sma_direction,
                        "min_sma_growth": st.session_state.min_sma_growth,
                        "max_sma_growth": st.session_state.max_sma_growth,
                        "fund_mode": st.session_state.fund_mode,
                        "fund_rate": st.session_state.fund_rate,
                        "fund_years": st.session_state.fund_years,
                        "min_ath": st.session_state.min_ath,
                        "max_ath": st.session_state.max_ath,
                        "display_mode": st.session_state.display_mode,
                        "sma_slope": st.session_state.get("sma_slope", "Disabled"),
                        "sma_slope_period": st.session_state.get("sma_slope_period", "50"),
                        "visible_columns": st.session_state.visible_columns,
                    }
                    save_presets(presets)
                    st.session_state._preset_toast = f"✅ Preset '{active}' updated"
                    st.rerun()
            with acol3:
                if st.button("🗑️ Delete", use_container_width=True, key="delete_btn"):
                    presets = load_presets()
                    del presets[active]
                    if active == fav:
                        presets["_favorite"] = ""
                    st.session_state._active_preset = ""
                    save_presets(presets)
                    st.rerun()
    
    col_t, col_r, col_f, col_a = st.columns(4)

    with col_t:
        with st.expander("📈 Technical", expanded=filters_expanded):
            sma50_chk = st.checkbox(
                "Above 50-Day SMA", value=True, key="sma50_chk",
                help="Turn on to only show stocks currently trading above their 50-day average — catches short-term momentum."
            )
            sma100_chk = st.checkbox(
                "Above 100-Day SMA", value=False, key="sma100_chk",
                help="Add the 100-day filter to confirm the stock has medium-term bullish strength behind it."
            )
            sma150_chk = st.checkbox(
                "Above 150-Day SMA", value=False, key="sma150_chk",
                help="Enables a mid-to-long-term trend filter. Best for investors looking for sustained upward moves."
            )
            sma200_chk = st.checkbox(
                "Above 200-Day SMA", value=False, key="sma200_chk",
                help="The classic bull-market indicator. Stocks above their 200-day average are considered in a long-term uptrend."
            )
            alignment_chk = st.checkbox(
                "Bullish Alignment (50>100>150>200)", value=False, key="alignment_chk",
                help="Turn on to require all four SMAs to stack in order (50 over 100 over 150 over 200). This is known as a 'perfect order' and signals strong momentum across every timeframe."
            )
            st.markdown("**Volatility**")
            acol1, acol2 = st.columns(2)
            with acol1:
                min_atr_pct = st.text_input(
                    "Min ATR%", value="", key="min_atr_pct", placeholder="e.g. 0.5",
                    help="Minimum Average True Range as a % of price. Higher values mean more volatile stocks. For example, '2' means the stock must have at least 2% daily volatility."
                )
            with acol2:
                max_atr_pct = st.text_input(
                    "Max ATR%", value="", key="max_atr_pct", placeholder="e.g. 5",
                    help="Maximum Average True Range as a % of price. For example, '5' caps volatility to at most 5%. Leave blank for no limit."
                )

    with col_r:
        with st.expander("📐 Relative Position", expanded=filters_expanded):
            growth_target_sma = st.selectbox(
                "Target SMA",
                ["50", "100", "150", "200"],
                index=0,
                format_func=lambda x: f"{x}-Day SMA",
                label_visibility="collapsed",
                key="growth_target_sma",
                help="Pick which Simple Moving Average (SMA) to measure distance from: 50-Day (short-term), 100-Day (medium), 150-Day (mid-long), or 200-Day (long-term)."
            )
            sma_direction = st.selectbox(
                "Position", ["Above (Price > SMA)", "Below (Price < SMA)"], index=0,
                label_visibility="collapsed", key="sma_direction",
                help="'Above' finds stocks trading over the SMA (bullish). 'Below' finds stocks under it (could mean a pullback or discount)."
            )
            min_sma_growth = st.text_input(
                "Min %", value="", key="min_sma_growth", placeholder="e.g. 0",
                help="Set a floor. For example, '2' means the stock must be at least 2% above (or below) the target SMA."
            )
            max_sma_growth = st.text_input(
                "Max %", value="", key="max_sma_growth", placeholder="e.g. 10",
                help="Set a ceiling. For example, '10' caps results to stocks no more than 10% away from the SMA. Leave blank for no limit."
            )
            sma_slope = st.selectbox(
                "SMA Slope", ["Disabled", "Rising", "Falling"], index=0, key="sma_slope",
                help="'Rising' requires the target SMA to be higher than 5 days ago (uptrend). 'Falling' requires it to be lower (downtrend)."
            )
            sma_slope_period = st.selectbox(
                "Slope SMA", ["50", "100", "150", "200"], index=0,
                format_func=lambda x: f"{x}-Day SMA", label_visibility="collapsed",
                key="sma_slope_period"
            )

    with col_f:
        with st.expander("💰 Fundamentals", expanded=filters_expanded):
            fund_mode = st.selectbox(
                "Metric", ["Disabled", "Revenue Growth", "Earnings Growth"], index=0,
                label_visibility="collapsed", key="fund_mode",
                help="Choose 'Revenue Growth' to screen by sales increases, or 'Earnings Growth' for bottom-line profit growth. Set to 'Disabled' to skip fundamentals entirely."
            )
            fund_rate = st.text_input(
                "Min Rate %", value="", key="fund_rate", placeholder="e.g. 5",
                help="The minimum year-over-year growth rate required. For example, '10' means the company must have grown revenue or earnings by at least 10% each year."
            )
            fund_years = st.selectbox(
                "Years", ["1", "2", "3"], index=0,
                label_visibility="collapsed", key="fund_years",
                help="How many consecutive years the growth rate must hold. '2' means the company grew at the minimum rate for two years running."
            )

    with col_a:
        with st.expander("🏔️ Distance to ATH", expanded=filters_expanded):
            min_ath = st.text_input(
                "Min %", value="", key="min_ath", placeholder="e.g. 1",
                help="The furthest below its all-time high a stock can be. For example, '5' means the stock must be at least 5% below its peak."
            )
            max_ath = st.text_input(
                "Max %", value="", key="max_ath", placeholder="e.g. 20",
                help="The closest to its all-time high a stock can be. For example, '20' excludes stocks more than 20% below their peak. Helps avoid deeply beaten-down names."
            )

    # --- ACTION BUTTONS ROW ---
    acol1, acol2, acol3, acol4 = st.columns([1.5, 0.5, 1, 2.5])
    with acol1:
        inner1, inner2 = st.columns(2)
        with inner1:
            if st.button("🔽 Minimize", use_container_width=True, key="min_filters"):
                st.session_state.filters_minimized = True
                st.rerun()
        with inner2:
            if st.button("🔍 Maximize", use_container_width=True, key="max_filters"):
                st.session_state.filters_minimized = False
                st.rerun()
    with acol2:
        if st.button("🔄 Reset Filters", use_container_width=True, key="reset_filters"):
            st.session_state._reset_filters = True
            st.rerun()
    with acol4:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        run_clicked = st.button("🚀 RUN MARKET SCAN", disabled=not can_run_scan, use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Inputs Parsing and Live Validation
    try:
        min_ath_filter = parse_numeric_filter(min_ath, "Min Distance to ATH")
        max_ath_filter = parse_numeric_filter(max_ath, "Max Distance to ATH")
        
        min_sma_growth_filter = parse_numeric_filter(min_sma_growth, "Min Position %")
        max_sma_growth_filter = parse_numeric_filter(max_sma_growth, "Max Position %")
        
        min_atr_filter = parse_numeric_filter(min_atr_pct, "Min ATR%")
        max_atr_filter = parse_numeric_filter(max_atr_pct, "Max ATR%")
        
        fund_rate_target = parse_numeric_filter(fund_rate, "Min Rate %")
        fund_years_target = int(fund_years)
        
        if fund_mode != "Disabled" and fund_rate_target is None:
            validation_error = "⚠️ Please provide a target Min Rate percentage for fundamentals."
            can_run_scan = False
    except ValueError as e:
        validation_error = f"⚠️ {str(e)}"
        can_run_scan = False

    # Pack selected checkboxed variables
    sma_selections = {
        50: sma50_chk,
        100: sma100_chk,
        150: sma150_chk,
        200: sma200_chk
    }
    is_above_mode = ("Above" in sma_direction)

    # --- MAIN SCREEN INTERFACE ---
    if validation_error:
        st.error(validation_error)
    
    st.markdown("---")

    # SCAN PROCESS EXECUTION HANDLER
    refresh = st.session_state.get("refresh_clicked", False)
    if refresh:
        st.session_state.refresh_clicked = False
    if run_clicked or refresh:
        st.session_state.filters_minimized = True
        with st.status("🚀 Running Scan...", expanded=True) as status:
            try:
                status.write("📥 Downloading 1-year market data for all S&P 500 stocks...")
                start_time = time.time()
                
                raw_results, num_candidates = run_scanner(
                    tickers=st.session_state.sp500_tickers,
                    tickers_dict=st.session_state.sp500_data,
                    sma_selections=sma_selections,
                    alignment_enabled=alignment_chk,
                    min_ath_filter=min_ath_filter,
                    max_ath_filter=max_ath_filter,
                    growth_sma_period=int(growth_target_sma),
                    is_above_mode=is_above_mode,
                    min_sma_growth_filter=min_sma_growth_filter,
                    max_sma_growth_filter=max_sma_growth_filter,
                    min_atr_filter=min_atr_filter,
                    max_atr_filter=max_atr_filter,
                    fund_mode=fund_mode,
                    fund_rate_target=fund_rate_target,
                    fund_years_target=fund_years_target,
                    sma_slope=st.session_state.get("sma_slope", "Disabled"),
                    sma_slope_period=int(st.session_state.get("sma_slope_period", "50"))
                )
                
                duration = time.time() - start_time
                
                # Save results to context state
                st.session_state.raw_results = raw_results
                st.session_state.num_candidates = num_candidates
                st.session_state.last_duration = duration
                st.session_state.last_scan_time = time.strftime("%d/%m/%Y %H:%M")
                st.session_state.scan_complete = True
                
                status.update(label=f"✅ Scan completed in {duration:.1f}s!", state="complete", expanded=False)
                st.rerun()
                
            except Exception as e:
                status.update(label="❌ Scan failed!", state="error", expanded=True)
                st.error(f"Engine Failure: {e}")

    # WATCHLIST RESULTS RENDERER
    if st.session_state.get("scan_complete", False):
        rcol1, rcol2, rcol3, rcol4, rcol5 = st.columns([2, 1, 1.5, 1.5, 1])
        
        with rcol1:
            st.markdown("#### 📋 Market Watchlist Results")
        
        with rcol2:
            count = len(st.session_state.get("raw_results", []))
            st.markdown(
                f"<p style='font-size: 1.5rem; font-weight: 700; color: #2962ff; margin: 0.2rem 0 0 0; text-align: center;'>"
                f"{count} <span style='font-weight: 400; color: #787b86;'>matches</span></p>",
                unsafe_allow_html=True
            )
        
        with rcol3:
            if "last_duration" in st.session_state:
                st.markdown(
                    f"<p style='font-size: 1rem; font-weight: 700; color: #2962ff; margin: 0.2rem 0 0 0; text-align: center;'>⏱️ Scan duration</p>"
                    f"<p style='font-size: 1.5rem; font-weight: 700; color: #d1d4dc; margin: 0; text-align: center;'>{st.session_state.last_duration:.1f}s</p>",
                    unsafe_allow_html=True
                )
            if "last_scan_time" in st.session_state:
                st.markdown(
                    f"<p style='font-size: 0.85rem; color: #787b86; margin: 0; text-align: center;'>{st.session_state.last_scan_time}</p>",
                    unsafe_allow_html=True
                )
        
        with rcol4:
            display_mode = st.radio(
                "Display", ["Absolute Prices ($)", "Percentage Distance (%)"], index=0,
                horizontal=True, key="display_mode",
                help="Only affects Price column format. SMA and SMA Dist columns are always visible."
            )
        
        with rcol5:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()
                st.session_state.refresh_clicked = True
                st.rerun()
            
        raw_results = st.session_state.get("raw_results", [])
        
        if not raw_results:
            st.warning("⚠️ No stocks matched all filtering criteria.")
        else:
            df_display = build_numeric_display_df(raw_results, display_mode)
            is_percentage_mode = (display_mode == "Percentage Distance (%)")
            visible_columns = list(st.session_state.get("visible_columns", LOGICAL_COLUMNS))

            selected_cols = resolve_visible_columns(visible_columns, is_percentage_mode)
            selected_cols = [c for c in selected_cols if c in df_display.columns]
            df_display = df_display[selected_cols].dropna(how='all')

            pct_cols = [c for c in df_display.columns if 'SMA Dist' in c or 'ATH Dist' in c or c in ('1D', '5D', '1M', '6M', 'YTD', '1Y')]

            # Conditional color coding for percentage columns
            def _style_pct(v):
                if pd.isna(v):
                    return ''
                if v > 0:
                    return 'color: #089981;'
                if v < 0:
                    return 'color: #f23645;'
                return 'color: #d1d4dc;'

            styled = (df_display.style.map if hasattr(df_display.style, "map") else df_display.style.applymap)(_style_pct, subset=pct_cols)

            col_config = {}
            for col in df_display.columns:
                if col == "Price":
                    col_config[col] = st.column_config.NumberColumn(format="$%.2f")
                elif col == "ATR%":
                    col_config[col] = st.column_config.NumberColumn(format="%.2f%%")
                elif col in ("1D", "5D", "1M", "6M", "YTD", "1Y"):
                    col_config[col] = st.column_config.NumberColumn(format="%+.2f%%")
                elif "SMA Dist" in col or "ATH Dist" in col:
                    col_config[col] = st.column_config.NumberColumn(format="%+.2f%%")
                elif col == "ATH":
                    col_config[col] = st.column_config.NumberColumn(format="$%.2f")
                elif "SMA" in col:
                    col_config[col] = st.column_config.NumberColumn(format="$%.2f")

            nrows = len(df_display)
            table_height = max(50, min(600, 38 + nrows * 37))
            st.dataframe(
                styled,
                column_config=col_config,
                height=table_height,
                use_container_width=True,
                hide_index=True,
            )

            csv_data = df_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Current Watchlist as CSV",
                data=csv_data,
                file_name=f"sp500_watchlist_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.info("ℹ️ System Ready. Set your filters above, then click **RUN MARKET SCAN** to begin.")


# =====================================================================
# PLATFORM/RUNTIME INTERPRETER CHECK & BOOTSTRAP LAUNCHER
# =====================================================================

def is_running_in_streamlit():
    """True if running within an active Streamlit session instance."""
    return get_script_run_ctx() is not None


if __name__ == "__main__":
    if is_running_in_streamlit():
        run_streamlit_app()
    else:
        import os
        import sys
        import subprocess
        
        streamlit_bin = os.path.join(os.path.dirname(sys.executable), "streamlit")
        if not os.path.exists(streamlit_bin):
            streamlit_bin = "streamlit"
            
        cmd = [streamlit_bin, "run", __file__]
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass
        sys.exit(0)