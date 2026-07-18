import streamlit as st
import yfinance as yf
import pandas as pd
import urllib.request
import ssl
import io
import time
from concurrent.futures import ThreadPoolExecutor
from streamlit.runtime.scriptrunner import get_script_run_ctx

# =====================================================================
# CACHED DATA FETCHING LAYER
# =====================================================================

@st.cache_data(ttl=86400)  # Cache Wikipedia S&P 500 list for 24 hours
def get_sp500_tickers():
    """Acquires and returns S&P 500 companies list from Wikipedia."""
    ssl_context = ssl._create_unverified_context()
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36'
    }
    
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, context=ssl_context) as response:
        html_content = response.read().decode('utf-8')
    
    html_stream = io.StringIO(html_content)
    table = pd.read_html(html_stream)
    df = table[0] 
    
    sp500_data = {}
    for _, row in df.iterrows():
        clean_ticker = str(row['Symbol']).replace('.', '-')
        sp500_data[clean_ticker] = str(row['Security'])
        
    return sp500_data


@st.cache_data(ttl=3600)  # Cache bulk pricing data for 1 hour
def download_technical_data(tickers):
    """Downloads 1-year daily historical pricing timelines for all tickers."""
    price_df = yf.download(tickers, period="1y", group_by='ticker', progress=False, threads=True)
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
    fund_mode, fund_rate_target, fund_years_target, company_name
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
    fund_years_target
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
    if is_above_mode:
        passed_mask &= (current_prices > target_sma_vals)
        sma_pos_pct = ((current_prices - target_sma_vals) / target_sma_vals) * 100
    else:
        passed_mask &= (current_prices < target_sma_vals)
        sma_pos_pct = ((target_sma_vals - current_prices) / current_prices) * 100
        
    if min_sma_growth_filter is not None:
        passed_mask &= (sma_pos_pct >= min_sma_growth_filter)
    if max_sma_growth_filter is not None:
        passed_mask &= (sma_pos_pct <= max_sma_growth_filter)
    
    # 5. ATR% Volatility filters
    if min_atr_filter is not None:
        passed_mask &= (atr_pct_vals >= min_atr_filter)
    if max_atr_filter is not None:
        passed_mask &= (atr_pct_vals <= max_atr_filter)

    # Identify candidates that passed ALL technical filters
    candidates = passed_mask[passed_mask].index.tolist()
    
    # Prepare data for those candidates
    pre_calculated_data = []
    for ticker in candidates:
        try:
            s_dict = {p: float(smas_matrix[p][ticker]) for p in sma_periods}
            pre_calculated_data.append((
                ticker, float(current_prices[ticker]), s_dict, 
                float(ath_prices[ticker]), float(ath_dist_pct[ticker]),
                float(atr_pct_vals[ticker])
            ))
        except Exception: 
            continue

    # Step 2: High-concurrency processing for candidates
    matches = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [
            executor.submit(
                process_ticker_fundamentals, 
                t, p, s, ath, ath_n, atr_v,
                fund_mode, fund_rate_target, fund_years_target, 
                tickers_dict.get(t, "Unknown")
            )
            for t, p, s, ath, ath_n, atr_v in pre_calculated_data
        ]
        for future in futures:
            res = future.result()
            if res is not None: 
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
    """Generates display-ready Pandas DataFrame with raw floats for correct sorting."""
    if not raw_results:
        return pd.DataFrame()
    
    rows = []
    is_percentage_mode = (display_mode == "Percentage Distance (%)")
    
    for item in raw_results:
        price = item["price"]
        ticker = item["ticker"]
        name = item["name"]
        ath = item["ath"]
        ath_dist = item["ath_needed"]
        atr_pct = item.get("atr_pct", None)
        
        row = {
            "Ticker": ticker,
            "Company Name": name,
            "Price": price,
        }
        
        for p in [50, 100, 150, 200]:
            sma_val = item[f"sma{p}"]
            if is_percentage_mode:
                pct_dist = ((price - sma_val) / sma_val) * 100
                row[f"{p} SMA Dist"] = pct_dist
            else:
                row[f"{p} SMA"] = sma_val
                
        row["ATH"] = ath
        row["ATH Dist"] = ath_dist
        if atr_pct is not None:
            row["ATR%"] = atr_pct
        
        rows.append(row)
        
    return pd.DataFrame(rows)


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

    # Initialize state before UI blocks
    can_run_scan = True
    validation_error = None
    
    # Kick off background data download so the UI stays responsive
    if "preload_started" not in st.session_state:
        st.session_state.preload_started = True
        import threading
        threading.Thread(
            target=download_technical_data,
            args=(st.session_state.sp500_tickers,),
            daemon=True
        ).start()

    # --- TITLE ---
    col_title1, col_title2, col_title3 = st.columns([1.5, 2, 1])
    with col_title1:
        st.markdown(
            "<h2 style='text-align: left; margin-bottom: 0; color: #2962ff;'>Trendik</h2>"
            "<p style='text-align: left; font-size: 0.85rem; color: #787b86; margin-top: 0;'>Find the perfect stocks for you.</p>",
            unsafe_allow_html=True
        )
    with col_title3:
        st.markdown("<div style='margin-top: 0.3rem;'>", unsafe_allow_html=True)
        run_clicked = st.button("🚀 RUN MARKET SCAN", disabled=not can_run_scan, use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- HORIZONTAL FILTER PANE ---
    col_t, col_r, col_f, col_a = st.columns(4)

    with col_t:
        with st.expander("📈 Technical"):
            sma50_chk = st.checkbox(
                "Above 50-Day SMA", value=True,
                help="Turn on to only show stocks currently trading above their 50-day average — catches short-term momentum."
            )
            sma100_chk = st.checkbox(
                "Above 100-Day SMA", value=False,
                help="Add the 100-day filter to confirm the stock has medium-term bullish strength behind it."
            )
            sma150_chk = st.checkbox(
                "Above 150-Day SMA", value=False,
                help="Enables a mid-to-long-term trend filter. Best for investors looking for sustained upward moves."
            )
            sma200_chk = st.checkbox(
                "Above 200-Day SMA", value=False,
                help="The classic bull-market indicator. Stocks above their 200-day average are considered in a long-term uptrend."
            )
            alignment_chk = st.checkbox(
                "Bullish Alignment (50>100>150>200)", value=False,
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
        with st.expander("📐 Relative Position"):
            growth_target_sma = st.selectbox(
                "Target SMA",
                ["50", "100", "150", "200"],
                index=0,
                format_func=lambda x: f"{x}-Day SMA",
                label_visibility="collapsed",
                help="Pick which Simple Moving Average (SMA) to measure distance from: 50-Day (short-term), 100-Day (medium), 150-Day (mid-long), or 200-Day (long-term)."
            )
            sma_direction = st.selectbox(
                "Position", ["Above (Price > SMA)", "Below (Price < SMA)"], index=0,
                label_visibility="collapsed",
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

    with col_f:
        with st.expander("💰 Fundamentals"):
            fund_mode = st.selectbox(
                "Metric", ["Disabled", "Revenue Growth", "Earnings Growth"], index=0,
                label_visibility="collapsed",
                help="Choose 'Revenue Growth' to screen by sales increases, or 'Earnings Growth' for bottom-line profit growth. Set to 'Disabled' to skip fundamentals entirely."
            )
            fund_rate = st.text_input(
                "Min Rate %", value="", key="fund_rate", placeholder="e.g. 5",
                help="The minimum year-over-year growth rate required. For example, '10' means the company must have grown revenue or earnings by at least 10% each year."
            )
            fund_years = st.selectbox(
                "Years", ["1", "2", "3"], index=0,
                label_visibility="collapsed",
                help="How many consecutive years the growth rate must hold. '2' means the company grew at the minimum rate for two years running."
            )

    with col_a:
        with st.expander("🏔️ Distance to ATH"):
            min_ath = st.text_input(
                "Min %", value="", key="min_ath", placeholder="e.g. 1",
                help="The furthest below its all-time high a stock can be. For example, '5' means the stock must be at least 5% below its peak."
            )
            max_ath = st.text_input(
                "Max %", value="", key="max_ath", placeholder="e.g. 20",
                help="The closest to its all-time high a stock can be. For example, '20' excludes stocks more than 20% below their peak. Helps avoid deeply beaten-down names."
            )

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
    if run_clicked:
        with st.status("🚀 Running Scan...", expanded=True) as status:
            try:
                status.write("⚙️ Running technical filters...")
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
                    fund_years_target=fund_years_target
                )
                
                duration = time.time() - start_time
                
                # Save results to context state
                st.session_state.raw_results = raw_results
                st.session_state.num_candidates = num_candidates
                st.session_state.last_duration = duration
                st.session_state.last_scan_time = time.strftime("%b %d, %Y at %I:%M %p")
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
                horizontal=True,
                help="'Absolute Prices' shows raw dollar figures. 'Percentage Distance' shows how far each stock is from each SMA as a % — easier for comparing across stocks."
            )
        
        with rcol5:
            if st.button("🔄 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.session_state.clear()
                st.toast("Cache cleared & state reset!")
                st.rerun()
            
        raw_results = st.session_state.get("raw_results", [])
        
        if not raw_results:
            st.warning("⚠️ No stocks matched all filtering criteria.")
        else:
            # Rebuild dataframe
            df_display = build_numeric_display_df(raw_results, display_mode)
            
            is_percentage_mode = (display_mode == "Percentage Distance (%)")
            
            # Configure native formatting configurations for optimal sorting and design
            col_config = {
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Company Name": st.column_config.TextColumn("Company Name", width="medium"),
                "Price": st.column_config.NumberColumn("Price ($)", format="$%.2f"),
                "ATH": st.column_config.NumberColumn("ATH ($)", format="$%.2f"),
                "ATH Dist": st.column_config.NumberColumn("Distance to ATH", format="%.2f%%"),
                "ATR%": st.column_config.NumberColumn("ATR%", format="%.2f%%"),
            }
            
            for p in [50, 100, 150, 200]:
                if is_percentage_mode:
                    col_config[f"{p} SMA Dist"] = st.column_config.NumberColumn(f"{p} SMA Dist", format="%+.2f%%")
                else:
                    col_config[f"{p} SMA"] = st.column_config.NumberColumn(f"{p} SMA ($)", format="$%.2f")
            
            # Conditional color for percentage columns
            pct_cols = [c for c in df_display.columns if 'SMA Dist' in c]
            
            def color_pct(val):
                if val is None or pd.isna(val):
                    return ''
                if val > 0:
                    return 'color: #089981'
                elif val < 0:
                    return 'color: #f23645'
                else:
                    return 'color: #d1d4dc'
            
            styled_df = df_display.style.map(color_pct, subset=pct_cols)
            
            # Display beautifully styled table
            st.dataframe(
                styled_df,
                column_config=col_config,
                use_container_width=True,
                hide_index=True,
            )
            
            # Interactive CSV download button
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