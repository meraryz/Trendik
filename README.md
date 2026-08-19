# 📊 Trendik — S&P 500 Technical & Fundamental Stock Scanner

A real-time, browser-based stock market scanner built with Python, Streamlit, and `yfinance`. The
app bulk-downloads a year of technical data for the entire S&P 500 index and dynamically
processes fundamental income statements via multithreading to isolate premium trading candidates.

---

## ⚡ Core Features

* **Bulk Technical Parsing:** Downloads 1-year daily historical pricing intervals for all S&P 500 tickers using vectorized dataframes.
* **Multi-SMA Criteria & Alignment Filter:** Screen assets based on 50, 100, 150, and 200-day Simple Moving Averages, including strict sequential trend alignment constraints ($50 > 100 > 150 > 200$), plus SMA slope (rising/falling) and ATR% volatility filters.
* **Relative Position Filter:** Find stocks a given % above or below a chosen SMA.
* **Live Fundamental Growth Parsing:** Dynamically crawls multi-year annual corporate filings to compute consecutive Year-over-Year (YoY) revenue or earnings growth performance.
* **Distance to All-Time High:** Filter by how far a stock is trading from its 1-year high.
* **Asynchronous Multi-Threaded Engine:** Integrates a thread-isolated `ThreadPoolExecutor` context layer to process complex fundamental evaluations simultaneously without locking up the UI.
* **Dual UI Display Formats:** Toggle results instantly between structural Absolute Prices ($) and relative Percentage Distance (%) configurations.
* **Interactive Results Grid:** Drag-to-reorder and show/hide columns (saved per-user), adjustable table font size, CSV export.
* **Filter Presets:** Save, update, delete, and favorite named filter configurations for quick recall.

---

## 🛠️ Installation & Setup

Ensure you have Python 3.8+ installed on your system.

### 1. Clone the repository

### 2. Install dependencies
Run the following command in your terminal to install the pinned library dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run main.py
```

(or `python main.py`, which re-execs itself into `streamlit run` if not already running inside a
Streamlit runtime).
