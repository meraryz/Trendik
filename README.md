# 📊 S&P 500 Technical & Fundamental Stock Scanner

A real-time, desktop-based stock market scanner built with Python, Tkinter, and `yfinance`. This application bulk-downloads technical data for the entire S&P 500 index and dynamically processes fundamental income statements via multithreading to isolate premium trading candidates.

---

## ⚡ Core Features

* **Bulk Technical Parsing:** Downloads 2-year daily historical pricing intervals for all S&P 500 tickers instantly using vectorized dataframes.
* **Multi-SMA Criteria & Alignment Filter:** Screen assets based on 50, 100, 150, and 200-day Simple Moving Averages, including strict sequential trend alignment constraints ($50 > 100 > 150 > 200$).
* **Live Fundamental Growth Parsing:** Dynamically crawls multi-year annual corporate filings to compute consecutive Year-over-Year (YoY) revenue or earnings growth performance.
* **Asynchronous Multi-Threaded Engine:** Integrates a thread-isolated `ThreadPoolExecutor` context layer to process complex fundamental evaluations simultaneously without locking up the UI.
* **Dual UI Display Formats:** Toggle results instantly between structural Absolute Prices ($) and relative Percentage Distance (%) configurations.

---

## 🛠️ Installation & Setup

Ensure you have Python 3.8+ installed on your system.

### 1. Clone or Copy the Repository
Save the application source code as `scanner.py`.

### 2. Install Required Modules
Run the following command in your terminal terminal to install the necessary library dependencies:

```bash
pip install yfinance pandas lxml html5lib requests