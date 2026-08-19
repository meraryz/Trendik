# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

"Trendik" — a single-file Streamlit app (`main.py`) that scans the S&P 500 for stocks matching
technical (SMA/ATR/ATH) and fundamental (revenue/earnings YoY growth) criteria, styled as a dark
trading-terminal UI. There is no separate backend; Streamlit's script-rerun model *is* the app.

## Commands

```bash
# Install dependencies (frozen from the working venv — no version ranges, pin exactly)
pip install -r requirements.txt

# Run the app (either works — main.py re-execs itself into `streamlit run` when not
# already inside a Streamlit runtime, see the __main__ block)
streamlit run main.py
python main.py

# Quick syntax/import sanity check without starting the server
python -c "import main"
```

There is no test suite, linter, or formatter configured in this repo — don't assume `pytest`,
`ruff`, etc. are available unless you add and wire them up yourself.

## Architecture

Everything lives in `main.py`, organized top-to-bottom into clearly commented sections:

1. **Persistence helpers** (top of file) — `load_presets`/`save_presets` and
   `load_column_prefs`/`save_column_prefs` read/write two JSON files that double as the app's only
   durable state:
   - `filter_presets.json` — named filter configurations, plus two reserved keys: `_favorite`
     (name of the preset auto-loaded on first run) and `_font_size` (table font tag). Any key
     prefixed with `_` is metadata, not a user preset — preserve that convention when adding new
     global settings here rather than inventing a second config file.
   - `user_columns.json` — the user's visible-column selection/order for the results grid.
   Both files are also mirrored into `st.session_state` (`_presets_cache`) so repeated reruns
   within a session don't re-hit disk.

2. **Cached data fetching layer** (`@st.cache_data`) — `get_sp500_tickers()` scrapes the
   Wikipedia constituents table; `download_technical_data()` bulk-downloads 1y OHLC via
   `yf.download` (TTL 1h); `get_cached_financials()` fetches yearly statements per-ticker
   (TTL 24h). The TTLs are the mechanism that keeps repeated scans fast — don't bypass
   `st.cache_data` with manual caching.

3. **Core scanning engine** (`run_scanner` + `process_ticker_fundamentals`) — two-stage
   screening, not a single pass:
   - Stage 1 is fully vectorized pandas over the whole universe at once: SMAs (50/100/150/200),
     ATR%, ATH distance, return windows (1D/5D/1M/6M/YTD/1Y), and all technical filter masks are
     computed as DataFrame/Series operations, never per-ticker loops. New technical filters should
     follow this pattern (build a boolean mask, AND it into `passed_mask`) rather than looping.
   - Stage 2 only runs for tickers that survived stage 1: a `ThreadPoolExecutor(max_workers=50)`
     fans out `process_ticker_fundamentals`, which is the only place that does the (slow,
     per-ticker) fundamentals lookup. This ordering — cheap vectorized filter first, expensive
     per-ticker work second, on a pre-shrunk candidate set — is intentional for performance; keep
     new expensive/per-ticker checks in stage 2.

4. **UI helpers** — `parse_numeric_filter` (blank-tolerant float parsing for filter text inputs),
   `build_numeric_display_df` (turns raw scan results into a display DataFrame, applying the
   Absolute vs. Percentage display mode), `resolve_visible_columns` / `LOGICAL_COLUMNS` /
   `COLUMN_GROUPS` (the mapping between a "logical" column name like `SMA50` and its concrete
   display column, which differs depending on display mode — e.g. `50 SMA` vs `50 SMA Dist`).

5. **Grid rendering** (`render_watchlist_grid`) — renders results via `st_aggrid`
   (drag-to-reorder columns, custom cell formatters/colors as `JsCode`), and persists any
   column-order change the user makes back to `user_columns.json` via
   `_persist_grid_column_order`. Falls back to a plain styled `st.dataframe`
   (`_render_table_fallback`) if AG Grid raises — when touching grid rendering/formatting, mirror
   the change in both code paths or the fallback silently drifts out of sync.

6. **`run_streamlit_app()`** — the entire page: injected CSS block, session-state
   initialization/reset/preset-load handling (all done *before* widgets render, since Streamlit
   requires session_state mutations for a widget's key to happen pre-instantiation), the filter
   panel (Technical / Relative Position / Fundamentals / Distance to ATH), the presets popover
   (save/update/delete/favorite), the scan trigger, and the results section.

### Streamlit-specific conventions to preserve

- Session-state mutations that need to affect widget defaults (reset, preset load, favorite
  auto-load) are applied at the top of `run_streamlit_app()`, guarded by a `_reset_filters` /
  `_load_preset_name` flag set on the *previous* rerun, then `st.rerun()`'d — never mutate a
  widget's session_state key after the widget has already been instantiated in the same run.
- Deferred toasts follow the same pattern: set `st.session_state._preset_toast`, `st.rerun()`,
  then pop and show it before any widgets render on the next run.
- Widget `key=` strings are the actual state contract (presets save/restore by these exact keys)
  — renaming a widget's `key` breaks every saved preset in `filter_presets.json` unless you
  migrate them.
