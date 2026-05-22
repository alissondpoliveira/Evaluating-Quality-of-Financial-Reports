# Financial Statement Quality Analysis — Beneish M-Score

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![CFA](https://img.shields.io/badge/CFA-Level%20II%20Candidate-003087?logo=data:image/svg+xml;base64,)

---

## What this does

This tool fetches real financial statement data from Brazil's **CVM (Comissão de Valores Mobiliários)** regulatory portal and applies the **Beneish M-Score model** (Beneish, 1999) to estimate the probability of earnings manipulation in publicly traded companies on B3.

The model computes eight financial indices derived from two consecutive years of income statement and balance sheet data. When the composite score exceeds the threshold of **M-Score > −1.78**, the company is classified as a potential manipulator. The output is delivered through a Bloomberg Terminal-style dark web UI, with a Gemini AI narrative layer for plain-language risk interpretation.

Key capabilities:

- Live data pull from the CVM DFP (Demonstrações Financeiras Padronizadas) API
- Full non-financial B3 universe via dynamic CVM registry (banks and insurers excluded — M-Score is not applicable to financial intermediaries)
- Sign-convention normalisation for CVM filings (depreciation stored as positive add-back vs. negative expense varies by company)
- Accrual Ratio (CFO-based) as a complementary quality signal
- Offline market-wide cache (`build_market_cache.py`) for ranking all evaluable companies
- AI-generated risk narrative via Google Gemini (3-sentence thesis, cached to disk)

---

## The Eight Beneish Indices

| Index | Full Name | Formula (simplified) | Red Flag Direction |
|-------|-----------|----------------------|--------------------|
| **DSRI** | Days Sales in Receivables Index | (Receivables_t / Sales_t) ÷ (Receivables_t−1 / Sales_t−1) | > 1 — receivables growing faster than revenue |
| **GMI** | Gross Margin Index | Gross Margin_t−1 ÷ Gross Margin_t | > 1 — deteriorating margins |
| **AQI** | Asset Quality Index | (1 − (CA + PPE) / TA)_t ÷ (1 − (CA + PPE) / TA)_t−1 | > 1 — rising non-current / intangible asset intensity |
| **SGI** | Sales Growth Index | Sales_t ÷ Sales_t−1 | > 1 — high growth increases manipulation incentive |
| **DEPI** | Depreciation Index | Depr. Rate_t−1 ÷ Depr. Rate_t | > 1 — slowing depreciation rate may inflate assets |
| **SGAI** | SG&A Expense Index | (SGA / Sales)_t ÷ (SGA / Sales)_t−1 | > 1 — overhead growing relative to revenue |
| **LVGI** | Leverage Index | (LTD + CL)_t / TA_t ÷ (LTD + CL)_t−1 / TA_t−1 | > 1 — increasing leverage, tighter debt covenants |
| **TATA** | Total Accruals to Total Assets | (Net Income − CFO) ÷ Total Assets_t | High positive — large accrual component vs. cash earnings |

**Composite score (Beneish 1999 probit model):**

```
M = −4.84 + 0.920·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI
        + 0.115·DEPI − 0.172·SGAI + 4.679·TATA − 0.327·LVGI

M > −1.78  →  Potential Manipulator
M ≤ −1.78  →  Non-Manipulator
```

---

## Methodology

The model is drawn directly from:

> Beneish, M. D. (1999). *The Detection of Earnings Manipulation*. Financial Analysts Journal, 55(5), 24–36.

The eight-variable probit specification was calibrated on a sample of SEC enforcement actions and remains one of the most widely cited quantitative screens for accounting quality in sell-side and forensic research. The model is covered in the **CFA Institute Level II curriculum** under the *Financial Statement Analysis* topic area (Reading: Evaluating Quality of Financial Reports).

**Implementation notes:**

- All cost/expense accounts (COGS, SG&A, D&A) are stored as absolute values internally — CVM filings are inconsistent in whether depreciation appears as a negative expense or a positive add-back in the indirect cash flow statement. A two-layer normalisation (fetcher + model `__post_init__`) ensures DEPI and TATA are never inverted.
- TATA is computed from the direct definition (Net Income − Operating CFO) / Total Assets, which is sign-invariant and does not require the balance-sheet accrual approximation.
- Indices that must be strictly positive (DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI) fall back to 1.0 (neutral) if the calculation yields a negative result, with a logged warning.

---

## Project Structure

```
.
├── app_dash.py                  # Dash web application (Bloomberg-style UI)
├── build_market_cache.py        # Offline batch scorer — writes market_ranking_current.json
├── entrypoint.sh                # Railway/Docker startup: cache build (bg) + gunicorn
├── main.py                      # CLI demo script
├── requirements.txt
├── Dockerfile
├── railway.json
│
├── src/
│   └── advisor_brain_fsa/
│       ├── beneish_mscore.py    # BeneishMScore model, FinancialData dataclass, MScoreResult
│       ├── accruals.py          # CFO-based Accrual Ratio
│       ├── cvm_accounts.py      # Account code → financial field mapping (CVM DFP schema)
│       ├── cvm_registry.py      # CVM company registry, resolve_query(), get_company_profile()
│       ├── data_fetcher.py      # CVMDataFetcher — pulls DFP data from the CVM API
│       ├── mda_analyst.py       # Gemini AI narrative generator (cached to disk)
│       ├── rank_market.py       # Market-wide ranking helpers
│       ├── report_generator.py  # Report assembly
│       ├── sector_scorer.py     # SectorScorer chain of responsibility
│       └── ticker_map.py        # Static B3 ticker → keyword / sector map (~200 tickers)
│
├── tests/
│   ├── test_beneish_mscore.py
│   ├── test_data_fetcher.py
│   ├── test_mda_analyst.py
│   └── test_rank_market.py
│
├── assets/
│   └── typography.css           # Dash CSS overrides (IBM Plex Mono, dark theme)
│
└── data/
    ├── cache/                   # Disk cache for CVM API responses
    └── ai_reports/              # Gemini narrative cache (one .md per ticker+year)
```

---

## Setup & Usage

**Requirements:** Python 3.10+, a [Google AI Studio](https://aistudio.google.com/) API key for the Gemini narrative feature.

```bash
# 1. Clone and install
git clone https://github.com/alissondpoliveira/evaluating-quality-of-financial-reports.git
cd evaluating-quality-of-financial-reports
pip install -r requirements.txt

# 2. Set environment variables
export GOOGLE_API_KEY="your_gemini_api_key"

# 3. Run the web app locally
python app_dash.py
# → open http://localhost:8050

# 4. Build the market-wide ranking cache (optional, ~20 min)
python build_market_cache.py --workers 4

# 5. Run unit tests
pytest tests/ -v
```

**Docker / Railway:**

```bash
docker build -t advisor-brain-fsa .
docker run -e GOOGLE_API_KEY=xxx -p 8080:8080 advisor-brain-fsa
```

The `entrypoint.sh` script starts `build_market_cache.py` in the background and immediately launches gunicorn, so the Railway health check passes without waiting for the full cache build.

---

## Example Output

```
Company:     WEG S.A. (WEGE3)
Period:      FY 2023 vs FY 2022
─────────────────────────────────────────────────────────
Index    Value    Threshold    Signal
DSRI     0.9714   > 1          ✓ Normal
GMI      1.0231   > 1          ⚠ Watch
AQI      0.8802   > 1          ✓ Normal
SGI      1.1543   > 1          ⚠ Watch (revenue growth +15.4%)
DEPI     1.0041   > 1          ⚠ Watch
SGAI     0.9878   > 1          ✓ Normal
LVGI     0.9201   > 1          ✓ Normal
TATA    −0.0312   high pos.    ✓ Normal (cash-backed earnings)
─────────────────────────────────────────────────────────
M-Score:       −2.4817
Threshold:     −1.78
Classification: Non-Manipulator  ✓
Accrual Ratio: −0.0312  (low accruals — high earnings quality)
Risk Level:    🟢 Normal
─────────────────────────────────────────────────────────
Grade: A  |  High-quality financials. Strong cash conversion,
           conservative asset recognition, no revenue-
           receivables divergence detected.
```

*Values above are illustrative. The live application pulls real CVM DFP filings.*

---

## Live Demo

Deployed on Railway: **[evaluating-quality-of-financial-reports-production.up.railway.app](https://evaluating-quality-of-financial-reports-production.up.railway.app)**

---

## Author

**Alisson Oliveira**

CFA Level II Candidate · CFP® · Production Engineer · 6 years in financial markets (Itaú, Santander, XP Investimentos)

- LinkedIn: [linkedin.com/in/alissonpoliveira](https://linkedin.com/in/alissonpoliveira)
- GitHub: [github.com/alissondpoliveira](https://github.com/alissondpoliveira)

This project is part of a portfolio focused on quantitative financial analysis, forensic accounting, and the practical application of CFA curriculum methodologies to real market data.

---

## License

MIT — see [LICENSE](LICENSE).
