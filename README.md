# Advisor-Brain-FSA

**Financial Statement Analysis** module — CFA Level 2 curriculum.

## Beneish M-Score

Detects earnings manipulation probability using 8 financial ratios (Beneish, 1999).

| Index | Name | Red Flag Direction |
|-------|------|--------------------|
| DSRI  | Days Sales in Receivables Index | > 1 |
| GMI   | Gross Margin Index              | > 1 |
| AQI   | Asset Quality Index             | > 1 |
| SGI   | Sales Growth Index              | > 1 |
| DEPI  | Depreciation Index              | > 1 |
| SGAI  | SGA Expense Index               | > 1 |
| LVGI  | Leverage Index                  | > 1 |
| TATA  | Total Accruals to Total Assets  | high positive |

**Threshold:** M-Score > −1.78 → *Potential Manipulator*

## Setup

```bash
pip install -r requirements.txt
python main.py          # run demo
pytest tests/           # run unit tests
```

## Project Structure

```
.
├── src/
│   └── advisor_brain_fsa/
│       ├── __init__.py
│       └── beneish_mscore.py   # BeneishMScore, FinancialData, MScoreResult
├── tests/
│   └── test_beneish_mscore.py
├── data/                       # raw financial data files (CSV, XLSX, etc.)
├── main.py                     # demo script
└── requirements.txt
```
