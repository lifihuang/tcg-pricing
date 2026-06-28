# Riftbound Hedonic Valuation Pipeline

An explainable pricing model for Riftbound TCG cards, with a specific application to ranking high-end tournament prize cards that have little or no public sales history.

---

## What it does

The pipeline decomposes card prices into the implicit ("shadow") value of each individual attribute — rarity, scarcity, foil treatment, champion identity, set — using a **hedonic regression on log-price**. Once those per-attribute prices are estimated from the regular card market, they can be recombined to value cards that have never been sold publicly, or to rank a set of thinly-traded prize cards by predicted value.

Three models run in sequence:

1. **Hedonic OLS** — the core model. Fits a log-price regression and outputs a coefficient for every attribute, each with a 95% confidence interval. Every dollar of any estimate traces directly to a named feature, making the output fully auditable.
2. **Quantile-regression reserve** — fits the 20th percentile of the price distribution rather than the mean, producing a defensible auction floor price rather than an optimistic point estimate.
3. **Gradient-boosted benchmark** — a non-linear cross-check. If its R² is materially higher than the OLS R², the linear model is missing something. Permutation importance confirms which attributes drive value without assuming any functional form.

The final output is a ranked table of the high-end promo cards, ordered by predicted value, with 80% prediction intervals and a Spearman rank correlation against known actual prices.

---

## Data sources

| Source | What it provides | Path |
|---|---|---|
| [tcgcsv.com](https://tcgcsv.com) | Regular-card market prices (scraped on first run, cached afterwards) | `data/comps.csv` |
| `highend_comps.csv` | High-end tournament prize cards with known populations and last-sale prices | `data/highend_comps.csv` |

The scraper fetches all 9 Riftbound groups from category 89 on tcgcsv.com, drops non-card products (anything without a card number), and filters down to cards whose name contains a known champion. Prices use `marketPrice` with a fallback to `midPrice`; `highPrice` is ignored due to price-parking on TCGplayer.

---

## Setup and usage

**Dependencies:**
```
pip install numpy pandas scikit-learn matplotlib requests
```

**First run** (no cached data):
```
python hedonic_valuation.py
```
The scraper will fetch all Riftbound card prices from tcgcsv.com and save them to `data/comps.csv`. Subsequent runs use the cache.

**Using your own comps:**
Replace `data/comps.csv` with any CSV that contains the expected columns (see `build_design()` in the source). The pipeline will use it as-is.

**Outputs** (written to `outputs/`):

| File | Contents |
|---|---|
| `promo_ranking.csv` | Full ranked table of prize cards with predicted values and intervals |
| `implicit_prices.png` | Bar chart of per-attribute value premiums with confidence intervals |
| `actual_vs_predicted.png` | Log-log scatter of hedonic fit on training data |
| `promo_ranking.png` | Horizontal bar chart of promo card rankings with actual prices overlaid |

---

## Methodological note

The same pipeline transfers directly to valuing any **unique, thinly-traded asset** — personalised licence plates are a direct analogue — where a set of reference market transactions exists for comparable (but not identical) items. The explainability and prediction-interval requirements make it appropriate for regulated or publicly accountable pricing decisions.

---

## Known limitations and TODOs

**Champion popularity**
Currently assigned by alphabetical rank of the champion name (A = low, Z = high). This is a stable placeholder that at least produces variation in the feature, but it has no connection to real demand. It should be replaced with a proper popularity signal, for example:

- Google Trends search volume for each champion name
- Champion pick/ban rate from competitive League of Legends data (e.g. [lolalytics.com](https://lolalytics.com))
- Cumulative TCGplayer sales volume per champion across all their cards

**Meta tier**
Also currently assigned by alphabetical rank. This is intended to capture how competitively playable a champion is in the Riftbound TCG meta, which is distinct from their popularity in League of Legends. Proper values should come from:

- Riftbound tournament result data (top-8 composition by champion)
- Community tier lists from dedicated Riftbound content creators

**Condition grades**
tcgcsv.com does not expose individual listing conditions — all prices are treated as near-mint. Cards graded by BGS or PSA (e.g. "Best of Leona BGS 9.5") command a significant premium that the model currently cannot capture. Adding a condition or grade column to `highend_comps.csv` and sourcing graded-sale comps would improve accuracy substantially.

**Print run**
For regular cards the actual print run is not available from tcgcsv, so it is proxied by rarity tier (Common = 50,000, …, Legendary = 600). For prize cards the population column is used as the true print run, which is correct in principle. Both should be replaced with verified print-run data when available.

**Training set size**
The model is most reliable when trained on hundreds of transactions. If the scraped dataset is small (< 200 rows after the champion filter), coefficient estimates will have wide confidence intervals and the ranking should be treated as directional only.

**Set name matching**
Prize card set names (Origins, Spiritforged, Unleashed) are matched against tcgcsv set names by substring. If tcgcsv uses different naming conventions, set dummies for the promo cards may all resolve to zero (i.e. the base set), slightly underweighting set effects.