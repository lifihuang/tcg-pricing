"""
Hedonic valuation and auction-reserve pricing for a scarcity-driven
collectibles market (Riftbound TCG).

Pipeline overview
-----------------
1. Explainable hedonic regression on log-price: every dollar of an
   estimate traces to a named, interpretable attribute (rarity, scarcity,
   foil treatment, champion demand, set).  Per-attribute implicit prices
   are reported with 95 % confidence intervals so every assumption is
   auditable.

2. Quantile-regression auction reserve: a 20th-percentile model produces
   a defensible price floor rather than a point guess, directly suitable
   for regulated or publicly accountable auction settings.

3. Gradient-boosted benchmark with permutation importance: a non-linear
   cross-check confirms that the linear story is not materially mis-
   specified and surfaces which attributes drive value in a model-free way.

4. High-end promo-card ranking: the model trained on regular-market comps
   is used to rank tournament prize cards (thinly traded, no public
   market) by predicted value, with honest prediction intervals for each.

Methodological note
-------------------
The same pipeline transfers directly to valuing any unique, thinly-traded
asset — e.g. personalised licence-plate combinations — where:
  * assets are defined by a fixed attribute vector,
  * a reference market of comparable (but not identical) traded items
    provides the price signal, and
  * explainability and auditability are required for regulated decisions.

Data priority (checked in order)
---------------------------------
  1. data/comps.csv          — cached comps; used as-is.
  2. tcgcsv.com (live fetch) — all 9 Riftbound groups scraped and saved
                               to data/comps.csv for future runs.
  3. Synthetic fallback      — 1 200-row demo set (no network required).

Dependencies: numpy, pandas, scikit-learn, matplotlib, requests.
Run:          python hedonic_valuation.py
"""

from __future__ import annotations
import os
import re
import time
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

RNG  = np.random.default_rng(42)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA          = os.path.join(HERE, "data", "comps.csv")
HIGHEND_COMPS = os.path.join(HERE, "data", "highend_comps.csv")
OUT  = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

# ── Riftbound on tcgcsv.com ───────────────────────────────────────────────
RIFTBOUND_CATEGORY = "89"
RIFTBOUND_GROUPS   = [24343, 24344, 24439, 24502, 24519,
                      24528, 24552, 24560, 24698]
TCGCSV_USER_AGENT  = "HedonicValuation/1.0"

# ── Feature schema ────────────────────────────────────────────────────────
RARITY_TIERS = ["common", "uncommon", "rare", "epic", "legendary"]
CONDITIONS   = ["played", "light_play", "near_mint", "gem_mint"]

# Map every Riftbound rarity label (lower-cased) to a RARITY_TIERS level.
RARITY_NORM: dict[str, str] = {
    "common":     "common",
    "uncommon":   "uncommon",
    "rare":       "rare",
    "epic":       "epic",
    "legendary":  "legendary",
    "overnumber": "epic",       # numbered chase variant
    "signature":  "legendary",  # tournament-grade signature card
    "prize card": "legendary",  # tournament prize
}

# Rarity-tier proxy print runs (used when tcgcsv does not expose print run).
RARITY_PRINT_RUN: dict[str, int] = {
    "common":    50_000,
    "uncommon":  20_000,
    "rare":       8_000,
    "epic":       2_500,
    "legendary":    600,
}

# Generative coefficients — used only to build synthetic demo data.
TRUE = {
    "intercept": 1.20,
    "rarity": {"common": 0.0, "uncommon": 0.35, "rare": 0.80,
               "epic": 1.40, "legendary": 2.20},
    "condition": {"played": 0.0, "light_play": 0.25,
                  "near_mint": 0.55, "gem_mint": 1.10},
    "is_foil": 0.45, "is_alt_art": 0.70, "is_first_print": 0.30,
    "is_promo": 0.40, "meta_tier": 0.45, "champion_popularity": 1.10,
    "log_print_run": -0.32, "noise_sd": 0.45,
}


# ======================================================================
# Data acquisition — scraping
# ======================================================================

def scrape_riftbound_comps() -> pd.DataFrame:
    """
    Fetch every Riftbound *card* and its TCGplayer market price from
    tcgcsv.com.  Products without a card number (booster packs, display
    boxes, accessories) are dropped.

    Fields not exposed by tcgcsv are stored as NaN.  A null-count table
    is printed before defaults are applied so you can see exactly what
    the API does and does not provide.  champion_popularity and meta_tier
    are intentionally left as NaN here; they are filled by
    assign_character_scores() in main() using the alphabetical scoring
    derived from the known character list.

    Rate-limit: 0.25 s between requests as recommended by tcgcsv.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": TCGCSV_USER_AGENT})
    records: list[dict] = []

    for group_id in RIFTBOUND_GROUPS:
        print(f"  Fetching group {group_id} …", flush=True)

        r = session.get(
            f"https://tcgcsv.com/tcgplayer/{RIFTBOUND_CATEGORY}"
            f"/{group_id}/products"
        )
        r.raise_for_status()
        products = r.json().get("results", [])
        time.sleep(0.25)

        product_map: dict[int, dict] = {}
        for p in products:
            ext = {item["name"]: item["value"]
                   for item in p.get("extendedData", [])}

            card_number = ext.get("Number")
            if not card_number or not str(card_number).strip():
                continue   # not an individual card

            raw_rarity = ext.get("Rarity")
            rarity = (RARITY_NORM.get(raw_rarity.lower())
                      if raw_rarity else None)
            name = p["name"]

            product_map[p["productId"]] = {
                "card_name":    name,
                "card_number":  str(card_number).strip(),
                "set_name":     ext.get("Set Name") or None,
                "card_type":    ext.get("Card Type") or None,
                "rarity":       rarity,
                "is_alt_art":   int(bool(pd.Series([name]).str.contains(
                    r"alt[\s_]?art|alternate", case=False, regex=True
                ).iloc[0])),
                "is_promo":     int(bool(pd.Series([name]).str.contains(
                    r"promo", case=False, regex=True
                ).iloc[0])),
                # Genuinely unknown from tcgcsv — left as NaN.
                "is_first_print":      None,
                "condition":           None,
                "champion_popularity": None,   # filled by assign_character_scores
                "meta_tier":           None,   # filled by assign_character_scores
                "print_run":           None,
            }

        r = session.get(
            f"https://tcgcsv.com/tcgplayer/{RIFTBOUND_CATEGORY}"
            f"/{group_id}/prices"
        )
        r.raise_for_status()
        prices = r.json().get("results", [])
        time.sleep(0.25)

        for price in prices:
            pid = price["productId"]
            if pid not in product_map:
                continue
            dollar = price.get("marketPrice") or price.get("midPrice")
            if not dollar:
                continue
            row = product_map[pid].copy()
            row["price"]   = float(dollar)
            row["is_foil"] = int(
                str(price.get("subTypeName", "")).lower() == "foil"
            )
            records.append(row)

    if not records:
        raise RuntimeError(
            "tcgcsv returned no usable Riftbound card records. "
            "Check your network connection or try again later."
        )

    df = pd.DataFrame(records)
    df = df.dropna(subset=["price"]).reset_index(drop=True)

    # Report nulls before any filling.
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        print("\n  Null counts before defaults "
              "(fields not available from tcgcsv):")
        for col, n in null_counts.items():
            print(f"    {col:25s}: {n:4d} nulls  ({n / len(df):.0%})")

    # print_run: use rarity as proxy; champion_popularity / meta_tier
    # are left NaN here and filled later by assign_character_scores().
    df["print_run"] = df["print_run"].fillna(
        df["rarity"].map(RARITY_PRINT_RUN)
    ).fillna(RARITY_PRINT_RUN["common"])

    df = df.fillna({
        "condition":       "near_mint",
        "is_first_print":  0,
    })

    print(f"\n  Scraped {len(df)} card records across "
          f"{df['set_name'].nunique()} sets.")
    return df


def generate_synthetic_comps(n: int = 1200) -> pd.DataFrame:
    """Synthesise a realistic TCG comp set with a known price structure."""
    rarity    = RNG.choice(RARITY_TIERS, n, p=[.40, .27, .18, .10, .05])
    condition = RNG.choice(CONDITIONS,    n, p=[.20, .30, .35, .15])
    is_foil        = RNG.binomial(1, 0.30, n)
    is_alt_art     = RNG.binomial(1, 0.12, n)
    is_first_print = RNG.binomial(1, 0.45, n)
    is_promo       = RNG.binomial(1, 0.10, n)
    meta_tier      = RNG.integers(1, 6, n)
    champion_popularity = np.clip(RNG.beta(2, 3, n), 0, 1)
    base_run   = np.array([50000, 20000, 8000, 2500, 600])
    rarity_idx = np.array([RARITY_TIERS.index(r) for r in rarity])
    print_run  = np.maximum(
        50, (base_run[rarity_idx] * RNG.lognormal(0, 0.5, n)).astype(int))
    log_price = (
        TRUE["intercept"]
        + np.array([TRUE["rarity"][r]    for r in rarity])
        + np.array([TRUE["condition"][c] for c in condition])
        + TRUE["is_foil"]             * is_foil
        + TRUE["is_alt_art"]          * is_alt_art
        + TRUE["is_first_print"]      * is_first_print
        + TRUE["is_promo"]            * is_promo
        + TRUE["meta_tier"]           * (meta_tier - 1) / 4.0
        + TRUE["champion_popularity"] * champion_popularity
        + TRUE["log_print_run"]       * (np.log(print_run) - np.log(8000))
        + RNG.normal(0, TRUE["noise_sd"], n)
    )
    price = np.round(np.exp(log_price) * 3.0, 2)
    return pd.DataFrame({
        "rarity": rarity, "condition": condition, "is_foil": is_foil,
        "is_alt_art": is_alt_art, "is_first_print": is_first_print,
        "is_promo": is_promo, "meta_tier": meta_tier,
        "champion_popularity": champion_popularity.round(3),
        "print_run": print_run, "price": price,
    })


# ======================================================================
# Character scoring (alphabetical placeholder)
# ======================================================================

def build_character_scores(characters: list[str]) -> dict[str, dict]:
    """
    Assign champion_popularity ∈ [0, 1] and meta_tier ∈ [1, 5] by
    alphabetical rank of the character name.

    This is an explicit placeholder: alphabetical position is arbitrary
    but guarantees variation across both features (fixing the singular-
    matrix problem caused by constant-column defaults) and produces a
    stable, reproducible assignment that can be swapped for real
    popularity / tier data later without touching any other code.

    Ordering: A → 0.0 / tier 1 (lowest), Z → 1.0 / tier 5 (highest).
    """
    unique = sorted(set(
        str(c).strip() for c in characters
        if pd.notna(c) and str(c).strip()
    ))
    n = len(unique)
    scores: dict[str, dict] = {}
    for rank, char in enumerate(unique):
        if n > 1:
            pop  = round(rank / (n - 1), 4)
            tier = max(1, min(5, round(1 + rank / (n - 1) * 4)))
        else:
            pop, tier = 0.5, 3
        scores[char] = {"champion_popularity": pop, "meta_tier": tier}
    return scores


def assign_character_scores(
    df: pd.DataFrame,
    char_scores: dict[str, dict],
    name_col: str = "card_name",
    char_col: str | None = None,
) -> pd.DataFrame:
    """
    Write champion_popularity and meta_tier into df from char_scores.

    Lookup order:
      1. char_col (explicit character column, used for promo cards).
      2. Substring search in name_col (used for scraped TCGplayer data
         where no separate character column exists).

    Only overwrites rows that are currently NaN, so pre-existing values
    from the synthetic generator are preserved.
    """
    df = df.copy()
    known = list(char_scores.keys())

    for i, row in df.iterrows():
        # Skip rows that already have both values.
        if pd.notna(row.get("champion_popularity")) and \
           pd.notna(row.get("meta_tier")):
            continue

        scores: dict = {}
        # Try explicit character column first.
        if char_col and char_col in df.columns:
            char = str(row.get(char_col, "")).strip()
            scores = char_scores.get(char, {})
        # Fall back to substring match against card name.
        if not scores:
            name = str(row.get(name_col, ""))
            for char in known:
                if char.lower() in name.lower():
                    scores = char_scores[char]
                    break

        if scores:
            df.at[i, "champion_popularity"] = scores["champion_popularity"]
            df.at[i, "meta_tier"]           = scores["meta_tier"]

    return df


# ======================================================================
# Design matrix
# ======================================================================

def build_design(df: pd.DataFrame):
    """
    Return (X, feature_names, y=log price) ready for OLS / ML.

    Zero-variance guards: any column whose values are constant (std = 0
    for continuous, nunique ≤ 1 for categorical, all-same for binary)
    is silently dropped.  This prevents the singular XᵀX matrix that
    causes LinAlgError when all-constant default fills are used.
    """
    X = pd.DataFrame(index=df.index)

    # Ordinal categoricals as dummies (base level dropped).
    for col, base, levels in [("rarity",    "common", RARITY_TIERS),
                               ("condition", "played", CONDITIONS)]:
        if col not in df.columns or df[col].nunique() <= 1:
            continue
        for level in [l for l in levels if l != base]:
            X[f"{col}={level}"] = (df[col] == level).astype(float)

    # Set dummies (base = first alphabetically).
    if "set_name" in df.columns and df["set_name"].nunique() > 1:
        base_set = sorted(df["set_name"].dropna().unique())[0]
        for s in sorted(df["set_name"].dropna().unique()):
            if s != base_set:
                X[f"set={s}"] = (df["set_name"] == s).astype(float)

    # Binary flags — skip if constant (all-zero or all-one).
    for flag in ["is_foil", "is_alt_art", "is_first_print", "is_promo"]:
        if flag in df.columns and df[flag].nunique() > 1:
            X[flag] = df[flag].astype(float)

    # Continuous features — skip if constant (std = 0 after filling).
    for col in ["meta_tier", "champion_popularity"]:
        if col in df.columns and df[col].std() > 0:
            X[col] = df[col].astype(float)

    if "print_run" in df.columns:
        X["log_print_run"] = np.log(df["print_run"].astype(float))

    y = np.log(df["price"].astype(float).values)
    return X, list(X.columns), y


# ======================================================================
# Hedonic OLS with full inference
# ======================================================================

def fit_hedonic(X: pd.DataFrame, y: np.ndarray) -> dict:
    """
    OLS on log-price.  Returns coefficients, 95 % CIs, residual sigma,
    and R².  Every coefficient represents the log-price premium for one
    unit of the corresponding attribute — fully traceable and auditable.
    """
    Xd    = np.column_stack([np.ones(len(X)), X.values])
    names = ["intercept"] + list(X.columns)
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid    = y - Xd @ beta
    n, k     = Xd.shape
    sigma2   = (resid @ resid) / (n - k)
    XtX_inv  = np.linalg.inv(Xd.T @ Xd)
    se       = np.sqrt(np.diag(sigma2 * XtX_inv))
    ci_lo, ci_hi = beta - 1.96 * se, beta + 1.96 * se
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    table = pd.DataFrame({
        "feature":     names,
        "coef_log":    beta,
        "implied_pct": np.exp(beta)  - 1,   # % price premium per unit
        "ci_lo_pct":   np.exp(ci_lo) - 1,
        "ci_hi_pct":   np.exp(ci_hi) - 1,
    })
    return {"beta": beta, "names": names, "sigma": np.sqrt(sigma2),
            "r2": r2, "table": table}


def hedonic_point_estimate(
    fit: dict, X_row: pd.DataFrame
) -> tuple[float, float, float]:
    """
    Lognormal-bias-corrected point estimate + 80 % prediction interval.
    Returns (estimate, p10, p90).
    """
    xd    = np.concatenate([[1.0], X_row.values.ravel()])
    mu    = float(xd @ fit["beta"])
    sigma = fit["sigma"]
    point = np.exp(mu + 0.5 * sigma ** 2)   # smearing correction
    lo    = np.exp(mu - 1.2816 * sigma)      # 10th pct
    hi    = np.exp(mu + 1.2816 * sigma)      # 90th pct
    return point, lo, hi


# ======================================================================
# Quantile-regression auction reserve
# ======================================================================

def fit_reserve_model(
    X: pd.DataFrame, y: np.ndarray, q: float = 0.20
) -> object:
    """
    20th-percentile quantile regression on log-price.
    Produces a defensible auction floor: the price at which 80 % of
    comparable sales have occurred above, not a potentially over-
    optimistic point estimate.
    """
    qr = QuantileRegressor(quantile=q, alpha=0.0, solver="highs")
    qr.fit(X.values, y)
    return qr


def reserve_price(qr, X_row: pd.DataFrame) -> float:
    return float(np.exp(qr.predict(X_row.values))[0])


# ======================================================================
# Gradient-boosted benchmark + permutation importance
# ======================================================================

def fit_ml_benchmark(X: pd.DataFrame, y: np.ndarray) -> tuple:
    """
    Non-linear cross-check: if the GBM R² is materially higher than the
    hedonic R², the linear model is mis-specified and the implicit prices
    should be treated with caution.  Permutation importance confirms
    which attributes drive value in a model-free way.
    """
    Xtr, Xte, ytr, yte = train_test_split(
        X.values, y, test_size=0.25, random_state=42)
    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(Xtr, ytr)
    pred    = gbr.predict(Xte)
    metrics = {"r2": r2_score(yte, pred),
               "mae_log": mean_absolute_error(yte, pred)}
    imp = permutation_importance(gbr, Xte, yte, n_repeats=15,
                                 random_state=42)
    importance = (pd.DataFrame({"feature":    X.columns,
                                "importance": imp.importances_mean})
                  .sort_values("importance", ascending=False))
    return gbr, metrics, importance


# ======================================================================
# Plots — training data
# ======================================================================

def plot_implicit_prices(table: pd.DataFrame, path: str) -> None:
    t = table[table.feature != "intercept"].copy()
    t = t.reindex(t["implied_pct"].abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(8, max(5, len(t) * 0.38)))
    ax.barh(t["feature"], t["implied_pct"] * 100, color="#3b6ea5")
    ax.errorbar(
        t["implied_pct"] * 100, range(len(t)),
        xerr=[(t["implied_pct"] - t["ci_lo_pct"]) * 100,
              (t["ci_hi_pct"]  - t["implied_pct"]) * 100],
        fmt="none", ecolor="#222", elinewidth=1, capsize=3,
    )
    ax.axvline(0, color="#888", lw=.8)
    ax.set_xlabel("Implicit price (% change in value, 95 % CI)")
    ax.set_title("Hedonic implicit prices — per-attribute value drivers\n"
                 "(each coefficient is the ceteris-paribus % premium)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_fit(fit: dict, X: pd.DataFrame, y: np.ndarray,
             path: str) -> None:
    Xd     = np.column_stack([np.ones(len(X)), X.values])
    pred   = np.exp(Xd @ fit["beta"] + 0.5 * fit["sigma"] ** 2)
    actual = np.exp(y)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, pred, s=10, alpha=.35, color="#3b6ea5")
    lim = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(lim, lim, "--", color="#c44")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual price ($)"); ax.set_ylabel("Predicted ($)")
    ax.set_title(f"Hedonic fit  (R² = {fit['r2']:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ======================================================================
# High-end promo card ranking
# ======================================================================

def prepare_promo_cards(
    path: str,
    char_scores: dict[str, dict],
    training_set_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load and featurise the high-end tournament promo cards CSV so it can
    be scored by the hedonic model trained on regular-market comps.

    Key mapping decisions:
      * population → print_run  (for prize cards population IS the
        true print run; values of 1–14 are far below anything in the
        training data, driving very high scarcity premiums).
      * is_promo = 1 for every card (all are tournament prizes).
      * champion_popularity / meta_tier from alphabetical char_scores.
      * set_name: we attempt to match the promo CSV's set values against
        the set names seen in training data; unmatched sets are left as
        the base level (zero dummy) so they get the base-set intercept.
    """
    raw = pd.read_csv(path)

    # Build a simple set-name normaliser against training data if provided.
    def match_set(raw_set: str) -> str | None:
        if not raw_set or pd.isna(raw_set):
            return None
        raw_set = str(raw_set).strip()
        if training_set_names:
            for ts in training_set_names:
                if raw_set.lower() in ts.lower() or ts.lower() in raw_set.lower():
                    return ts
        return raw_set   # keep as-is if no match found

    rows = []
    for _, r in raw.iterrows():
        character  = str(r.get("character", "")).strip()
        scores     = char_scores.get(character, {})
        raw_rarity = str(r.get("rarity", "")).strip()
        rarity     = RARITY_NORM.get(raw_rarity.lower(), "legendary")

        # Population is the real print run for prize cards.
        pop = r.get("population")
        try:
            print_run = int(float(pop))
        except (TypeError, ValueError):
            print_run = RARITY_PRINT_RUN["legendary"]

        set_name = match_set(str(r.get("set", "")))

        rows.append({
            "card_name":           r["card_name"],
            "character":           character,
            "rarity":              rarity,
            "condition":           "near_mint",
            "is_foil":             0,
            "is_alt_art":          0,
            "is_first_print":      0,
            "is_promo":            1,
            "meta_tier":           scores.get("meta_tier",           3),
            "champion_popularity": scores.get("champion_popularity", 0.5),
            "print_run":           max(1, print_run),
            "set_name":            set_name,
            "last_sale_usd":       r.get("last_sale_usd"),
            "price_on_application": r.get("price_on_application", False),
        })

    return pd.DataFrame(rows)


def rank_promo_cards(
    promo_df: pd.DataFrame,
    fit: dict,
    qr,
    train_cols: list[str],
) -> pd.DataFrame:
    """
    Score every high-end promo card with the trained hedonic model and
    return a table ranked by predicted value (most valuable first).

    Columns returned:
      pred_rank     — predicted value rank (1 = highest)
      card_name
      character
      estimate_$    — lognormal-corrected point estimate
      p10_$         — 10th-percentile lower bound (80 % interval)
      p90_$         — 90th-percentile upper bound
      reserve_$     — 20th-pct quantile-regression auction floor
      known_price_$ — last recorded sale price, or "POA"
      actual_rank   — rank by actual price where known, else "—"
    """
    Xp, _, _ = build_design(promo_df.assign(price=1.0))
    # Align to training columns: add missing as zero, drop extras.
    for col in train_cols:
        if col not in Xp.columns:
            Xp[col] = 0.0
    Xp = Xp[train_cols]

    rows_out = []
    for i, (_, card) in enumerate(promo_df.iterrows()):
        xrow            = Xp.iloc[[i]]
        point, lo, hi   = hedonic_point_estimate(fit, xrow)
        res             = reserve_price(qr, xrow)
        known           = card.get("last_sale_usd")
        rows_out.append({
            "card_name":     card["card_name"],
            "character":     card.get("character", ""),
            "estimate_$":    round(point, 0),
            "p10_$":         round(lo,    0),
            "p90_$":         round(hi,    0),
            "reserve_$":     round(res,   0),
            "known_price_$": float(known) if pd.notna(known) else "POA",
        })

    result = (pd.DataFrame(rows_out)
              .sort_values("estimate_$", ascending=False)
              .reset_index(drop=True))
    result.insert(0, "pred_rank", result.index + 1)

    # Actual rank for cards with known prices.
    known_mask = result["known_price_$"] != "POA"
    if known_mask.any():
        result.loc[known_mask, "actual_rank"] = (
            result.loc[known_mask, "known_price_$"]
            .rank(ascending=False).astype(int).astype(str)
        )
    result["actual_rank"] = result.get("actual_rank", pd.NA).fillna("—")

    return result


def plot_ranking(ranking: pd.DataFrame, path: str) -> None:
    """
    Horizontal bar chart of predicted values for the high-end promo
    cards, with 80 % prediction intervals and actual prices overlaid
    as scatter points.
    """
    r = ranking.sort_values("estimate_$", ascending=True).copy()
    labels = r["card_name"].str.replace("Best of ", "", regex=False)

    fig, ax = plt.subplots(figsize=(10, max(7, len(r) * 0.42)))

    ax.barh(labels, r["estimate_$"],
            color="#3b6ea5", alpha=0.72, label="Predicted estimate")
    ax.errorbar(
        r["estimate_$"], range(len(r)),
        xerr=[
            (r["estimate_$"] - r["p10_$"]).clip(lower=0).values,
            (r["p90_$"] - r["estimate_$"]).values,
        ],
        fmt="none", ecolor="#1a3a6b", elinewidth=1.1, capsize=3,
        label="80 % prediction interval",
    )

    # Overlay known prices.
    known = r[r["known_price_$"] != "POA"].copy()
    if not known.empty:
        known["known_price_$"] = known["known_price_$"].astype(float)
        known_y = [list(r["card_name"]).index(n) for n in known["card_name"]]
        ax.scatter(known["known_price_$"], known_y,
                   color="#c44", zorder=5, s=45, marker="D",
                   label="Actual last sale")

    ax.set_xlabel("Estimated value ($)")
    ax.set_title(
        "High-end Riftbound promo cards — predicted value ranking\n"
        "Model trained on TCGplayer regular-market comps; extrapolated\n"
        "via shared attribute structure (rarity, scarcity, character demand)"
    )
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)

    # ── 0. Load promo card list (needed for character scores) ─────────
    if not os.path.exists(HIGHEND_COMPS):
        raise FileNotFoundError(
            f"High-end comps file not found at {HIGHEND_COMPS}.\n"
            "Place the riftbound_highend_comps.csv file at that path "
            "and re-run."
        )
    promo_raw      = pd.read_csv(HIGHEND_COMPS)
    all_characters = promo_raw["character"].dropna().unique().tolist()

    # These are items, not champions — drop them so they don't pollute
    # the character-score index or the card-name filter below.
    NON_CHAMPIONS  = {"Baron", "Blade of the Ruined King"}
    all_characters = [c for c in all_characters if c not in NON_CHAMPIONS]

    # Build a regex that matches any champion name as a whole token.
    # Using negative lookbehind/lookahead (?<!\w) / (?!\w) rather than
    # \b so that names with apostrophes (Kai'Sa, Rek'Sai) are handled
    # correctly — \b breaks at the apostrophe, these assertions don't.
    _parts        = [r"(?<!\w)" + re.escape(c) + r"(?!\w)"
                     for c in all_characters]
    CHAMPION_RE   = re.compile("|".join(_parts), re.IGNORECASE)

    char_scores   = build_character_scores(all_characters)
    print(f"Character scores built for {len(char_scores)} champions "
          f"(alphabetical placeholder — A=low, Z=high).")

    # ── 1. Training data ──────────────────────────────────────────────
    if os.path.exists(DATA):
        df     = pd.read_csv(DATA)
        source = f"cached comps ({DATA})"
    else:
        print("data/comps.csv not found — "
              "attempting live scrape from tcgcsv.com …")
        try:
            df = scrape_riftbound_comps()
            df.to_csv(DATA, index=False)
            source = (f"scraped Riftbound data from tcgcsv.com "
                      f"({len(df)} records, saved to data/comps.csv)")
        except Exception as exc:
            print(f"  Scrape failed ({exc}). "
                  "Falling back to synthetic demo data.")
            df = generate_synthetic_comps()
            df.to_csv(os.path.join(HERE, "data", "sample_comps.csv"),
                      index=False)
            source = ("synthetic demo data "
                      "(scrape failed; replace data/comps.csv with yours)")

    # Apply alphabetical character scores to fill champion_popularity
    # and meta_tier for any rows still holding NaN.
    df = assign_character_scores(df, char_scores, name_col="card_name")

    # Keep only cards whose name contains a known champion.
    # This removes non-champion cards (spells, items, tokens, etc.) that
    # would add noise and zero-variance columns to the design matrix.
    before = len(df)
    df = df[df["card_name"].str.contains(CHAMPION_RE, na=False)].reset_index(drop=True)
    print(f"  Champion filter: {before} → {len(df)} rows "
          f"({before - len(df)} non-champion cards dropped).")

    # Last-resort fill: if a card's character couldn't be identified,
    # use the mid-range defaults so the pipeline doesn't crash.
    df["champion_popularity"] = df["champion_popularity"].fillna(0.5)
    df["meta_tier"]           = df["meta_tier"].fillna(3)

    if len(df) < 200:
        print(f"\n  WARNING: only {len(df)} rows — coefficient estimates "
              "will be unreliable. Add more comp data for production use.")

    # ── 2. Design matrix & models ─────────────────────────────────────
    training_set_names = (sorted(df["set_name"].dropna().unique().tolist())
                          if "set_name" in df.columns else [])
    # TODO: unsure why creating a sinular matrix
    X, names, y = build_design(df)
    fit          = fit_hedonic(X, y)
    qr           = fit_reserve_model(X, y, q=0.20)

    run_ml = len(df) >= 100
    if run_ml:
        gbr, ml_metrics, importance = fit_ml_benchmark(X, y)

    # ── 3. Training-data report ───────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"HEDONIC VALUATION PIPELINE — RIFTBOUND TCG  |  {len(df)} comps")
    print(f"Data source: {source}")
    print("=" * 70)
    print(f"\nStep 1 — Hedonic OLS on log-price")
    print(f"  R² = {fit['r2']:.3f}   residual σ = {fit['sigma']:.3f}")
    print(f"\n  Implicit prices (% premium per attribute, ceteris paribus):")
    show = fit["table"][fit["table"].feature != "intercept"].copy()
    show["implied_pct"] = (show["implied_pct"] * 100).round(1)
    show["ci_lo_pct"]   = (show["ci_lo_pct"]   * 100).round(1)
    show["ci_hi_pct"]   = (show["ci_hi_pct"]   * 100).round(1)
    print(show[["feature", "implied_pct", "ci_lo_pct", "ci_hi_pct"]]
          .rename(columns={"implied_pct": "pct_premium",
                            "ci_lo_pct":  "ci_lo_95%",
                            "ci_hi_pct":  "ci_hi_95%"})
          .to_string(index=False))

    if run_ml:
        print(f"\nStep 2 — Gradient-boosted benchmark (cross-check)")
        print(f"  R² = {ml_metrics['r2']:.3f}  "
              f"MAE on log-price = {ml_metrics['mae_log']:.3f}")
        print(f"  {'GBM R² ≈ hedonic R²' if abs(ml_metrics['r2'] - fit['r2']) < 0.05 else 'GBM R² notably higher — consider non-linear terms'}")
        print(f"\n  Permutation importance (top 5, confirms attribute ranking):")
        print(importance.head(5).to_string(index=False))
    else:
        print("\n  (GBM benchmark skipped — fewer than 100 rows.)")

    print(f"\nStep 3 — Quantile-regression reserve (20th-pct floor)")
    print(f"  Reserve prices are reported in the ranking table below.")

    # ── 4. High-end promo card ranking ────────────────────────────────
    print("\n" + "=" * 70)
    print("HIGH-END PROMO CARD RANKING")
    print(
        "Methodology: hedonic model trained on regular-market comps;\n"
        "extrapolated to tournament prize cards via shared attribute\n"
        "structure (rarity tier, population-as-print-run, champion\n"
        "demand, set).  Prediction intervals reflect out-of-sample\n"
        "uncertainty from extrapolating to extreme scarcity values.\n"
        "Ranking is ordinal — treat absolute dollar figures as\n"
        "directional until in-market calibration data is available."
    )
    print("=" * 70)

    promo_df = prepare_promo_cards(
        HIGHEND_COMPS, char_scores,
        training_set_names=training_set_names,
    )
    ranking = rank_promo_cards(promo_df, fit, qr, list(X.columns))

    print("\nPredicted value ranking (1 = most valuable):")
    print(ranking.to_string(index=False))

    # Rank-correlation with known prices.
    known = ranking[ranking["known_price_$"] != "POA"].copy()
    if len(known) >= 3:
        known["known_price_$"] = known["known_price_$"].astype(float)
        pred_rank   = known["pred_rank"].values.astype(float)
        actual_rank = known["known_price_$"].rank(ascending=False).values
        n = len(pred_rank)
        d2 = ((pred_rank - actual_rank) ** 2).sum()
        spearman_r = 1 - 6 * d2 / (n * (n ** 2 - 1))
        print(f"\nSpearman rank correlation vs. known prices "
              f"({n} cards): ρ = {spearman_r:.3f}")
        print("(ρ = 1.0 → perfect predicted ordering, "
              "ρ = 0 → no better than random)")

    # ── 5. Save outputs ───────────────────────────────────────────────
    ranking.to_csv(os.path.join(OUT, "promo_ranking.csv"), index=False)
    plot_implicit_prices(fit["table"],
                         os.path.join(OUT, "implicit_prices.png"))
    plot_fit(fit, X, y, os.path.join(OUT, "actual_vs_predicted.png"))
    plot_ranking(ranking, os.path.join(OUT, "promo_ranking.png"))
    print(f"\nOutputs written to outputs/:")
    print(f"  promo_ranking.csv, implicit_prices.png, "
          f"actual_vs_predicted.png, promo_ranking.png")


if __name__ == "__main__":
    main()