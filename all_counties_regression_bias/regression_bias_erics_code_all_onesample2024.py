"""
Monte Carlo noise simulation for assessment regressivity metrics.

Python translation of monte_carlo_graphs.R from Eric's code (cmfproperty
package, Center for Municipal Finance, UChicago), with the Gini and Suits
measures from McMillen & Singh (2023) added.

THIS VERSION: runs on the First American national data (one parquet file per
county) and loops over every county. Parts 1 and 2 are unchanged. Part 3
(loading) and the RUN IT section are new.

THE IDEA IN ONE PARAGRAPH
Pretend the assessor got every home exactly right, so each home's true value
equals its assessed value. Then invent fake sale prices by adding random
"luck" (noise) to those values: 0%, 1%, 2% ... up to 25%. At each noise level,
score the fake data with every regressivity metric. Because the assessments
are perfect by construction, any regressivity the metrics report is fake,
caused purely by noise. Finally, compare each metric's score on the REAL data
to these fake curves: "how much noise would it take for luck alone to produce
a score this bad?"

HOW TO USE (one dataset)
    df = load_data(...)                 # needs SALE_PRICE and ASSESSED_VALUE
    avg = monte_carlo_sim(df, iters=10) # the fake experiment
    real = compute_all_metrics(df)      # scores on the real data
    implied = implied_noise(avg, real)  # where real scores land on the curves
    make_graphs(avg, real, implied)     # one graph per metric

Differences from the R version are marked "NOTE (diff from R)".

IAAO vertical equity measures (2026 exposure draft, section 8.2):
    VEI  primary measure (Appendix E)   -> vei(), vei_details()   NEW
    PRB  the regression measure         -> prb()
    MKI  Modified Kakwani Index         -> mki()                  NEW
    PRD  price-related differential     -> prd()
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save graphs to files without opening windows
import matplotlib.pyplot as plt


# =============================================================================
# PART 1: THE METRICS ("the rulers")
# Each function takes arrays of numbers and returns one score.
# ratio = assessed value / sale price. Ratio of 1.0 = assessed exactly right.
# =============================================================================

def median_ratio(ratio):
    """LEVEL: the typical ratio. Is the roll centred at all?
    Ideal 1.0 (IAAO band 0.90 to 1.10)."""
    return np.median(ratio)


def cod(ratio):
    """UNIFORMITY (not regressivity): average % gap between each home's
    ratio and the median ratio. Measures scatter, regardless of price.
    Ideal 0 (IAAO band for single-family homes: 5 to 15)."""
    med = np.median(ratio)
    return 100 * np.mean(np.abs(ratio - med)) / med


def prd(ratio, sale_price):
    """VERTICAL EQUITY: plain average ratio / dollar-weighted average ratio.
    The dollar-weighted average is dominated by expensive homes, so if cheap
    homes have higher ratios the plain average is bigger and PRD > 1.
    Ideal 1.0. Above 1.03 = regressive, below 0.98 = progressive."""
    return np.mean(ratio) / np.average(ratio, weights=sale_price)


def prb(ratio, assessed_value, sale_price):
    """VERTICAL EQUITY: slope of a line through (value, ratio).
    y = % gap of each ratio from the median ratio
    x = log base 2 of a value estimate (average of sale price and assessed
        value rescaled by the median ratio), so a 1-unit step in x means
        "value doubled".
    Read it as: each time value doubles, the ratio changes by PRB (as a %).
    Ideal 0. Negative = regressive. IAAO band -0.05 to +0.05."""
    med = np.median(ratio)
    y = (ratio - med) / med
    x = np.log(0.5 * (sale_price + assessed_value / med)) / np.log(2)
    slope, _ = np.polyfit(x, y, 1)
    return slope


def paglin72(assessed_value, sale_price):
    """Paglin & Fogarty (1972): slope from regressing assessed value on sale
    price, in dollars. (The R code keeps the slope, so we do too.)"""
    slope, _ = np.polyfit(sale_price, assessed_value, 1)
    return slope


def cheng74(assessed_value, sale_price):
    """Cheng (1974): same as Paglin but on a log scale, i.e. comparing
    percentages instead of dollars. Slope of 1 = proportional; below 1 =
    regressive. (Cheng slope minus 1 = the "LC" in Daniel's intro doc.)
    The +1 inside the log matches the R code."""
    slope, _ = np.polyfit(np.log(sale_price + 1), np.log(assessed_value + 1), 1)
    return slope


def lc(assessed_value, sale_price):
    """NEW. The log regression from Daniel's intro doc ("LC"):
    regress log(assessed / sale price) on log(sale price).
    Same idea as Cheng, just shifted so fair = 0 instead of 1
    (LC is about Cheng - 1; Cheng adds +1 inside the logs, this doesn't).
    Read it as: when sale price rises 1%, the ratio changes by LC %.
    Ideal 0. Negative = regressive."""
    slope, _ = np.polyfit(np.log(sale_price), np.log(assessed_value / sale_price), 1)
    return slope


def iaao78(ratio, sale_price):
    """IAAO (1978) / Almy et al.: slope from regressing the ratio on sale
    price. Negative = ratio falls as price rises = regressive."""
    slope, _ = np.polyfit(sale_price, ratio, 1)
    return slope


def gini_diff(assessed_value, sale_price):
    """NEW (not in Eric's code), from McMillen & Singh section 4.
    Line up homes from cheapest to priciest SALE PRICE.
      G_p = how unequally sale prices are spread across those homes
      G_a = how unequally assessed values are spread, same ordering
    If assessments are regressive, cheap homes get "too much" assessed value,
    so assessments look MORE equal than prices and G_a < G_p.
    Returns G_a - G_p. Ideal 0. Negative = regressive."""
    order = np.argsort(sale_price, kind="stable")
    p = sale_price[order]
    a = assessed_value[order]
    n = len(p)
    i = np.arange(1, n + 1)  # rank 1 = cheapest sale

    def g(x):  # Gini formula from the paper: 2*sum(i*x)/(n*sum(x)) - (n+1)/n
        return 2 * np.sum(i * x) / (n * np.sum(x)) - (n + 1) / n

    return g(a) - g(p)


def mki(assessed_value, sale_price):
    """NEW. IAAO Modified Kakwani Index (2026 draft, Appendix F; Quintos 2020).
    Same two Gini pieces as gini_diff, but divided instead of subtracted:
      MKI = G_a / G_p
    Ideal 1. Below 1 = regressive, above 1 = progressive.
    IAAO: 0.95-1.05 optimal (0.90-1.10 for messier or smaller samples)."""
    order = np.argsort(sale_price, kind="stable")
    p = sale_price[order]
    a = assessed_value[order]
    n = len(p)
    i = np.arange(1, n + 1)  # rank 1 = cheapest sale

    def g(x):
        return 2 * np.sum(i * x) / (n * np.sum(x)) - (n + 1) / n

    return g(a) / g(p)


def _vei_groups(n):
    """How many value groups the IAAO says to use, based on sample size."""
    if n < 20:
        return None  # too few sales: VEI not computed
    if n <= 50:
        return 2     # halves
    if n <= 500:
        return 4     # quartiles
    return 10        # deciles


def _median_ci(sorted_ratios, z=1.645):
    """90% confidence interval for a median, by rank (no bell-curve
    assumption). Takes the ratios at these positions in sorted order:
      lower: n/2 - z*sqrt(n)/2, rounded UP
      upper: 1 + n/2 + z*sqrt(n)/2, rounded DOWN
    This reproduces the IAAO worked example (Appendix E, Tables 25 and 27)."""
    n = len(sorted_ratios)
    lo = int(np.ceil(n / 2 - z * np.sqrt(n) / 2))
    hi = int(np.floor(1 + n / 2 + z * np.sqrt(n) / 2))
    lo = min(max(lo, 1), n)
    hi = min(max(hi, 1), n)
    return sorted_ratios[lo - 1], sorted_ratios[hi - 1]


def vei_details(ratio, assessed_value, sale_price, method="R7"):
    """NEW. IAAO Vertical Equity Indicator, the draft's PRIMARY measure
    (2026 draft, Appendix E). Steps:
      1. ratio = assessed / sale price
      2. value proxy = half sale price + half (assessed / median ratio).
         Using both, instead of sale price alone, reduces the noise bias.
         (The draft prints the formula without the second 0.50, but its
         worked example only matches with it, so that's a typo.)
      3. sort homes by the proxy and split into groups
         (2, 4 or 10 depending on sample size)
      4. median ratio + 90% confidence interval in each group
      5. VEI = 100 * (median of top group - median of bottom group)
                   / overall median
         Negative = regressive, positive = progressive. Within +/-10 = OK.
      6-7. If outside +/-10, check whether the top and bottom groups'
         confidence intervals overlap; if not, "VEI significance" = the
         gap between the closest ends of the two intervals, as a %.
    method: "R7" (numpy/R default) or "R6" percentile grouping; the draft
    says both are fine and they give slightly different groups.
    Returns a dict with the VEI, the significance, the outcome, and a
    table of the groups."""
    ratio = np.asarray(ratio, dtype=float)
    n = len(ratio)
    k = _vei_groups(n)
    if k is None:
        return {"VEI": np.nan, "VEI_significance": np.nan,
                "outcome": "too few sales (<20)", "groups": None}

    med = np.median(ratio)
    proxy = 0.5 * sale_price + 0.5 * assessed_value / med
    order = np.argsort(proxy, kind="stable")
    r_sorted = ratio[order]
    proxy_sorted = proxy[order]

    # Where each group ends (1-based positions), per the draft's R6 / R7
    pcts = np.arange(1, k + 1) * 100 / k
    if method == "R7":
        ends = [int(np.floor(p / 100 * (n - 1))) + 1 for p in pcts]
    else:
        ends = [int(np.floor(p / 100 * (n + 1))) for p in pcts]
    ends = [min(e, n) for e in ends]
    starts = [0] + ends[:-1]

    rows = []
    for g, (a, b) in enumerate(zip(starts, ends), 1):
        grp = np.sort(r_sorted[a:b])
        lo, hi = _median_ci(grp)
        rows.append({"group": g, "sales": len(grp),
                     "median_proxy": np.median(proxy_sorted[a:b]),
                     "median_ratio": np.median(grp),
                     "lower_ci": lo, "upper_ci": hi})
    groups = pd.DataFrame(rows)

    first, last = groups.iloc[0], groups.iloc[-1]
    vei_val = 100 * (last["median_ratio"] - first["median_ratio"]) / med

    # Steps 6-7: only needed if the point estimate is outside +/-10
    sig = np.nan
    if abs(vei_val) <= 10:
        outcome = "acceptable (VEI within +/-10)"
    else:
        high, low = (last, first) if last["median_ratio"] >= first["median_ratio"] else (first, last)
        if high["lower_ci"] <= low["upper_ci"]:
            outcome = "acceptable (confidence intervals overlap)"
        else:
            sig = 100 * (high["lower_ci"] - low["upper_ci"]) / med
            if sig <= 10:
                outcome = "acceptable (VEI significance <= 10)"
            elif vei_val < 0:
                outcome = "UNACCEPTABLE regressivity"
            else:
                outcome = "UNACCEPTABLE progressivity"

    return {"VEI": vei_val, "VEI_significance": sig,
            "outcome": outcome, "groups": groups}


def vei(ratio, assessed_value, sale_price, method="R7"):
    """Just the VEI number (what the simulation uses)."""
    return vei_details(ratio, assessed_value, sale_price, method)["VEI"]


def suits(assessed_value, sale_price):
    """NEW (not in Eric's code), from McMillen & Singh section 4.
    Line up homes from cheapest to priciest sale price. Plot:
      x = running % of total sale price
      y = running % of total assessed value
    Fair assessments give a 45-degree line. If cheap homes are over-assessed,
    they "use up" assessed value faster, so the curve bows ABOVE the line.
    S = 1 - (area under curve) / (area under 45-degree line = 5000).
    Ideal 0. Negative = regressive."""
    order = np.argsort(sale_price, kind="stable")
    p = sale_price[order]
    a = assessed_value[order]
    x = np.concatenate([[0], 100 * np.cumsum(p) / p.sum()])
    y = np.concatenate([[0], 100 * np.cumsum(a) / a.sum()])
    area = np.trapezoid(y, x)  # trapezoid rule, same as the paper
    return 1 - area / 5000


METRICS = ["median_ratio", "COD", "PRD", "PRB", "VEI", "MKI",
           "paglin72", "cheng74", "LC", "IAAO78", "gini_diff", "suits"]


def compute_all_metrics(df):
    """Score one dataset with every metric. This plays the role of R's
    get_stats() + paglin_cheng_IAAO_coefs() combined.

    NOTE (diff from R): no bootstrap. In R, get_stats reshuffles the data
    several times to get margins of error. The simulation never uses those,
    and skipping them is the main speed-up."""
    a = df["ASSESSED_VALUE"].to_numpy(dtype=float)
    p = df["SALE_PRICE"].to_numpy(dtype=float)
    r = a / p
    return {
        "median_ratio": median_ratio(r),
        "COD": cod(r),
        "PRD": prd(r, p),
        "PRB": prb(r, a, p),
        "VEI": vei(r, a, p),
        "MKI": mki(a, p),
        "paglin72": paglin72(a, p),
        "cheng74": cheng74(a, p),
        "LC": lc(a, p),
        "IAAO78": iaao78(r, p),
        "gini_diff": gini_diff(a, p),
        "suits": suits(a, p),
    }


def bootstrap_se(df, bootstrap_iters, rng):
    """The loop from Eric's R code, added back. Mirrors R's get_stats():
    reshuffle the homes `bootstrap_iters` times, recompute COD and PRD each
    time, and report how much they bounce around (standard error = margin
    of error). PRB's margin of error comes straight from its regression,
    like in R.
    This does NOT change how many simulated samples are run, and it does
    NOT change the metric values themselves. It only adds _SE columns."""
    a = df["ASSESSED_VALUE"].to_numpy(dtype=float)
    p = df["SALE_PRICE"].to_numpy(dtype=float)
    r = a / p
    n = len(r)
    cods, prds = [], []
    for _ in range(bootstrap_iters):
        idx = rng.integers(0, n, n)  # draw n homes WITH replacement
        cods.append(cod(r[idx]))
        # Same as R's prd_func: reshuffled ratios, ORIGINAL sale-price
        # weights (looks like a small bug in R; kept so results match)
        prds.append(np.mean(r[idx]) / np.average(r[idx], weights=p))

    # PRB standard error from the regression itself (R: summary(lm)$coef)
    med = np.median(r)
    y = (r - med) / med
    x = np.log(0.5 * (p + a / med)) / np.log(2)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    prb_se = np.sqrt(np.sum(resid ** 2) / (n - 2) / np.sum((x - x.mean()) ** 2))

    return {"COD_SE": np.std(cods, ddof=1) if bootstrap_iters > 1 else np.nan,
            "PRD_SE": np.std(prds, ddof=1) if bootstrap_iters > 1 else np.nan,
            "PRB_SE": prb_se}


# =============================================================================
# PART 2: THE SIMULATION ("the experiment")
# =============================================================================

def monte_carlo_sim(df, iters=10, max_noise=0.25, step=0.01, seed=0,
                    bootstrap_iters=5):
    """The pretend experiment. Translation of R's monte_carlo_sim().

    For each noise level (0%, 1%, ... 25%):
      1. Treat each home's assessed value as its TRUE value (perfect assessor)
      2. Draw a random "luck" number per home from a bell curve centred on 0,
         whose spread is the noise level (e.g. 0.05 = typical luck of +/-5%)
      3. Fake sale price = assessed value * (1 + luck)
      4. Score the fake data with every metric
    Repeat `iters` times per level (luck is random, so runs differ), then
    average. Returns one row per noise level.

    NOTE (diff from R): R uses iters=2, which gives bumpy curves. Default here
    is 10; raise it if the curves look jagged.
    NOTE (diff from R): `seed` makes the random draws repeatable, so you get
    the same answer every time you run it.

    HOW MANY SAMPLES ARE RUN: (number of noise levels) x iters.
    0% to 25% in 1% steps = 26 noise levels, so:
        iters=1  ->  26 simulated samples (one per noise level)
        iters=10 -> 260 simulated samples
    bootstrap_iters is Eric's inner loop (R: get_stats(cur, 5)). It runs
    INSIDE each sample and only adds margins of error (the _SE columns),
    so it doesn't change the sample count or the curves. Set it to 0 to
    skip it and run faster.
    """
    rng = np.random.default_rng(seed)
    # Separate random stream for the bootstrap, so turning it on or off
    # never changes the noise draws (results stay identical either way)
    boot_rng = np.random.default_rng(seed + 1)

    # Drop rows with tiny values (likely data errors), same as R
    df = df[(df["SALE_PRICE"] > 100) & (df["ASSESSED_VALUE"] > 100)]
    av = df["ASSESSED_VALUE"].to_numpy(dtype=float)

    results = []
    noise_levels = np.round(np.arange(0, max_noise + step / 2, step), 4)
    for noise in noise_levels:
        for it in range(iters):
            shock = rng.normal(0, noise, size=len(av))  # the random luck

            # Throw out impossible draws (luck of -100% or worse would make
            # the price zero or negative). Same as R: abs(shock) < 1.
            keep = np.abs(shock) < 1
            fake = pd.DataFrame({
                "ASSESSED_VALUE": av[keep],
                "SALE_PRICE": av[keep] * (1 + shock[keep]),  # fake sale price
            })

            row = compute_all_metrics(fake)
            if bootstrap_iters > 0:  # Eric's inner loop
                row.update(bootstrap_se(fake, bootstrap_iters, boot_rng))
            row["noise"] = noise
            row["iter"] = it
            results.append(row)

    # NOTE (diff from R): R grows its results table inside the loop with
    # rbind, which is slow. Collecting a list and building once is faster.
    all_results = pd.DataFrame(results)

    # Average across iterations: one row per noise level
    # (the _SE columns, if computed, are averaged too)
    cols = METRICS + [c for c in ["COD_SE", "PRD_SE", "PRB_SE"]
                      if c in all_results.columns]
    return all_results.groupby("noise")[cols].mean().reset_index()


def implied_noise(avg, real):
    """The key output: for each metric, how much noise would it take for
    pure luck to produce the score we see in the REAL data?

    Works like reading a graph backwards: find the real score on the y-axis
    of the fake curve, go across to the curve, then down to the noise level.
    Translation of R's approxfun(...) lines.

    Returns NaN if the real score is worse than even 25% noise produces
    (R prints this as ">0.25"). Only meaningful for metrics that move steadily
    in one direction as noise rises; median_ratio barely moves, so ignore it.
    """
    out = {}
    for m in METRICS:
        x = avg[m].to_numpy()
        y = avg["noise"].to_numpy()
        order = np.argsort(x)  # np.interp needs the x values sorted
        out[m] = np.interp(real[m], x[order], y[order],
                           left=np.nan, right=np.nan)
    return out


def make_graphs(avg, real, implied, path="monte_carlo_graphs.png"):
    """One small graph per metric. Translation of R's ggplot section.
      curve           = average fake score at each noise level
      horizontal line = score on the REAL data
      vertical line   = implied noise (where the two cross)"""
    fig, axes = plt.subplots(4, 3, figsize=(13, 14))
    for ax in axes.flat[len(METRICS):]:
        ax.axis("off")  # hide the unused box
    for ax, m in zip(axes.flat, METRICS):
        ax.plot(avg["noise"] * 100, avg[m], marker="o", lw=2)
        ax.axhline(real[m], color="gray", ls="--", label="real data")
        if not np.isnan(implied[m]):
            ax.axvline(implied[m] * 100, color="red", ls=":",
                       label=f"implied noise {implied[m]:.1%}")
        else:
            ax.set_title(f"{m} (implied noise > 25%)")
        ax.set_title(ax.get_title() or m)
        ax.set_xlabel("Noise (%)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig


# =============================================================================
# PART 3: LOADING DATA (NEW: First American, one file per county)
# =============================================================================

def trim_outliers(df):
    """Drop extreme ratios within each year: anything more than 1.5 x IQR
    beyond the 25th/75th percentile. Matches cmfproperty's reformat_data().

    (reformat_data() also makes inflation-adjusted copies of the columns,
    but the Monte Carlo code never uses them, so they're skipped here.)"""
    r = df["ASSESSED_VALUE"] / df["SALE_PRICE"]
    by_year = r.groupby(df["SALE_YEAR"])
    q1 = by_year.transform(lambda s: s.quantile(0.25))  # each row gets its
    q3 = by_year.transform(lambda s: s.quantile(0.75))  # own year's cutoffs
    iqr = q3 - q1
    return df[(r >= q1 - 1.5 * iqr) & (r <= q3 + 1.5 * iqr)]


def load_county(path, year, trim=True):
    """NEW. Load one county's residential sales from a First American file.

    - One file per county, named by its FIPS code (17031.parquet = Cook)
    - MarketTotalValue is the assessor's MARKET value estimate, so no x10
      needed. (The "Value" column is the assessed value after each state's
      assessment rate, e.g. 10% in Cook, so we don't use it.)
    - PropertyClassID "R" = residential. This is broader than CCAO's
      regression classes (it probably includes condos), because this data
      doesn't have detailed class codes.
    - NOTE: this data has no arm's-length flag, so junk sales (family
      transfers etc.) are still in. Trimming does more of the cleaning here
      than it did with base.parquet.

    Returns (data, state abbreviation)."""
    df = pd.read_parquet(path, columns=["Year", "PropertyClassID",
                                        "MarketTotalValue", "SaleAmt",
                                        "SitusState"])
    df = df[(df["Year"] == year) & (df["PropertyClassID"] == "R")]

    states = df["SitusState"].dropna()
    state = states.mode().iat[0] if len(states) else ""

    df = pd.DataFrame({
        "SALE_YEAR": df["Year"],
        "SALE_PRICE": df["SaleAmt"].astype(float),
        "ASSESSED_VALUE": df["MarketTotalValue"].astype(float),
    }).dropna()
    df = df[(df["SALE_PRICE"] > 100) & (df["ASSESSED_VALUE"] > 100)]
    if trim and len(df):
        df = trim_outliers(df)
    return df, state


# The columns of the results table, in a fixed order, so every county's row
# lines up even when a county is skipped
RESULT_COLS = (["fips", "state", "n_sales", "status", "VEI_outcome"]
               + [f"{m}_real" for m in METRICS]
               + [f"{m}_implied_noise" for m in METRICS])


def run_county(path, year, trim, iters, bootstrap, min_sales):
    """NEW. Everything the old RUN IT section did, for one county.
    Returns (one results row, that county's simulation curves)."""
    fips = path.stem
    df, state = load_county(path, year, trim)
    row = {"fips": fips, "state": state, "n_sales": len(df)}

    if len(df) == 0:
        row["status"] = f"no residential sales in {year}"
        return row, None
    if len(df) < min_sales:
        row["status"] = f"skipped (fewer than {min_sales} sales)"
        return row, None

    real = compute_all_metrics(df)
    avg = monte_carlo_sim(df, iters=iters, bootstrap_iters=bootstrap)
    implied = implied_noise(avg, real)
    d = vei_details(df["ASSESSED_VALUE"] / df["SALE_PRICE"],
                    df["ASSESSED_VALUE"].to_numpy(float),
                    df["SALE_PRICE"].to_numpy(float))

    row["status"] = "ok"
    row["VEI_outcome"] = d["outcome"]
    for m in METRICS:
        row[f"{m}_real"] = real[m]
        row[f"{m}_implied_noise"] = implied[m]

    avg.insert(0, "fips", fips)
    return row, avg, real, implied


# =============================================================================
# RUN IT (NEW: loops over every county)
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path

    # ---- Settings: change these ------------------------------------------
    # Folder with one parquet file per county
    DATA_DIR = (Path.home() / "regression_bias/first_american_data"
                / "First American Data Joined")
    YEAR = 2025       # which year of sales to use
    TRIM = True       # True = drop extreme ratios like Eric's code
    ITERS = 1         # simulated samples per noise level
    BOOTSTRAP = 5     # Eric's inner loop (5 = same as his R code; 0 = skip)
    MIN_SALES = 100   # skip counties with fewer sales than this
    GRAPH_COUNTIES = ["17031"]  # FIPS codes to also save graphs for
    TEST_RUN = None   # e.g. 5 = only do the first 5 counties (to test);
                      # None = all counties

    # Small counties can make the regressions throw math warnings; they
    # don't stop anything, so hide them to keep the progress output readable
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=np.exceptions.RankWarning)

    # ---- Where results go ---------------------------------------------------
    # Folder name includes the settings, so different runs don't mix
    tag = f"{YEAR}_{'trim' if TRIM else 'notrim'}_iters{ITERS}"
    out_dir = Path(f"results_{tag}")
    out_dir.mkdir(exist_ok=True)
    results_file = out_dir / "county_results.csv"      # one row per county
    curves_file = out_dir / "county_sim_curves.csv"    # 26 rows per county

    # ---- Resume: skip counties already finished -----------------------------
    # Results are saved after EVERY county, so if the run crashes or you stop
    # it, just run the script again and it picks up where it left off
    done = set()
    if results_file.exists():
        done = set(pd.read_csv(results_file, dtype={"fips": str})["fips"])

    files = sorted(DATA_DIR.glob("*.parquet"))
    if TEST_RUN:
        files = files[:TEST_RUN]
    print(f"{len(files)} county files found, {len(done)} already done\n")

    # ---- The loop -------------------------------------------------------------
    for i, path in enumerate(files, 1):
        fips = path.stem
        if fips in done:
            continue

        avg = None
        try:
            out = run_county(path, YEAR, TRIM, ITERS, BOOTSTRAP, MIN_SALES)
            row = out[0]
            if row["status"] == "ok":
                row, avg, real, implied = out
        except Exception as e:
            # One broken county shouldn't stop the whole run
            row = {"fips": fips, "status": f"error: {e}"}

        # Save this county's row right away
        pd.DataFrame([row]).reindex(columns=RESULT_COLS).to_csv(
            results_file, mode="a", header=not results_file.exists(),
            index=False)
        if avg is not None:
            avg.to_csv(curves_file, mode="a",
                       header=not curves_file.exists(), index=False)
            if fips in GRAPH_COUNTIES:
                fig = make_graphs(avg, real, implied,
                                  path=out_dir / f"graphs_{fips}.png")
                plt.close(fig)  # free memory inside the loop

        print(f"[{i}/{len(files)}] {fips} {row.get('state', '')}: "
              f"{row['status']} (n={row.get('n_sales', '?')})")

    # ---- Quick summary at the end -------------------------------------------
    res = pd.read_csv(results_file, dtype={"fips": str})
    ok = res[res["status"] == "ok"]
    print(f"\n{len(ok)} counties ran, {len(res) - len(ok)} skipped or errored")
    if len(ok):
        print("\nMedian implied noise across counties, by metric:")
        print(ok[[f"{m}_implied_noise" for m in METRICS]]
              .median().round(4).to_string())
    print(f"\nSaved to {out_dir}/")