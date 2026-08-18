import pandas as pd

RENEWABLE_COMPONENTS = [
    "Solar [MW]",
    "Wind onshore [MW]",
    "Wind offshore [MW]",
    "Hydro Run-of-River [MW]",
    "Hydro water reservoir [MW]",
]


def calculate_renewable_metrics(df):
    """Calculate renewable total, coverage and renewable-generation gap."""
    df = df.copy()

    available = [
        col for col in RENEWABLE_COMPONENTS
        if col in df.columns
    ]

    if not available:
        raise KeyError("No renewable-generation component columns found.")

    df["renewable_total_calc_MW"] = df[available].sum(
        axis=1, min_count=1
    )

    load = df["Load [MW]"].replace(0, pd.NA)

    df["renewable_coverage_calc_pct"] = (
        df["renewable_total_calc_MW"] / load
    ) * 100

    df["renewable_gap_calc_MW"] = (
        df["Load [MW]"] - df["renewable_total_calc_MW"]
    )

    return df


def add_four_conditions(
    df,
    high_demand_threshold,
    low_renewable_threshold,
):
    """Classify observations into the four demand-renewable conditions."""
    df = df.copy()

    df["demand_level"] = (
        df["Load [MW]"] >= high_demand_threshold
    ).map({True: "High", False: "Low"})

    df["renewable_level"] = (
        df["renewable_coverage_calc_pct"] < low_renewable_threshold
    ).map({True: "Low", False: "High"})

    mapping = {
        ("Low", "High"): "Low Demand - High Renewable",
        ("Low", "Low"): "Low Demand - Low Renewable",
        ("High", "High"): "High Demand - High Renewable",
        ("High", "Low"): "High Demand - Low Renewable",
    }

    df["condition"] = [
        mapping[(d, r)]
        for d, r in zip(
            df["demand_level"],
            df["renewable_level"],
        )
    ]

    return df


def identify_reliability_windows(
    df,
    high_demand_quantile=0.90,
    low_renewable_quantile=0.25,
):
    """
    Identify potentially critical windows where high demand coincides
    with low renewable coverage.
    """
    df = df.copy()

    high_demand = df["Load [MW]"].quantile(
        high_demand_quantile
    )
    low_renewable = df["renewable_coverage_calc_pct"].quantile(
        low_renewable_quantile
    )

    df["potentially_critical"] = (
        (df["Load [MW]"] >= high_demand)
        & (
            df["renewable_coverage_calc_pct"]
            < low_renewable
        )
    )

    change = df["potentially_critical"].ne(
        df["potentially_critical"].shift()
    )
    df["condition_group"] = change.cumsum()

    windows = []

    for _, group in df[df["potentially_critical"]].groupby(
        "condition_group"
    ):
        windows.append({
            "start": group["Timestamp_UTC"].min(),
            "end": group["Timestamp_UTC"].max(),
            "intervals": len(group),
            "duration_hours": len(group) * 0.25,
            "mean_load_mw": group["Load [MW]"].mean(),
            "max_load_mw": group["Load [MW]"].max(),
            "mean_renewable_coverage_pct": (
                group["renewable_coverage_calc_pct"].mean()
            ),
            "mean_renewable_gap_mw": (
                group["renewable_gap_calc_MW"].mean()
            ),
        })

    return (
        df,
        pd.DataFrame(windows),
        {
            "high_demand_threshold_mw": float(high_demand),
            "low_renewable_threshold_pct": float(low_renewable),
        },
    )
