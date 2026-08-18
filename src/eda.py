import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURE_DIR


def run_eda(df):
    """Generate reproducible core EDA figures."""

    # Load over time
    plt.figure(figsize=(14, 5))
    plt.plot(
        df["Timestamp_UTC"],
        df["Load [MW]"],
        linewidth=0.7,
    )
    plt.title("German Electricity Load Over Time")
    plt.xlabel("Time")
    plt.ylabel("Load (MW)")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "01_load_timeseries.png",
        dpi=300,
    )
    plt.close()

    # Load distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(
        df["Load [MW]"].dropna(),
        bins=60,
        kde=True,
    )
    plt.title("Electricity Load Distribution")
    plt.xlabel("Load (MW)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "02_load_distribution.png",
        dpi=300,
    )
    plt.close()

    # Average load by hour
    hourly = df.groupby(
        "hour",
        observed=True,
    )["Load [MW]"].mean()

    plt.figure(figsize=(9, 5))
    plt.plot(
        hourly.index,
        hourly.values,
        marker="o",
    )
    plt.title("Average Electricity Load by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Load (MW)")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "03_hourly_load.png",
        dpi=300,
    )
    plt.close()

    # Renewable coverage over time
    plt.figure(figsize=(14, 5))
    plt.plot(
        df["Timestamp_UTC"],
        df["renewable_coverage_calc_pct"],
        linewidth=0.7,
    )
    plt.title("Renewable Coverage Over Time")
    plt.xlabel("Time")
    plt.ylabel("Renewable Coverage (%)")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "04_renewable_coverage.png",
        dpi=300,
    )
    plt.close()

    # Load vs renewable coverage
    sample = df[
        ["Load [MW]", "renewable_coverage_calc_pct"]
    ].dropna()

    if len(sample) > 20000:
        sample = sample.sample(
            20000,
            random_state=42,
        )

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=sample,
        x="Load [MW]",
        y="renewable_coverage_calc_pct",
        s=12,
        alpha=0.35,
    )
    plt.title(
        "Electricity Load vs Renewable Coverage"
    )
    plt.xlabel("Load (MW)")
    plt.ylabel("Renewable Coverage (%)")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "05_load_vs_renewable_coverage.png",
        dpi=300,
    )
    plt.close()
