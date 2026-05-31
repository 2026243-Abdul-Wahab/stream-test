"""
Ireland Agriculture Data Analytics Dashboard
-------------------------------------------
A Streamlit dashboard connected to the MySQL database created by the notebook.

Default database credentials match the notebook:
    host: localhost
    port: 3306
    user: classpractice
    password: "Wahab@9244"
    database: ireland_agriculture_db

You can override them with environment variables:
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Ireland Agriculture Dashboard",
    page_icon="🌾",
    layout="wide",
)

COUNTRIES = ["Ireland", "United Kingdom", "France", "Germany", "Netherlands", "Denmark"]
DATA_DIR = Path("data")

COLOURS = {
    "Ireland": "#1a9641",
    "United Kingdom": "#2166ac",
    "France": "#d7191c",
    "Germany": "#f4a442",
    "Netherlands": "#762a83",
    "Denmark": "#8c510a",
}


# -----------------------------------------------------------------------------
# Database helpers
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_engine() -> Tuple[Optional[Engine], str, Optional[str]]:
    """Create a database connection using the same credentials as the notebook."""
    mysql_host = os.getenv("MYSQL_HOST", "localhost")
    mysql_port = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_user = os.getenv("MYSQL_USER", "root")
    mysql_password = os.getenv("MYSQL_PASSWORD", "")
    mysql_database = os.getenv("MYSQL_DATABASE", "ireland_agriculture_db")

    mysql_url = (
        f"mysql+pymysql://{mysql_user}:{quote_plus(mysql_password)}"
        f"@{mysql_host}:{mysql_port}/{mysql_database}?charset=utf8mb4"
    )

    try:
        engine = create_engine(mysql_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine, "MySQL", None
    except Exception as exc:
        # Optional local fallback, useful if the notebook used SQLite fallback.
        sqlite_path = DATA_DIR / "ireland_agriculture.db"
        if sqlite_path.exists():
            try:
                engine = create_engine(f"sqlite:///{sqlite_path}", pool_pre_ping=True)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return engine, "SQLite fallback", f"MySQL was unavailable: {type(exc).__name__}: {exc}"
            except Exception as sqlite_exc:
                return None, "No database", f"MySQL error: {exc}; SQLite error: {sqlite_exc}"
        return None, "No database", f"MySQL was unavailable: {type(exc).__name__}: {exc}"


@st.cache_data(show_spinner=False)
def list_tables_cached(_engine: Engine) -> list[str]:
    return inspect(_engine).get_table_names()


def has_table(tables: Iterable[str], table_name: str) -> bool:
    return table_name in set(tables)


@st.cache_data(show_spinner=False)
def read_sql_cached(query: str, _engine: Engine, params: Optional[dict] = None) -> pd.DataFrame:
    return pd.read_sql(text(query), _engine, params=params or {})


@st.cache_data(show_spinner=False)
def read_table_cached(table_name: str, _engine: Engine) -> pd.DataFrame:
    return pd.read_sql_table(table_name, _engine)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise expected FAOSTAT/notebook columns into dashboard-friendly names."""
    if df.empty:
        return df

    rename_map = {
        "Area": "country",
        "Item": "commodity",
        "Element": "element",
        "Year": "year",
        "Unit": "unit",
        "Value": "value",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()

    if "year" in df.columns:
        df["year"] = safe_numeric(df["year"]).astype("Int64")
    if "value" in df.columns:
        df["value"] = safe_numeric(df["value"])

    return df


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_production_data(_engine: Optional[Engine], tables: tuple[str, ...]) -> pd.DataFrame:
    """Load cleaned production data if available, otherwise build the dashboard slice from raw FAOSTAT."""
    if _engine is not None and has_table(tables, "production_clean"):
        df = read_table_cached("production_clean", _engine)
        return normalise_columns(df)

    if _engine is not None and has_table(tables, "faostat_production"):
        query = """
            SELECT Area, Item, Element, Year, Unit, Value
            FROM faostat_production
            WHERE Area IN :countries
        """
        # SQLAlchemy expanding parameters are awkward with pd.read_sql text and MySQL,
        # so use a safe quoted list from the fixed COUNTRIES constant.
        country_list = ", ".join([f"'{c.replace(chr(39), chr(39)+chr(39))}'" for c in COUNTRIES])
        query = f"""
            SELECT Area, Item, Element, Year, Unit, Value
            FROM faostat_production
            WHERE Area IN ({country_list})
        """
        df = read_sql_cached(query, _engine)
        return normalise_columns(df)

    # CSV fallback, if user exports from notebook.
    csv_path = DATA_DIR / "production_clean.csv"
    if csv_path.exists():
        return normalise_columns(pd.read_csv(csv_path))

    return pd.DataFrame(columns=["country", "commodity", "element", "year", "unit", "value"])


@st.cache_data(show_spinner=True)
def load_trade_balance(_engine: Optional[Engine], tables: tuple[str, ...]) -> pd.DataFrame:
    """Load or calculate trade balance, using case-insensitive Export value / Import value detection."""
    if _engine is not None and has_table(tables, "trade_balance"):
        df = read_table_cached("trade_balance", _engine)
        df = normalise_columns(df)
        for col in ["export_val", "import_val", "trade_balance"]:
            if col in df.columns:
                df[col] = safe_numeric(df[col])
        return df

    csv_path = DATA_DIR / "trade_balance.csv"
    if csv_path.exists():
        df = normalise_columns(pd.read_csv(csv_path))
        for col in ["export_val", "import_val", "trade_balance"]:
            if col in df.columns:
                df[col] = safe_numeric(df[col])
        return df

    if _engine is not None and has_table(tables, "faostat_trade"):
        country_list = ", ".join([f"'{c.replace(chr(39), chr(39)+chr(39))}'" for c in COUNTRIES])
        df = read_sql_cached(
            f"""
            SELECT Area, Item, Element, Year, Unit, Value
            FROM faostat_trade
            WHERE Area IN ({country_list})
            """,
            _engine,
        )
        df = normalise_columns(df)
        if df.empty or "element" not in df.columns:
            return pd.DataFrame(columns=["country", "year", "export_val", "import_val", "trade_balance"])

        elements = df["element"].dropna().astype(str).str.strip().unique().tolist()
        export_element = next((e for e in elements if "export" in e.lower() and "value" in e.lower()), None)
        import_element = next((e for e in elements if "import" in e.lower() and "value" in e.lower()), None)

        if export_element is None or import_element is None:
            return pd.DataFrame(columns=["country", "year", "export_val", "import_val", "trade_balance"])

        trade_agg = (
            df[df["element"].isin([export_element, import_element])]
            .groupby(["country", "year", "element"], dropna=False)["value"]
            .sum()
            .unstack("element", fill_value=0)
            .reset_index()
        )
        trade_agg.columns.name = None
        trade_agg = trade_agg.rename(columns={export_element: "export_val", import_element: "import_val"})
        for col in ["export_val", "import_val"]:
            if col not in trade_agg.columns:
                trade_agg[col] = 0.0
        trade_agg["trade_balance"] = trade_agg["export_val"] - trade_agg["import_val"]
        return trade_agg

    return pd.DataFrame(columns=["country", "year", "export_val", "import_val", "trade_balance"])


@st.cache_data(show_spinner=True)
def load_master_data(_engine: Optional[Engine], tables: tuple[str, ...]) -> pd.DataFrame:
    if _engine is not None and has_table(tables, "master_dataset"):
        return normalise_columns(read_table_cached("master_dataset", _engine))
    csv_path = DATA_DIR / "master_dataset.csv"
    if csv_path.exists():
        return normalise_columns(pd.read_csv(csv_path))
    return pd.DataFrame()


@st.cache_data(show_spinner=True)
def load_sentiment_data(_engine: Optional[Engine], tables: tuple[str, ...]) -> pd.DataFrame:
    for table_name in ["sentiment_scored", "sentiment_news"]:
        if _engine is not None and has_table(tables, table_name):
            return read_table_cached(table_name, _engine)
    for csv_name in ["sentiment_scored.csv", "sentiment_news.csv"]:
        csv_path = DATA_DIR / csv_name
        if csv_path.exists():
            return pd.read_csv(csv_path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def build_simple_forecast(milk_ireland: pd.DataFrame) -> pd.DataFrame:
    """Simple 5-year trend forecast for display without requiring statsmodels."""
    d = milk_ireland.dropna(subset=["year", "value"]).sort_values("year")
    if len(d) < 3:
        return pd.DataFrame(columns=["year", "forecast"])

    x = d["year"].astype(float).to_numpy()
    y = d["value"].astype(float).to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    last_year = int(np.nanmax(x))
    future_years = np.arange(last_year + 1, last_year + 6)
    forecast = intercept + slope * future_years
    return pd.DataFrame({"year": future_years, "forecast": forecast})


# -----------------------------------------------------------------------------
# Load app data
# -----------------------------------------------------------------------------
engine, backend, db_warning = get_engine()
tables = tuple(list_tables_cached(engine)) if engine is not None else tuple()

production = load_production_data(engine, tables)
trade_balance = load_trade_balance(engine, tables)
master = load_master_data(engine, tables)
sentiment = load_sentiment_data(engine, tables)

# Main milk production slice
milk = production[
    production.get("commodity", pd.Series(dtype=str)).astype(str).str.lower().eq("milk, whole fresh cow")
    & production.get("element", pd.Series(dtype=str)).astype(str).str.lower().eq("production")
].copy()

if not milk.empty:
    milk = milk[milk["year"].notna() & milk["value"].notna()]
    milk["year"] = milk["year"].astype(int)


# -----------------------------------------------------------------------------
# Header and sidebar
# -----------------------------------------------------------------------------
st.title("🌾 Ireland Agricultural Intelligence Dashboard")
st.caption("Streamlit dashboard connected to the agriculture database created from the notebook.")

with st.sidebar:
    st.header("Dashboard Controls")
    st.write(f"**Database backend:** {backend}")

    if db_warning:
        st.warning(db_warning)

    if tables:
        with st.expander("Available database tables"):
            st.write(", ".join(tables))
    else:
        st.error("No database tables found. Run the notebook database cells first, then refresh this app.")

    available_countries = sorted(milk["country"].dropna().unique().tolist()) if not milk.empty else COUNTRIES
    default_countries = [c for c in ["Ireland", "United Kingdom", "France"] if c in available_countries]
    selected_countries = st.multiselect(
        "Production countries",
        options=available_countries,
        default=default_countries or available_countries[:1],
    )

    if not milk.empty:
        min_year = int(milk["year"].min())
        max_year = int(milk["year"].max())
        year_range = st.slider("Production year range", min_year, max_year, (max(min_year, 2000), max_year))
    else:
        year_range = (2000, 2024)

    trade_countries = sorted(trade_balance["country"].dropna().unique().tolist()) if not trade_balance.empty else COUNTRIES
    selected_trade_country = st.selectbox(
        "Trade country",
        options=trade_countries,
        index=trade_countries.index("Ireland") if "Ireland" in trade_countries else 0,
    ) if trade_countries else "Ireland"


# -----------------------------------------------------------------------------
# KPI cards
# -----------------------------------------------------------------------------
ire_milk = milk[milk["country"].eq("Ireland")].sort_values("year") if not milk.empty else pd.DataFrame()
if not ire_milk.empty:
    latest_milk_year = int(ire_milk["year"].max())
    latest_milk_value = float(ire_milk.loc[ire_milk["year"].eq(latest_milk_year), "value"].sum())
else:
    latest_milk_year = None
    latest_milk_value = np.nan

ire_trade = trade_balance[trade_balance.get("country", pd.Series(dtype=str)).eq("Ireland")].sort_values("year") if not trade_balance.empty else pd.DataFrame()
latest_trade_balance = float(ire_trade["trade_balance"].iloc[-1]) if not ire_trade.empty and "trade_balance" in ire_trade.columns else np.nan

if not master.empty and "cattle_head" in master.columns and "country" in master.columns:
    cattle_series = master[master["country"].eq("Ireland")].sort_values("year")["cattle_head"].dropna()
    latest_cattle = float(cattle_series.iloc[-1]) if not cattle_series.empty else np.nan
else:
    latest_cattle = np.nan

sentiment_mean = safe_numeric(sentiment["compound"]).mean() if "compound" in sentiment.columns else np.nan

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(
    "Ireland Milk Production",
    "N/A" if pd.isna(latest_milk_value) else f"{latest_milk_value / 1_000_000:.2f}M t",
    help=f"Latest year available: {latest_milk_year}" if latest_milk_year else None,
)
kpi2.metric(
    "Ireland Trade Balance",
    "N/A" if pd.isna(latest_trade_balance) else f"${latest_trade_balance / 1_000:,.0f}k",
    help="Agricultural export value minus import value",
)
kpi3.metric(
    "Cattle Herd",
    "N/A" if pd.isna(latest_cattle) else f"{latest_cattle / 1_000_000:.2f}M",
    help="Uses master_dataset cattle_head if available",
)
kpi4.metric(
    "Mean Sentiment",
    "N/A" if pd.isna(sentiment_mean) else f"{sentiment_mean:+.3f}",
    help="Uses sentiment_scored or sentiment_news if available",
)


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_overview, tab_production, tab_trade, tab_sentiment, tab_forecast, tab_data = st.tabs(
    ["Overview", "Production", "Trade Balance", "Sentiment", "Forecast", "Data Tables"]
)


with tab_overview:
    st.subheader("Project Overview")
    st.write(
        "This dashboard summarises Ireland's agricultural production, agricultural trade balance, "
        "news sentiment and simple milk-production forecasting. It is designed to keep working even "
        "when optional notebook outputs are not available yet."
    )

    c1, c2 = st.columns(2)
    with c1:
        if not milk.empty:
            latest_by_country = (
                milk.sort_values("year")
                .groupby("country", as_index=False)
                .tail(1)
                .sort_values("value", ascending=False)
            )
            fig = px.bar(
                latest_by_country,
                x="country",
                y="value",
                title="Latest Cow Milk Production by Country",
                labels={"value": "Production", "country": "Country"},
                color="country",
                color_discrete_map=COLOURS,
            )
            fig.update_yaxes(tickformat=".2s")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Milk production data is not available yet.")

    with c2:
        if not ire_trade.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ire_trade["year"], y=ire_trade["trade_balance"], mode="lines+markers", name="Trade balance"))
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            fig.update_layout(
                title="Ireland Agricultural Trade Balance",
                xaxis_title="Year",
                yaxis_title="Trade Balance",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Trade balance data is not available yet.")


with tab_production:
    st.subheader("Cow Milk Production")
    if milk.empty:
        st.warning("No milk production data was found. Check production_clean or faostat_production in the database.")
    else:
        filtered_milk = milk[
            milk["country"].isin(selected_countries)
            & milk["year"].between(year_range[0], year_range[1])
        ].sort_values("year")

        fig = go.Figure()
        for country in selected_countries:
            d = filtered_milk[filtered_milk["country"].eq(country)]
            if d.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=d["year"],
                    y=d["value"] / 1_000_000,
                    name=country,
                    mode="lines+markers" if country == "Ireland" else "lines",
                    line=dict(width=3 if country == "Ireland" else 2, color=COLOURS.get(country)),
                )
            )
        fig.add_vline(
            x=2015,
            line_dash="dot",
            line_color="black",
            opacity=0.5,
            annotation_text="EU quota abolition 2015",
            annotation_position="top right",
        )
        fig.update_layout(
            title="Cow Milk Production, 2000 onwards",
            xaxis_title="Year",
            yaxis_title="Million tonnes",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View filtered production data"):
            st.dataframe(filtered_milk, use_container_width=True)


with tab_trade:
    st.subheader("Agricultural Trade Balance")
    if trade_balance.empty:
        st.warning("No trade balance data was found. Check trade_balance or faostat_trade in the database.")
    else:
        d = trade_balance[trade_balance["country"].eq(selected_trade_country)].sort_values("year")
        if d.empty:
            st.warning(f"No trade balance rows found for {selected_trade_country}.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=d["year"], y=d["export_val"] / 1_000, name="Exports", opacity=0.8))
            fig.add_trace(go.Bar(x=d["year"], y=-d["import_val"] / 1_000, name="Imports", opacity=0.8))
            fig.add_trace(go.Scatter(x=d["year"], y=d["trade_balance"] / 1_000, name="Balance", mode="lines+markers"))
            fig.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.5)
            fig.update_layout(
                title=f"{selected_trade_country} Agricultural Trade Balance",
                xaxis_title="Year",
                yaxis_title="Value, thousand USD or source unit",
                barmode="overlay",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            latest = d.iloc[-1]
            st.info(
                f"Latest year shown: **{int(latest['year'])}** | "
                f"Exports: **{latest['export_val'] / 1_000:,.0f}k** | "
                f"Imports: **{latest['import_val'] / 1_000:,.0f}k** | "
                f"Balance: **{latest['trade_balance'] / 1_000:,.0f}k**"
            )

            with st.expander("View trade balance data"):
                st.dataframe(d, use_container_width=True)


with tab_sentiment:
    st.subheader("Agriculture News Sentiment")
    if sentiment.empty or "compound" not in sentiment.columns:
        st.warning("Sentiment scores are not available. Run the notebook sentiment cells to create sentiment_scored.")
    else:
        sentiment = sentiment.copy()
        sentiment["compound"] = safe_numeric(sentiment["compound"])
        if "sentiment_label" not in sentiment.columns:
            sentiment["sentiment_label"] = np.select(
                [sentiment["compound"] > 0.05, sentiment["compound"] < -0.05],
                ["Positive", "Negative"],
                default="Neutral",
            )

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                sentiment,
                x="compound",
                color="perspective" if "perspective" in sentiment.columns else None,
                nbins=30,
                title="Compound Sentiment Score Distribution",
            )
            fig.add_vline(x=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "perspective" in sentiment.columns:
                cats = sentiment.groupby(["perspective", "sentiment_label"]).size().reset_index(name="count")
                fig = px.bar(cats, x="perspective", y="count", color="sentiment_label", barmode="group", title="Sentiment by Perspective")
            else:
                cats = sentiment["sentiment_label"].value_counts().reset_index()
                cats.columns = ["sentiment_label", "count"]
                fig = px.bar(cats, x="sentiment_label", y="count", color="sentiment_label", title="Sentiment Categories")
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("View sentiment rows"):
            st.dataframe(sentiment, use_container_width=True)


with tab_forecast:
    st.subheader("Simple Milk Production Forecast")
    if ire_milk.empty:
        st.warning("Ireland milk production data is required for the forecast chart.")
    else:
        forecast = build_simple_forecast(ire_milk)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ire_milk["year"],
                y=ire_milk["value"] / 1_000_000,
                mode="lines+markers",
                name="Historical",
                line=dict(color=COLOURS["Ireland"], width=3),
            )
        )
        if not forecast.empty:
            fig.add_trace(
                go.Scatter(
                    x=forecast["year"],
                    y=forecast["forecast"] / 1_000_000,
                    mode="lines+markers",
                    name="Trend forecast",
                    line=dict(dash="dash", width=3),
                )
            )
        fig.update_layout(
            title="Ireland Milk Production, Historical and 5-Year Trend Forecast",
            xaxis_title="Year",
            yaxis_title="Million tonnes",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This is a simple linear trend forecast for dashboard display. Use the notebook's ML/ARIMA section for formal model evaluation.")


with tab_data:
    st.subheader("Loaded Data Summary")
    summary = pd.DataFrame(
        [
            {"dataset": "production / milk slice", "rows": len(milk), "columns": len(milk.columns) if not milk.empty else 0},
            {"dataset": "trade_balance", "rows": len(trade_balance), "columns": len(trade_balance.columns) if not trade_balance.empty else 0},
            {"dataset": "master_dataset", "rows": len(master), "columns": len(master.columns) if not master.empty else 0},
            {"dataset": "sentiment", "rows": len(sentiment), "columns": len(sentiment.columns) if not sentiment.empty else 0},
        ]
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    selected_dataset = st.selectbox(
        "Preview dataset",
        ["Milk production", "Trade balance", "Master dataset", "Sentiment"],
    )
    preview_map = {
        "Milk production": milk,
        "Trade balance": trade_balance,
        "Master dataset": master,
        "Sentiment": sentiment,
    }
    st.dataframe(preview_map[selected_dataset].head(200), use_container_width=True)
