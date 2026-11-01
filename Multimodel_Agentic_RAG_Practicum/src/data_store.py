"""Structured financial data store + deterministic analytics.

The LLM is *not* trusted to do arithmetic. All numeric look-ups, rankings,
year-over-year differences and percentage changes are computed here with
pandas and handed to the LLM as pre-computed, citable facts.
"""
from __future__ import annotations

import difflib
import re
from functools import lru_cache

import pandas as pd

from src import config
from src.ingestion.loaders import CANONICAL_COLUMNS

METRICS = {
    "net_cost": "net_cost_bil",
    "gross_cost": "gross_cost_bil",
    "earned_revenue": "earned_revenue_bil",
    "subtotal": "subtotal_bil",
    "change_in_assumptions": "change_in_assumptions_bil",
}
METRIC_LABELS = {
    "net_cost_bil": "Net Cost",
    "gross_cost_bil": "Gross Cost",
    "earned_revenue_bil": "Earned Revenue",
    "subtotal_bil": "Subtotal (Gross Cost − Earned Revenue)",
    "change_in_assumptions_bil": "(Gain)/Loss from Changes in Assumptions",
}
# Rows that are totals / reconciling items, not agencies — excluded from rankings.
NON_AGENCY_ROWS = {"total", "subtotal", "all other entities",
                   "unmatched transactions and balances (note 1.t)"}

# Common abbreviations users type → canonical agency names.
AGENCY_ALIASES = {
    "hhs": "Department of Health and Human Services", "health": "Department of Health and Human Services",
    "ssa": "Social Security Administration", "social security": "Social Security Administration",
    "dod": "Department of Defense", "defense": "Department of Defense", "pentagon": "Department of Defense",
    "military": "Department of Defense",
    "interest": "Interest on Treasury Securities held by the public",
    "va": "Department of Veterans Affairs", "veterans": "Department of Veterans Affairs",
    "treasury": "Department of the Treasury",
    "usda": "Department of Agriculture", "agriculture": "Department of Agriculture",
    "dot": "Department of Transportation", "transportation": "Department of Transportation",
    "dhs": "Department of Homeland Security", "homeland": "Department of Homeland Security",
    "hud": "Department of Housing and Urban Development", "housing": "Department of Housing and Urban Development",
    "doe": "Department of Energy", "energy": "Department of Energy",
    "dol": "Department of Labor", "labor": "Department of Labor",
    "doj": "Department of Justice", "justice": "Department of Justice",
    "epa": "Environmental Protection Agency",
    "state department": "Department of State", "dos": "Department of State",
    "interior": "Department of the Interior", "doi": "Department of the Interior",
    "nasa": "National Aeronautics and Space Administration",
    "commerce": "Department of Commerce", "doc": "Department of Commerce",
    "usaid": "Agency for International Development",
    "rrb": "Railroad Retirement Board", "usps": "United States Postal Service",
    "postal": "United States Postal Service", "fcc": "Federal Communications Commission",
    "nsf": "National Science Foundation", "sba": "Small Business Administration",
    "ncua": "National Credit Union Administration", "gsa": "General Services Administration",
    "pbgc": "Pension Benefit Guaranty Corporation", "nrc": "U.S. Nuclear Regulatory Commission",
    "dfc": "United States International Development Finance Corporation",
    "exim": "Export-Import Bank of the United States", "ex-im": "Export-Import Bank of the United States",
    "tva": "Tennessee Valley Authority", "sec": "Securities and Exchange Commission",
    "fdic": "Federal Deposit Insurance Corporation", "opm": "Office of Personnel Management",
    "education": "Department of Education", "ed": "Department of Education",
    "whole government": "Total", "federal government": "Total", "government total": "Total",
}


def fmt_b(value: float | None) -> str:
    """Format a billions value: 1886.3 -> '$1,886.3B'."""
    if value is None or pd.isna(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.1f}B"


class FinancialDataStore:
    """Wraps the unified Statements-of-Net-Cost table."""

    def __init__(self, df: pd.DataFrame | None = None):
        if df is None:
            df = (pd.read_csv(config.UNIFIED_CSV) if config.UNIFIED_CSV.exists()
                  else pd.DataFrame(columns=CANONICAL_COLUMNS))
        self.df = df.copy()
        if not self.df.empty:
            self.df["statement_fiscal_year"] = self.df["statement_fiscal_year"].astype(int)
            self.df["report_fiscal_year"] = self.df["report_fiscal_year"].astype(int)

    # ------------------------------------------------------------------ basics
    @property
    def empty(self) -> bool:
        return self.df.empty

    @property
    def agencies(self) -> list[str]:
        return sorted(self.df["agency_name"].unique()) if not self.empty else []

    @property
    def years(self) -> list[int]:
        return sorted(self.df["statement_fiscal_year"].unique().tolist()) if not self.empty else []

    def resolve_agency(self, name: str) -> str | None:
        """Fuzzy-map user text ('DoD', 'veterans affairs') to a canonical agency name."""
        if not name:
            return None
        key = name.strip().lower()
        if key in AGENCY_ALIASES:
            return AGENCY_ALIASES[key]
        lower_map = {a.lower(): a for a in self.agencies}
        if key in lower_map:
            return lower_map[key]
        for low, orig in lower_map.items():   # substring match
            if key in low or low in key:
                return orig
        match = difflib.get_close_matches(key, list(lower_map), n=1, cutoff=0.6)
        return lower_map[match[0]] if match else None

    def find_agencies_in_text(self, text: str) -> list[str]:
        found: list[str] = []
        low = f" {text.lower()} "
        for agency in self.agencies:
            if agency.lower() in low and agency not in found:
                found.append(agency)
        for alias, agency in AGENCY_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", low) and agency not in found:
                found.append(agency)
        return found

    # --------------------------------------------------------------- selection
    def preferred(self, include_restated: bool = False) -> pd.DataFrame:
        """One row per (agency, fiscal year).

        Prefers the *originally reported* figure (restatement_flag == 'N'); falls
        back to the restated figure when no original exists (e.g. FY2022, which
        only appears as a restatement inside the FY2023 report).
        """
        if self.empty:
            return self.df
        if include_restated:
            return self.df
        d = self.df.copy()
        d["_rank"] = (d["restatement_flag"] != "N").astype(int)
        d = d.sort_values(["agency_name", "statement_fiscal_year", "_rank", "report_fiscal_year"])
        return d.drop_duplicates(["agency_name", "statement_fiscal_year"]).drop(columns="_rank")

    # ------------------------------------------------------------------- tools
    def lookup(self, agencies: list[str] | None = None, years: list[int] | None = None,
               include_restated: bool = False) -> pd.DataFrame:
        d = self.preferred(include_restated)
        if agencies:
            d = d[d["agency_name"].isin(agencies)]
        if years:
            d = d[d["statement_fiscal_year"].isin(years)]
        return d.sort_values(["agency_name", "statement_fiscal_year", "report_fiscal_year"])

    def rank(self, year: int, metric: str = "net_cost_bil", top_n: int = 10,
             ascending: bool = False) -> pd.DataFrame:
        d = self.preferred()
        d = d[(d["statement_fiscal_year"] == year)
              & (~d["agency_name"].str.lower().isin(NON_AGENCY_ROWS))]
        return d.sort_values(metric, ascending=ascending).head(top_n)

    def compare(self, agency: str, year_a: int, year_b: int,
                metric: str = "net_cost_bil") -> dict | None:
        d = self.preferred()
        d = d[d["agency_name"] == agency].set_index("statement_fiscal_year")
        if year_a not in d.index or year_b not in d.index:
            return None
        a, b = float(d.loc[year_a, metric]), float(d.loc[year_b, metric])
        diff = round(b - a, 1)
        pct = round(diff / abs(a) * 100, 1) if a else None
        return {"agency": agency, "metric": metric, "year_a": year_a, "value_a": a,
                "year_b": year_b, "value_b": b, "change": diff, "pct_change": pct,
                "sources": sorted(set(d.loc[[year_a, year_b], "source_file"]))}

    def restatement_diff(self, year: int) -> pd.DataFrame:
        """Originally reported vs restated figures for a fiscal year."""
        d = self.df[self.df["statement_fiscal_year"] == year]
        orig = d[d["restatement_flag"] == "N"].set_index("agency_name")["net_cost_bil"]
        rest = d[d["restatement_flag"] == "Y"].set_index("agency_name")["net_cost_bil"]
        out = pd.concat([orig.rename("original"), rest.rename("restated")], axis=1).dropna()
        out["difference"] = (out["restated"] - out["original"]).round(1)
        return out[out["difference"] != 0].sort_values("difference", key=abs, ascending=False)

    # ------------------------------------------------------------ rendering
    @staticmethod
    def to_markdown(df: pd.DataFrame, metrics: list[str] | None = None) -> str:
        metrics = metrics or ["gross_cost_bil", "earned_revenue_bil", "subtotal_bil",
                              "change_in_assumptions_bil", "net_cost_bil"]
        cols = ["agency_name", "statement_fiscal_year", "restatement_flag", *metrics, "source_file"]
        view = df[[c for c in cols if c in df.columns]].copy()
        for m in metrics:
            if m in view:
                view[m] = view[m].map(fmt_b)
        view = view.rename(columns={"agency_name": "Agency", "statement_fiscal_year": "FY",
                                    "restatement_flag": "Restated?", "source_file": "Source",
                                    **METRIC_LABELS})
        return view.to_markdown(index=False) if not view.empty else "_No matching rows._"


@lru_cache(maxsize=1)
def get_store() -> FinancialDataStore:
    return FinancialDataStore()


def reload_store() -> FinancialDataStore:
    get_store.cache_clear()
    return get_store()
