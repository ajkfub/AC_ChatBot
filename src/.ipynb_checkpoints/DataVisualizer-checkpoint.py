import yfinance as yf
import pandas as pd

import seaborn as sns

sns.set(font="Times")
sns.set_style("ticks")
sns.set_context("poster", font_scale=0.75, rc={"grid.linewidth": 0.75})

import matplotlib.pyplot as plt


class DataVisualizer:

    def __init__(self, ticker):
        self.ticker = ticker
        self.company = yf.Ticker(ticker)
        self.data = self._get_yf_financial_data()

    def _get_yf_financial_data(self) -> pd.DataFrame:
        df_quarter = pd.concat(
            [
                self.company.quarterly_balance_sheet,
                self.company.quarterly_cash_flow,
                self.company.quarterly_financials,
                self.company.quarterly_income_stmt,
            ]
        )

        df_annual = pd.concat(
            [
                self.company.balance_sheet,
                self.company.cash_flow,
                self.company.financials,
                self.company.income_stmt,
            ]
        )

        df = pd.concat(
            [
                self._process_yf_financial_data(df_annual, "A"),
                self._process_yf_financial_data(df_quarter, "Q"),
            ]
        )

        return df

    def _process_yf_financial_data(self, df, freq) -> pd.DataFrame:
        df = df.reset_index().drop_duplicates().set_index("index").sort_index()
        df = df.melt(ignore_index=False)
        df = df.reset_index().rename(columns={"index": "item", "variable": "period"})
        df["period"] = pd.to_datetime(df["period"])
        df["ticker"] = self.ticker
        df["frequency"] = freq

        df = df.sort_values(["period", "item"])

        return df

    def data_visualization(self, item, freq):
        fig, ax = plt.subplots(figsize=(7.5, 4))

        df_plot = self.data.query("item == @item and frequency == @freq")
        df_plot = df_plot.dropna(subset="value")

        sns.barplot(
            data=df_plot,
            x="period",
            y="value",
            color="#95d0fc",
            linewidth=2,
            edgecolor="black",
        )

        freq_str = "Quarterly" if freq == "Q" else "Annual"
        ax.set_title(f"{freq_str} {item} [{self.ticker}]", weight="bold")

        return

    def display_markdown_table(self, item, freq):
        df_display = self.data.query("item == @item and frequency == @freq")
        df_display = df_display.set_index("period")
        print(df_display.to_markdown(tablefmt="grid"))

    def display(self, item, freq):
        self.data_visualization(item, freq)
        self.display_markdown_table(item, freq)


if __name__ == "__main__":
    AAPL = DataVisualizer("AAPL")
    AAPL.display("Net Income", "Q")
