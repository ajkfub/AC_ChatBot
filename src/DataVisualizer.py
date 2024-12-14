import yfinance as yf
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

# Set Seaborn styling
sns.set(font="Times")
sns.set_style("ticks")
sns.set_context("poster", font_scale=0.75, rc={"grid.linewidth": 0.75})


class DataVisualizer:
    """
    A class to visualize financial data for a given stock ticker using Yahoo Finance.

    Attributes:
        ticker (str): The stock ticker symbol.
        company (yf.Ticker): An object representing the financial data for the ticker.
        data (pd.DataFrame): A DataFrame containing processed financial data.
    """

    def __init__(self, ticker: str) -> None:
        """
        Initializes the DataVisualizer with the specified stock ticker.

        Args:
            ticker (str): The stock ticker symbol to visualize.
        """
        self.ticker = ticker
        self.company = yf.Ticker(ticker)
        self.data = self._get_yf_financial_data()

    def _get_yf_financial_data(self) -> pd.DataFrame:
        """
        Fetches and combines financial data from Yahoo Finance into a single DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing both annual and quarterly financial data.
        """
        # Retrieve quarterly and annual financial data
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

        # Process and combine both annual and quarterly data
        df = pd.concat(
            [
                self._process_yf_financial_data(df_annual, "A"),
                self._process_yf_financial_data(df_quarter, "Q"),
            ]
        )

        return df

    def _process_yf_financial_data(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Processes the financial data DataFrame to a long format suitable for visualization.

        Args:
            df (pd.DataFrame): The financial data DataFrame to process.
            freq (str): Frequency of the data ('A' for Annual, 'Q' for Quarterly).

        Returns:
            pd.DataFrame: A processed DataFrame in long format with additional metadata.
        """
        df = df.reset_index().drop_duplicates().set_index("index").sort_index()
        df = df.melt(ignore_index=False)
        df = df.reset_index().rename(columns={"index": "item", "variable": "period"})
        df["period"] = pd.to_datetime(df["period"])
        df["ticker"] = self.ticker
        df["frequency"] = freq

        df = df.sort_values(["period", "item"])

        return df

    def data_visualization(self, item: str, freq: str) -> None:
        """
        Visualizes the specified financial item over time using a bar plot.

        Args:
            item (str): The financial item to visualize.
            freq (str): The frequency of the data ('A' for Annual, 'Q' for Quarterly).
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Filter data for the specified item and frequency
        df_plot = self.data.query("item == @item and frequency == @freq")
        df_plot = df_plot.dropna(subset="value")

        # Create a bar plot
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
        plt.show()

    def display_markdown_table(self, item: str, freq: str) -> None:
        """
        Displays the financial data in a markdown table format.

        Args:
            item (str): The financial item to display.
            freq (str): The frequency of the data ('A' for Annual, 'Q' for Quarterly).
        """
        df_display = self.data.query("item == @item and frequency == @freq")
        df_display = df_display.set_index("period")
        df_display = df_display.dropna(subset="value")
        print(df_display.to_markdown(tablefmt="grid"))

    def display(self, item: str, freq: str) -> None:
        """
        Displays both the visualization and markdown table for the specified financial item.

        Args:
            item (str): The financial item to visualize and display.
            freq (str): The frequency of the data ('A' for Annual, 'Q' for Quarterly).
        """
        self.data_visualization(item, freq)
        self.display_markdown_table(item, freq)


if __name__ == "__main__":
    # Set up argument parsing for command line execution
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stock_code", required=True, help="Stock ticker symbol (e.g., AAPL)"
    )
    parser.add_argument(
        "--item",
        required=True,
        help='Financial item to visualize (e.g., "totalAssets")',
    )
    parser.add_argument(
        "--freq",
        required=True,
        choices=["A", "Q"],
        help='Frequency of the data: "A" for Annual, "Q" for Quarterly',
    )

    # Parse command line arguments
    args = parser.parse_args()
    stock_code, item, freq = args.stock_code, args.item, args.freq

    # Create an instance of DataVisualizer and display the data
    visualizer = DataVisualizer(stock_code)
    visualizer.display(item, freq)
