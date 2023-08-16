import pandas as pd
import QuantLib as ql
from datetime import datetime, date, timedelta, time
import logging

logging.basicConfig(level=logging.INFO)

CONVERSION_PRICE = 295
CONVERSION_NUMBERS = 100 / 295
CORPORATEBOND_RATE = 3.25
PUTABLE_DATE = date(2028, 3, 29)
CONVERSIONEFFECTIVE_DATE = date(2023, 3, 29)
TODAY_DATE = date(2023, 7, 19)
YEARS_LEFT = 5 * (PUTABLE_DATE - TODAY_DATE) / (PUTABLE_DATE - CONVERSIONEFFECTIVE_DATE)

# use TW 10y rate as risk free rate
RISK_FREE_RATE = 0.011575
FIRST_CONVERTABLE_DATE = date(2028, 6, 30)


def create_option(
    spot_price, strike, maturity, calculation_datetime, volatility=0.08012
):
    global RISK_FREE_RATE

    risk_free_rate = RISK_FREE_RATE

    maturity_date = ql.Date(maturity.day, maturity.month, maturity.year)
    FIRST_CONVERTABLE_DATE = ql.Date(30, 6, 2023)

    calculation_date = ql.Date(
        calculation_datetime.day, calculation_datetime.month, calculation_datetime.year
    )

    day_count = ql.Actual365Fixed()
    calendar = ql.Taiwan()

    ql.Settings.instance().evaluationDate = calculation_date
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    us_exercise = ql.AmericanExercise(FIRST_CONVERTABLE_DATE, maturity_date)
    option = ql.VanillaOption(payoff, us_exercise)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calculation_date, risk_free_rate, day_count)
    )

    flat_vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(calculation_date, calendar, volatility, day_count)
    )

    bs_process = ql.BlackScholesProcess(spot_handle, flat_ts, flat_vol_ts)
    return option, bs_process


def _cal_IV(spot_price, strike, maturity, premium, tradeDate):
    """Calculate the option IV"""

    option, bs_process = create_option(spot_price, strike, maturity, tradeDate)
    try:
        return option.impliedVolatility(premium, bs_process)
    except Exception as e:
        print(e)
        return 0


def cb_option_price(cb_price):
    return (cb_price - (100 - CORPORATEBOND_RATE * YEARS_LEFT)) / CONVERSION_NUMBERS


def find_bid_mid_price(row):
    values = row[
        ["BidPrice0", "BidPrice1", "BidPrice2", "BidPrice3", "BidPrice4"]
    ].values
    values = values[values != 0]
    if len(values) == 0:
        return None
    else:
        return values.mean()


def find_ask_mid_price(row):
    values = row[
        ["AskPrice0", "AskPrice1", "AskPrice2", "AskPrice3", "AskPrice3"]
    ].values
    values = values[values != 0]
    if len(values) == 0:
        return None
    else:
        return values.mean()


def data_preprocessing(df):
    df["Bid_Mid"] = df.apply(find_bid_mid_price, axis=1)
    df["Ask_Mid"] = df.apply(find_ask_mid_price, axis=1)

    df["Datetime"] = df["Timestamp"].apply(
        lambda x: datetime.fromtimestamp(x / 1000000000)
    )

    df = df[["Datetime", "Bid_Mid", "Ask_Mid"]]
    new_df = df.fillna(method="ffill")
    new_df = new_df.dropna(subset=["Bid_Mid", "Ask_Mid"], how="all")
    return new_df


def read_n_process_data(stock_folder_path, CB_folder_path):
    import os
    import sys

    name_list_1 = os.listdir(stock_folder_path)
    name_list_2 = os.listdir(CB_folder_path)
    name_list_1.sort()
    name_list_2.sort()

    if len(name_list_1) != len(name_list_2):
        logging.error("CB and Stock data not haveing same dates")
        sys.exit(1)

    df_list = []
    for i in range(len(name_list_1)):
        logging.info(f"Processing {name_list_1[i]}")
        logging.info(f"Processing {name_list_2[i]}")

        stock_2727_df = data_preprocessing(
            pd.read_csv(stock_folder_path + name_list_1[i])
        )
        cb_2727_df = data_preprocessing(pd.read_csv(CB_folder_path + name_list_2[i]))

        stock_2727_df = stock_2727_df.rename(
            columns={"Bid_Mid": "Stock_Bid_Mid", "Ask_Mid": "Stock_Ask_Mid"}
        )
        cb_2727_df = cb_2727_df.rename(
            columns={"Bid_Mid": "CB_Bid_Mid", "Ask_Mid": "CB_Ask_Mid"}
        )

        new_stock_2727_df = stock_2727_df.fillna(method="ffill")
        new_cb_2727_df = cb_2727_df.fillna(method="ffill")

        new_cb_2727_df["CB_Bid_One_Mid"] = new_cb_2727_df["CB_Bid_Mid"].apply(
            cb_option_price
        )
        new_cb_2727_df["CB_Ask_One_Mid"] = new_cb_2727_df["CB_Ask_Mid"].apply(
            cb_option_price
        )

        merge_df = pd.merge_asof(new_stock_2727_df, new_cb_2727_df, on="Datetime")
        merge_df["Time"] = merge_df["Datetime"].apply(lambda x: x.time())
        merge_df = merge_df[merge_df["Time"] > time(8, 59)]
        df_list.append(merge_df)
    return pd.concat(df_list).reset_index(drop=True)


def cal_IV(df):
    df["Bid_IV"] = df.apply(
        lambda row: _cal_IV(
            row["Stock_Bid_Mid"],
            CONVERSION_PRICE,
            PUTABLE_DATE,
            row["CB_Bid_One_Mid"],
            TODAY_DATE,
        ),
        axis=1,
    )
    df["Ask_IV"] = df.apply(
        lambda row: _cal_IV(
            row["Stock_Ask_Mid"],
            CONVERSION_PRICE,
            PUTABLE_DATE,
            row["CB_Ask_One_Mid"],
            TODAY_DATE,
        ),
        axis=1,
    )

    df["Bid_IV"] = df["Bid_IV"] * 100
    df["Ask_IV"] = df["Ask_IV"] * 100
    return df


def draw_chart(df):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    fig, ax = plt.subplots(figsize=(70, 15), dpi=70) 
    ax.patch.set_facecolor("#EFE9E6")
    df.replace(0, np.nan, inplace=True)
    df["Datetime"] = df["Datetime"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S.%f"))

    ax.plot(df["Datetime"], df["Bid_IV"], label="Bid_IV", color="#FF0000")
    ax.plot(df["Datetime"], df["Ask_IV"], label="Ask_IV", color="#0000FF")

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(600))
    plt.xlabel("Datetime", size=13)
    plt.ylabel("IV", size=13)
    plt.xticks(rotation=90)
    plt.grid(axis = "y")

    plt.legend(loc="upper left")
    plt.title("27271_IV Chart", size=20)
    plt.savefig("27271_IV Chart_ADJ.png")


def main():
    df = read_n_process_data("./Cathay_CB_27271_Project/20230501_20230531_2727/", "./Cathay_CB_27271_Project/20230501_20230531_27271/")
    df = cal_IV(df)

    df.to_csv("27271_IV.csv", index=False)
    # df = pd.read_csv("27271_IV.csv")
    draw_chart(df)


if __name__ == "__main__":
    main()
