import numpy as np
from datetime import datetime, timedelta, time
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import os
from io import StringIO
import requests
from bs4 import BeautifulSoup
from pytrends.request import TrendReq
from GoogleNews import GoogleNews
from scipy.signal import argrelextrema

try:
    from talib.abstract import SMA, MACD, RSI, SAR, BBANDS, ATR
    from talib import (
        CDLHAMMER,
        CDLHANGINGMAN,
        CDLENGULFING,
        CDLDARKCLOUDCOVER,
        CDLPIERCING,
        CDLHARAMI,
        CDLHARAMICROSS,
        CDLEVENINGSTAR,
        CDLMORNINGSTAR,
        CDLEVENINGDOJISTAR,
        CDLMORNINGDOJISTAR,
        CDLGRAVESTONEDOJI,
        CDLLONGLEGGEDDOJI,
        CDLSPINNINGTOP,
        CDLHIGHWAVE,
        CDLDOJI,
    )
except Exception as e:
    print(f"Talib not found, do not call candles function : {e}")


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf

from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import warnings
from pandas.core.common import SettingWithCopyWarning
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pandas.core.base import PandasObject
import pandas as pd
from pathlib import Path
from ib_insync import *

global ib
ib = IB()


DIR = f"{Path('.').parent.absolute()}"
global PATH
PATH = {
    "symbols": f"{DIR}/data/",
    "day": f"{DIR}/data/day/",
    "min": f"{DIR}/data/min/",
}


pd.options.plotting.backend = "plotly"


warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

kwargs = {
    "CO": "",
    "OH": "",
    "OL": "",
    "OC": "",
    "SAR": "",
    "VWAP": "",
    "SMA": [
        9,
        20,
        50,
        200,
    ],
    "MACD": [
        (12, 26, 9),
    ],
    "RSI": [
        14,
    ],
    "avg_vol": [10, 60],
    "BBANDS": [
        (9, "close", False),
    ],
    "ATR": [
        14,
    ],
    "pansy": [
        0.2,
    ],
    "Extreme": [
        (4, "high", np.greater_equal),
        (4, "low", np.less_equal),
    ],
}


class Symbols:
    def __init__(self, exchange):
        self.exchange = exchange
        self.__check(exchange)

    def __check(self, file):
        if not os.path.exists(f"{PATH['symbols']}symbolsDelta"):
            df = pd.DataFrame(
                columns=[
                    "symbol",
                    "name",
                    "lastsale",
                    "marketcap",
                    "ipoyear",
                    "sector",
                    "industry",
                    "summary quote",
                    "date",
                    "status",
                    "exchange",
                ],
            )
            df.set_index("symbol", inplace=True)
            df.to_csv(f"{PATH['symbols']}symbolsDelta", index=True)
            print(f"{PATH['symbols']}symbolsDelta has been created")
        if not os.path.exists(f"{PATH['symbols']}{file}"):
            data = self.download()
            if isinstance(data, pd.DataFrame):
                data.to_csv(f"{PATH['symbols']}{file}", index=True)
                print(f"{PATH['symbols']}{file} has been created")
            else:
                print(
                    f"Error when trying to download stock list for {self.exchange} : {data}"
                )

    def diff(self, list1, list2):
        return list(set(list1) - set(list2))

    def download(self):
        """ "
        exchange paramter - string - can be one of the following : nasdaq, amex, nyse
        """
        r = requests.get(
            "https://old.nasdaq.com/screening/companies-by-name.aspx",
            params={"exchange": self.exchange, "render": "download"},
            headers={
                "Accept": "*/*",
                "Host": "old.nasdaq.com",
                "User-agent": "Mozilla/5.0 (compatible; Rigor/1.0.0; http://rigor.com)",
            },
        )
        if r.status_code == 200:
            data = r.content.decode("utf8")
            df = pd.read_csv(StringIO(data))
            df.columns = map(str.lower, df.columns)
            df.drop(
                df.columns[df.columns.str.contains("unnamed", case=False)],
                axis=1,
                inplace=True,
            )
            df.set_index("symbol", inplace=True)
            df.index = df.index.str.replace(" ", "")
            return df
        else:
            return r.reason

    def read(self):
        return pd.read_csv(f"{PATH['symbols']}{self.exchange}", index_col="symbol")

    def refresh(self):
        today = datetime.now().strftime("%Y-%m-%d")
        current = self.download()  # latest
        if not isinstance(current, pd.DataFrame):
            return f"Error when trying to download stock list for {self.exchange} : {current}"

        local = self.read()  # current
        local_symbols, current_symbols = list(local.index), list(current.index)
        added_symbols = self.diff(current_symbols, local_symbols)
        removed_symbols = self.diff(local_symbols, current_symbols)

        current.to_csv(f"{PATH['symbols']}{self.exchange}", index=True)
        if added_symbols or removed_symbols:
            delta = pd.read_csv(f"{PATH['symbols']}symbolsDelta", index_col="symbol")

            if added_symbols:
                added_symbols = current.loc[added_symbols]
                (
                    added_symbols["date"],
                    added_symbols["status"],
                    added_symbols["exchange"],
                ) = (today, "+", self.exchange)
                delta = pd.concat([delta, added_symbols])
            if removed_symbols:
                removed_symbols = local.loc[removed_symbols]
                (
                    removed_symbols["date"],
                    removed_symbols["status"],
                    removed_symbols["exchange"],
                ) = (today, "-", self.exchange)
                delta = pd.concat([delta, removed_symbols])

            delta.to_csv(f"{PATH['symbols']}symbolsDelta", index=True)
            return delta.loc[delta["date"] == today]


class Historical:
    global ib

    def __init__(self, contract):
        self.contract = contract

    def refresh(self):
        if os.path.exists(f"{PATH['day']}{self.contract.symbol}"):
            local = pd.read_pickle(f"{PATH['day']}{self.contract.symbol}")
            local.index = pd.to_datetime(local.index)
            last_day = local.index[-1]
            today = datetime.now()
            while today.weekday() > 4:
                today -= timedelta(days=1)
            days_missing = (today - last_day).days
            if days_missing != 0:
                bars = ib.reqHistoricalData(
                    self.contract,
                    endDateTime="",
                    durationStr=f"{str(days_missing+1)} D",
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                )
                df = util.df(bars)
                df.set_index(
                    "date", inplace=True
                )  # should consider updating util.df method
                df.index = pd.to_datetime(df.index)
                if not df.empty:
                    df = df.loc[df.index > last_day]
                    local = pd.concat([local, df], sort=True)
                    local.to_pickle(f"{PATH['day']}{self.contract.symbol}")
        else:
            local = self.download()
            if not local.empty:
                local.to_pickle(f"{PATH['day']}{self.contract.symbol}")
            else:
                print(f"Bars for {self.contract.symbol} came back empty")

    def download(self):
        df = pd.DataFrame()
        try:
            starting = ib.reqHeadTimeStamp(
                self.contract, whatToShow="TRADES", useRTH=True
            )
            years = relativedelta(datetime.today(), starting).years
        except Exception as e:
            print(f"Error {e} at {self.contract.symbol}")
            years = 49

        bars = ib.reqHistoricalData(
            self.contract,
            endDateTime="",
            durationStr=f"{str(years+1)} Y",
            # durationStr='50 Y',
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        df = util.df(bars)
        df.set_index("date", inplace=True)  # should consider updating util.df method
        df.index = pd.to_datetime(df.index)
        return df

    def granular(
        self, start, end, durationStr="1 D", barSizeSetting="1 min", useRTH=True
    ):
        data = pd.DataFrame()
        for each in tqdm(working_days(start, end)):
            try:
                bars = ib.reqHistoricalData(
                    self.contract,
                    endDateTime=each,
                    durationStr=durationStr,
                    barSizeSetting=barSizeSetting,
                    whatToShow="TRADES",
                    useRTH=useRTH,
                    formatDate=1,
                )
                df = util.df(bars)
                df.set_index("date", inplace=True)
                df.index = pd.to_datetime(df.index)
                data = pd.concat([data, df.loc[each.strftime("%Y-%m-%d")]], sort=True)
            except Exception as e:
                print(f"Failed for {each} because of {e}")
        return data


def working_days(start, end):
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    dates = pd.date_range(start=start, end=end, freq=us_bd)
    return [date.date() for date in dates]


def topGainers(date=datetime.now().strftime("%Y-%m-%d")) -> pd:
    symbols = [f for f in os.listdir(PATH["day"]) if not f.startswith(".")]
    top = pd.DataFrame()
    # date = "2020-05-08"
    for symbol in tqdm(symbols):
        try:
            df = pd.read_pickle(f'{PATH["day"]}{symbol}')
        except Exception as e:
            print(f"Error {symbol}, {e}")
        df = df.dropna(how="any")
        df["% OH"] = df.high * 100 / df.open - 100
        df["% OL"] = df.low * 100 / df.open - 100
        df["% OC"] = df.close * 100 / df.open - 100
        df["% CO"] = df.open * 100 / df.close.shift(1) - 100
        df = df.loc[df.index == date]

        df = df[
            ((df["% OH"] > 15) | (df["% CO"] > 5))
            & (df["high"] > 0.1)
            & (df["volume"] > 7500)
        ]

        if not df.empty:
            df["Symbol"] = symbol
            top = pd.concat([top, df], sort=True)
    if not top.empty:
        # top[['% OH', '% OC', '% CO', '% OL', 'volume']] = top[['% OH', '% OC', '% CO', '% OL', 'volume']].astype('int64')
        top = top.sort_values(
            [
                "% OH",
                "% CO",
            ],
            ascending=(
                False,
                False,
            ),
        )
        # symbols_db = pd.read_csv(F"{PATH['symbols']}")
        # top = pd.merge(top, symbols_db, left_on="Symbol", right_on = "symbol")
        top = top.reindex(
            columns=[
                "Symbol",
                "% CO",
                "% OH",
                "% OL",
                "% OC",
                "volume",
                "open",
                "high",
                "low",
                "close",
            ]
        )

    return top


def spikers(
    over=None,
    under=None,
    starting="2020-01-01",
    ending=datetime.now().strftime("%Y-%m-%d"),
):

    symbols = [f for f in os.listdir(PATH["day"]) if not f.startswith(".")]
    data = pd.DataFrame(
        columns=[
            "symbol",
            "spikes",
            "spikes_mean",
            "spikes_median",
            "spikes_sum",
            "drops",
            "drops_mean",
            "drops_median",
            "drops_sum",
        ]
    )
    for symbol in tqdm(symbols):
        df = pd.read_pickle(f"{PATH['day']}{symbol}")
        df = df.loc[(df.index > starting) & (df.volume > 7500) & (df.index < ending)]
        temp_spikes, temp_drops = {}, {}
        if over:
            df["%OH"] = df.high * 100 / df.open - 100
            spikes = df["%OH"][df["%OH"] > over]
            if len(spikes.index):
                temp_spikes = {
                    "symbol": symbol,
                    "spikes": len(spikes.index),
                    "spikes_mean": int(spikes.mean()),
                    "spikes_median": int(spikes.median()),
                    "spikes_sum": int(spikes.sum()),
                }
        if under:
            df["%OL"] = df.low * 100 / df.open - 100
            spikes = df["%OL"][df["%OL"] < under]
            if len(spikes.index):
                temp_drops = {
                    "symbol": symbol,
                    "drops": len(spikes.index),
                    "drops_mean": int(spikes.mean()),
                    "drops_median": int(spikes.median()),
                    "drops_sum": int(spikes.sum()),
                }

        if temp_spikes or temp_drops:
            data = data.append({**temp_spikes, **temp_drops}, ignore_index=True)
    data.sort_values(
        by=["spikes", "drops"], ascending=False, inplace=True, na_position="last"
    )
    return data


def clean(
    df: pd.DataFrame,
    dropna: str = "any",
) -> pd.DataFrame:
    """
    df :       pd.DataFrame
    dropna :   'all', 'any', None
    tz :       specify this in case your input data comes with timezone info attached
               and you want it transformed to another timezone and removed

    Output:
    - a dataframe object with a DateTimeIndex and (open, high, low, close, volume) as columns

    What it does:
    - transforms column names to lower-case
    - sets volume to int64
    - set the column that contains 'date' to index and transforms index to pd.datetime object
    - drops columns that contain 'any' nan values
    - selectes only columns needed for ohlcv
    """
    df = df.copy()
    df.columns = map(str.lower, df.columns)

    # df.volume = df.volume.astype('int64')

    date_column = df.columns[df.columns.str.contains("date")]
    if not date_column.empty:
        df.set_index(date_column.values[0], inplace=True)
    df.index = pd.to_datetime(df.index)
    if df.index.tz:
        df = df.tz_convert("Europe/Bucharest")
        df = df.tz_localize(None)

    if dropna:
        df.dropna(how=dropna, inplace=True)

    df = df.reindex(columns=["open", "high", "low", "close", "volume"])

    i = df.index[-1].time()
    if not (i.second == 0 and (i.minute == 30 or i.minute == 0)):
        return df[:-1]
    else:
        return df


def technicals(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    This function adds the following collumns to a df:

    kwargs = {
        'CO': '', #previous close - open % change
        'OH': '', #open - high % change
        'OL': '', #open - low % change
        'OC': '', #open - close % change
        'SAR': '', #Parabolic SAR
        'VWAP': '', #VWAP - should be used only on intraday data
        'SMA': [9, 20, 50, 200, ], #input a list of timeperiods
        'MACD': [(12,26,9), ], #input a list of tuples containing
                                (fastperiod, slowperiod, signalperiod)
        'RSI': [14, ], #input a list of timeperiods
        'AVG_VOL': [60, ], #input a list of timeperiods
        'BBANDS': [(9, 'close', False), ], #input a list of tuples containing
                                            (timperiod, desired column, a bool to show middle line)
        'ATR': [14, ], #input a list of timeperiods
        'pansy': [0.2, ], #input a list of percentages
        'EXTREME': [(4 ,'high', np.greater_equal),
                     (4, 'low', np.less_equal),
                    ] #input a list of tuples containing
                    (number of points to compare on each side,
                    desired column, comparison function)
    }

    """
    if "CO" in kwargs:
        df["CO"] = df.open * 100 / df.close.shift(1) - 100

    if "OH" in kwargs:
        df["OH"] = df.high * 100 / df.open - 100

    if "OL" in kwargs:
        df["OL"] = df.low * 100 / df.open - 100

    if "OC" in kwargs:
        df["OC"] = df.close * 100 / df.open - 100

    if "avg_vol" in kwargs:
        for w in kwargs["avg_vol"]:
            df[f"avg_vol_{w}"] = df.volume.shift(1).rolling(window=w).mean()

    if "SMA" in kwargs:
        for t in kwargs["SMA"]:
            df[f"SMA_{t}"] = SMA(df[["close"]], timeperiod=t)

    if "RSI" in kwargs:
        for t in kwargs["RSI"]:
            if len(kwargs["RSI"]) == 1:
                df["RSI"] = RSI(df[["close"]], timeperiod=t)
            else:
                df[f"RSI_{t}"] = RSI(df[["close"]], timeperiod=t)

    if "MACD" in kwargs:
        for fast, slow, signal in kwargs["MACD"]:
            if len(kwargs["MACD"]) == 1:
                df["MACD"] = MACD(
                    df[["close"]], fastperiod=fast, slowperiod=slow, signalperiod=signal
                )[["macdhist"]]
            else:
                df[f"MACD_{fast},{slow},{signal}"] = MACD(
                    df[["close"]], fastperiod=fast, slowperiod=slow, signalperiod=signal
                )[["macdhist"]]

    if "SAR" in kwargs:
        df["SAR"] = SAR(df[["high", "low"]])

    if "BBANDS" in kwargs:
        for t, on, show_middle in kwargs["BBANDS"]:
            bands = BBANDS(df[[on]], timeperiod=t)
            if len(kwargs["BBANDS"]) == 1:
                df["BB_upper"] = bands[["upperband"]]
                df["BB_lower"] = bands[["lowerband"]]
                if show_middle:
                    df["BB_middle"] = bands[["middleband"]]
            else:
                df[f"BB_upper_{t}_{on}"] = bands[["upperband"]]
                df[f"BB_lower_{t}_{on}"] = bands[["lowerband"]]
                if show_middle:
                    df[f"BB_middle_{t}_{on}"] = bands[["middleband"]]

    if "ATR" in kwargs:
        for t in kwargs["ATR"]:
            if len(kwargs["ATR"]) == 1:
                df["ATR"] = ATR(df[["high", "low", "close"]], timeperiod=t)
            else:
                df[f"ATR_{t}"] = ATR(df[["high", "low", "close"]], timeperiod=t)

    if "WVAP" in kwargs:
        df["VWAP"] = np.cumsum(df.volume * (df.high + df.low) / 2) / np.cumsum(
            df.volume
        )

    if "Extreme" in kwargs:
        for order, on, func in kwargs["Extreme"]:
            if len(kwargs["Extreme"]) > 2:
                df[f"Extreme_{order}_{on}"] = df.iloc[
                    argrelextrema(df[on].values, func, order=order)[0]
                ][on]
            else:
                df[f"Extreme_{on}"] = df.iloc[
                    argrelextrema(df[on].values, func, order=order)[0]
                ][on]
    if "pansy" in kwargs:
        pansy = abs(df["close"] - df["open"])
        for value in kwargs["pansy"]:
            if len(kwargs["pansy"]) == 1:
                df["pansy"] = pansy.apply(
                    lambda x: True if x <= pansy.quantile(q=value) else False
                )
            else:
                df[f"pansy_{value}"] = pansy.apply(
                    lambda x: True if x <= pansy.quantile(q=value) else False
                )
    return df


def technicals_old(df, vwap=False, lite=False):
    df = df.dropna(how="any")
    df["volume"] = df["volume"] * 100
    # df['volume'] = df.volume.astype('int64')#.apply(lambda x : '{0:,}'.format(x))
    df["avg_vol_10"] = df.volume.shift(1).rolling(window=10).mean()
    df["SMA_9"] = SMA(df[["close"]], timeperiod=9)
    df["SMA_200"] = SMA(df[["close"]], timeperiod=200)
    df["MACD"] = MACD(df[["close"]])[["macdhist"]]  # [['macd','macdsignal']]
    df["RSI"] = RSI(df[["close"]])

    df["Extreme_high"] = df.iloc[
        argrelextrema(df["high"].values, np.greater_equal, order=4)[0]
    ]["high"]
    df["Extreme_low"] = df.iloc[
        argrelextrema(df["low"].values, np.less_equal, order=4)[0]
    ]["low"]

    if vwap:
        df["VWAP"] = np.cumsum(df.volume * (df.high + df.low) / 2) / np.cumsum(
            df.volume
        )
    if not lite:
        df["CO"] = df.open * 100 / df.close.shift(1) - 100
        df["OH"] = df.high * 100 / df.open - 100
        df["OL"] = df.low * 100 / df.open - 100
        df["OC"] = df.close * 100 / df.open - 100
        df["avg_vol_60"] = df.volume.shift(1).rolling(window=60).mean()
        df["SMA_20"] = SMA(df[["close"]], timeperiod=20)
        df["SMA_50"] = SMA(df[["close"]], timeperiod=50)
        df = df.round({"CO": 1, "OH": 1, "OC": 1, "avg60": 0, "avg10": 0})
        # df = df.reindex(columns=['CO', 'OH', 'OL', 'OC', 'average', 'open', 'high', 'low', 'close', 'barCount', 'volume', 'avg_vol_60', 'avg_vol_10'])
        df["SAR"] = SAR(df[["high", "low"]])
        bands = BBANDS(df[["close"]], timeperiod=9)
        # df['BB_upper'], df['BB_middle'], df['BB_lower'] = bands[['upperband']], bands[['middleband']], bands[['lowerband']]
        df["BB_upper"], df["BB_lower"] = bands[["upperband"]], bands[["lowerband"]]
        df["ATR"] = ATR(df[["high", "low", "close"]])

    return df


def get_profile(symbol):
    profile = {}
    r = requests.get(
        f"https://finance.yahoo.com/quote/{symbol}/profile?p={symbol}",
        headers={  # "content-type":"application/x-www-form-urlencoded",
            "Accept": "*/*",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36",
        },
    )
    if r.status_code == 200:
        soup = BeautifulSoup(r.content, "html.parser")
        for each, entry in zip(
            soup.find_all(class_="Fw(600)")[-3:], ["Sector", "Industry", "Employees"]
        ):
            profile[entry] = each.get_text()
        return pd.DataFrame(profile, index=[""])
    else:
        return f"Error {r.reason} while trying to get profile from yahoo finance"


def get_fundamentals(symbol):
    r = requests.get(
        f"https://finviz.com/quote.ashx?t={symbol}",
        headers={  # "content-type":"application/x-www-form-urlencoded",
            "Accept": "*/*",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36",
        },
    )
    if r.status_code == 200:
        r = pd.read_html(r.text)[6]
        index, values = [], []

        for each in r.columns:
            if not (each % 2):
                index.extend(r[each].values)
            elif each % 2:
                values.extend(r[each].values)
        return pd.DataFrame({"Fundamentals": values}, index=index)
    else:
        return f"Error {r.reason} while trying to get fundamentals from finviz"


def get_insiderTrading(symbol):
    r = requests.get(
        f"https://api.nasdaq.com/api/company/{symbol}/insider-trades?limit=99999&type=ALL",
        headers={
            "content-type": "application/json; charset=utf-8",
            "Accept": "*/*",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36",
        },
    )
    if r.status_code == 200:
        r = r.json()["data"]
        return pd.DataFrame(
            r["transactionTable"]["rows"], columns=r["transactionTable"]["headers"]
        ).drop(columns=["form"])
        # 'numberOfInsiderTrades': pd.DataFrame(r['numberOfTrades']['rows'],
        #                                       columns = r['numberOfTrades']['headers']),
        # 'numberOfInsiderSharesTraded': pd.DataFrame(r['numberOfSharesTraded']['rows'],
        #                                             columns = r['numberOfSharesTraded']['headers']),
        # 'insiderTransactionTable': pd.DataFrame(r['transactionTable']['rows'],
        #                                         columns = r['transactionTable']['headers']).drop(columns = ['form']),

    else:
        return f"Error {r.reason} while trying to get insider trading from nasdaq"


def get_analystPrice(symbol):
    r = requests.get(
        f"https://www.tipranks.com/api/stocks/getChartPageData/?ticker={symbol}",
        headers={  # "content-type":"application/x-www-form-urlencoded",
            "Accept": "*/*",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36",
        },
    )
    # requests.get("https://www.tipranks.com/api/stocks/getData/?name=TSLA&benchmark=1&period=1&break=1591096928923").json()
    if r.status_code == 200:
        return r.json()["analystPriceTarget"]
    else:
        return f"Error {r.reason} while getting analyst price target from tipranks"


def get_trends(symbols: list) -> pd:
    pytrends = TrendReq(hl="en-US", tz=360)
    df = pd.DataFrame()
    if isinstance(symbols, list):
        symbols = [f"{symbol} stock" for symbol in symbols]
        group = [symbols[x: x + 5] for x in range(0, len(symbols), 5)]
    else:
        group = [[f"{symbols} stock"]]
    for each in group:
        pytrends.build_payload(each, cat=0, timeframe="now 7-d", geo="", gprop="")
        data = pytrends.interest_over_time()
        df = pd.concat([df, data.drop(columns=["isPartial"])], axis=1, sort=False)
    return df


def get_news(
    asset: str,
    start=datetime.now().strftime("%m/%d/%Y"),
    end=datetime.now().strftime("%m/%d/%Y"),
) -> pd.DataFrame:
    googlenews = GoogleNews(start=start, end=end, lang="en-US")
    googlenews.search(f"{asset} stock")
    df = pd.DataFrame(googlenews.result())
    if not df.empty:
        df = df[["title", "date", "desc", "link"]]
    return df


def checkSuspension(symbol):
    r = requests.get(
        "https://listingcenter.nasdaq.com/IssuersPendingSuspensionDelisting.aspx",
        headers={
            "content-type": "application/json; charset=utf-8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36",
        },
    )
    if r.status_code == 200:
        r = pd.read_html(r.text)[4]
        r = r.loc[r.Symbol == symbol]
        if not r.empty:
            return r
        else:
            return False
    return f"Error {r.reason} while checking suspension from nasdaq"


def get_stats(symbol):
    try:
        df = pd.read_html(
            f"https://finance.yahoo.com/quote/{symbol}/key-statistics?p={symbol}"
        )
        shareStatistics = (
            df[2].rename(columns={0: "data", 1: "values"}).set_index("data")
        )
        priceHistory = df[1].rename(columns={0: "data", 1: "values"}).set_index("data")
        incomeStatement = (
            df[7].rename(columns={0: "data", 1: "values"}).set_index("data")
        )
        ballanceSheet = df[8].rename(columns={0: "data", 1: "values"}).set_index("data")
        stats = pd.concat(
            [shareStatistics, priceHistory, incomeStatement, ballanceSheet]
        )
        stats.index = [
            each.rsplit(" ", 1)[0] if each[-1].isdigit() else each
            for each in stats.index
        ]
    except Exception as e:
        return f"Error {e} while getting stats from yahoo finance"
    else:
        return stats


def check_candlestick_patterns(df):
    CANDLESTICK_PATTERNS = {
        "hammer": {"check": CDLHAMMER, "meaning": "trend reversal / bullish"},
        "hanging_man": {"check": CDLHANGINGMAN, "meaning": "trend reversal / bearish"},
        "engulfing": {"check": CDLENGULFING, "meaning": "trend reversal"},
        "dark_cloud": {"check": CDLDARKCLOUDCOVER, "meaning": "bearish"},
        "piercing": {"check": CDLPIERCING, "meaning": "bullish"},
        "harami": {"check": CDLHARAMI, "meaning": "trend exhaustion"},
        "harami_cross": {"check": CDLHARAMICROSS, "meaning": "trend exhaustion"},
        "evening_star": {"check": CDLEVENINGSTAR, "meaning": "bearish"},
        "evening_doji_star": {"check": CDLEVENINGDOJISTAR, "meaning": "bearish"},
        "morning_star": {"check": CDLMORNINGSTAR, "meaning": "bullish"},
        "morning_doji_star": {"check": CDLMORNINGDOJISTAR, "meaning": "bullish"},
        "long_legged_doji": {
            "check": CDLLONGLEGGEDDOJI,
            "meaning": "trend reversal / indecision",
        },
        "gravestone": {
            "check": CDLGRAVESTONEDOJI,
            "meaning": "trend reversal / indecision",
        },
        "spinning_tops": {
            "check": CDLSPINNINGTOP,
            "meaning": "trend reversal / indecision",
        },
        "high_wave": {"check": CDLHIGHWAVE, "meaning": "trend reversal / indecision"},
        "doji": {"check": CDLDOJI, "meaning": "trend reversal / indecision"},
    }

    df_patterns, patterns = (
        pd.DataFrame(
            {
                each: list(
                    CANDLESTICK_PATTERNS[each]["check"](
                        df["open"], df["high"], df["low"], df["close"]
                    )
                )
                for each in CANDLESTICK_PATTERNS
            },
            index=df.index,
        ),
        {each: CANDLESTICK_PATTERNS[each]["meaning"] for each in CANDLESTICK_PATTERNS},
    )

    data = {"Datetime": [], "patterns": []}
    for index, row in df_patterns.iterrows():
        if row.any():
            data["Datetime"].append(index)
            temp = ""
            for pattern in df_patterns[-1:].columns:
                if row[f"{pattern}"]:
                    temp += (
                        f"{pattern}({row[F'{pattern}']})  {patterns[F'{pattern}']}, "
                    )
            data["patterns"].append(temp)

    return pd.DataFrame.from_dict(data).set_index("Datetime")


def sendEmail(subject, text=None, html=None, path=None):
    sender = "user@gmail.com"
    gmail_password = "password"
    recipients = ["another_user@gmail.com"]
    outer = MIMEMultipart("alternative")
    outer["Subject"] = subject
    outer["To"] = ", ".join(recipients)
    outer["From"] = sender
    if text:
        outer.attach(MIMEText(text, "plain"))
    if html:
        outer.attach(MIMEText(html, "html"))
    # outer.preamble = 'You will not see this in a MIME-aware mail reader.\n'

    if path:
        attachments = [os.path.join(path, f) for f in os.listdir(path)]
        for file in attachments:
            try:
                with open(file, "rb") as fp:
                    msg = MIMEBase("application", "octet-stream")
                    msg.set_payload(fp.read())
                encoders.encode_base64(msg)
                msg.add_header(
                    "Content-Disposition", "attachment", filename=os.path.basename(file)
                )
                outer.attach(msg)
            except Exception as e:
                outer.attach(MIMEText(f"Unable to open {file} : {e}", "plain"))

    composed = outer.as_string()

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(sender, gmail_password)
            s.sendmail(sender, recipients, composed)
            s.close()
    except Exception as e:
        print(f"Unable to send the email. Error: {e}")


def sr_levels(df, type="filtered"):
    def isSupport(df, i):
        support = (
            df["low"][i] < df["low"][i - 1]
            and df["low"][i] < df["low"][i + 1]
            and df["low"][i + 1] < df["low"][i + 2]
            and df["low"][i - 1] < df["low"][i - 2]
        )
        return support

    def isResistance(df, i):
        resistance = (
            df["high"][i] > df["high"][i - 1]
            and df["high"][i] > df["high"][i + 1]
            and df["high"][i + 1] > df["high"][i + 2]
            and df["high"][i - 1] > df["high"][i - 2]
        )
        return resistance

    def isFarFromLevel(l, s):
        return np.sum([abs(l - x) < s for x in levels]) == 0

    levels = []
    if type == "raw":
        for i in range(2, df.shape[0] - 2):
            if isSupport(df, i):
                levels.append((i, df["low"][i]))
            elif isResistance(df, i):
                levels.append((i, df["high"][i]))
    elif type == "filtered":
        s = np.mean(df["high"] - df["low"])
        for i in range(2, df.shape[0] - 2):
            if isSupport(df, i):
                l = df["low"][i]
                if isFarFromLevel(l, s):
                    levels.append((i, l))
            elif isResistance(df, i):
                l = df["high"][i]
                if isFarFromLevel(l, s):
                    levels.append((i, l))
    return levels


def plot_live(df, size=(10, 8), useRTH=True):
    x = [i for i in range(len(df.index))]

    fig, (ax1, ax2, ax3, ax4) = mpf.plot(
        df,
        type="candle",
        style="charles",
        volume=True,
        show_nontrading=not useRTH,
        ylabel="",
        ylabel_lower="",
        returnfig=True,
        closefig=True,
    )
    fig.set_size_inches(size)

    ax1.text(
        0.1,
        0.9,
        f"RSI : {[int(x) for x in df.RSI[-5:].values]}",
        fontsize=12,
        transform=ax1.transAxes,
        ha="center",
        va="center",
        bbox={"facecolor": "green", "alpha": 0.5}
        if df.RSI[-1] <= 20
        else {"facecolor": "red", "alpha": 0.5}
        if df.RSI[-1] >= 80
        else {"alpha": 0.5},
    )
    ax1.text(
        0.1,
        0.8,
        f"MACD : {round(df.MACD[-2],5)} , {round(df.MACD[-1],5)}",
        fontsize=12,
        transform=ax1.transAxes,
        ha="center",
        va="center",
        bbox={"facecolor": "green", "alpha": 0.5}
        if df.MACD[-1] >= 0
        else {"facecolor": "red", "alpha": 0.5},
    )
    ax1.set_zorder(1)

    try:
        ax1.plot(
            x,
            df.VWAP,
            color="orange",
            linestyle="solid",
            linewidth=2,
            markersize=1,
            label="VWAP",
        )
    except Exception as e:
        pass

    ax1.plot(
        x,
        df.SMA_9,
        color="cyan",
        linestyle="solid",
        linewidth=1,
        markersize=1,
        label="SMA_9",
    )
    ax1.plot(
        x,
        df.SMA_200,
        color="red",
        linestyle="solid",
        linewidth=1,
        markersize=1,
        label="SMA_200",
    )
    try:
        ax1.scatter(x, df.Extreme_high, color="orange", marker="*")
        ax1.scatter(x, df.Extreme_low, color="orange", marker="*")
    except Exception as e:
        pass
    for level in sr_levels(df, type="filtered"):
        ax1.hlines(
            level[1],
            xmin=x[level[0]],
            xmax=x[-1],
            colors="black",
            linewidth=5,
            alpha=0.6,
        )

    ax3.plot(
        x,
        df.avg_vol_10,
        color="black",
        linestyle="solid",
        linewidth=2,
        markersize=1,
        label="Avg volume 10",
        alpha=0.8,
    )

    return fig


def plot_main_chart(df, sr=True):
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=[3, 1, 0.5, 1, 0.5],
    )
    fig.update_layout(
        height=900,
        width=1500,
        # title_text="",
        autosize=False,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(automargin=True)
    fig.update_layout(dict(template="plotly_white"))

    ###############################################################################
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Candles",
        ),
        row=1,
        col=1,
    )

    ###############################################################################
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="volume",
            marker_color=[
                "green" if x >= 0 else "red" for x in df.close * 100 / df.open - 100
            ],
        ),
        row=4,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df["avg_vol_10"],
            mode="lines",
            name="avg_vol_10",
            visible="legendonly",
            marker=dict(color="black"),
        ),
        row=4,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df["avg_vol_60"],
            mode="lines",
            name="avg_vol_60",
            visible="legendonly",
            marker=dict(color="black"),
        ),
        row=4,
        col=1,
    )

    ###############################################################################
    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df["SMA_9"],
            mode="lines",
            name="SMA_9",
            visible=True,
            marker=dict(color="cyan"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df["SMA_20"],
            mode="lines",
            name="SMA_20",
            visible="legendonly",
            marker=dict(color="#FA0F13"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df["SMA_50"],
            mode="lines",
            name="SMA_50",
            visible="legendonly",
            marker=dict(color="#FA0F13"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df["SMA_200"],
            mode="lines",
            name="SMA_200",
            visible="legendonly",
            marker=dict(color="#FA0F13"),
        ),
        row=1,
        col=1,
    )

    ###############################################################################
    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df["SAR"],
            mode="markers",
            name="SAR",
            visible="legendonly",
            marker=dict(color="cyan", size=5),
        ),
        row=1,
        col=1,
    )

    ###############################################################################
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["MACD"],
            name="MACD",
            visible=True,
            marker_color=["green" if x >= 0 else "red" for x in df["MACD"]],
        ),
        row=2,
        col=1,
    )

    ###############################################################################
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[50] * len(df.index),
            mode="lines",
            name="Fair value",
            opacity=0.7,
            showlegend=False,
            visible=True,
            marker=dict(color="#AD8800", size=0.1),
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            mode="lines+markers",
            line_color="black",
            opacity=0.75,
            name="RSI",
            visible=True,
            marker=dict(
                color=[
                    "red" if x <= 30 else "red" if x >= 70 else "black"
                    for x in df["RSI"].values
                ],
                size=6,
            ),
        ),
        row=3,
        col=1,
    )

    ###############################################################################
    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df["BB_upper"],
            mode="lines",
            name="BB_upper",
            visible="legendonly",
            marker=dict(color="yellow"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df["BB_lower"],
            mode="lines",
            name="BB_lower",
            visible="legendonly",
            marker=dict(color="yellow"),
        ),
        row=1,
        col=1,
    )

    ###############################################################################
    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df["Extreme_high"],
            mode="lines+markers",
            name="Extreme_high",
            visible="legendonly",
            marker=dict(color="black", size=10),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df["Extreme_low"],
            mode="lines+markers",
            name="Extreme_low",
            visible="legendonly",
            marker=dict(color="black", size=10),
        ),
        row=1,
        col=1,
    )

    ###############################################################################
    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df["ATR"],
            mode="lines",
            name="ATR",
            visible=True,
            marker=dict(color="black", size=10),
        ),
        row=5,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df["CO"],
            mode="lines",
            name="%CO",
            visible="legendonly",
            marker=dict(color="black", size=10),
        ),
        row=5,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df["OH"],
            mode="lines",
            name="%OH",
            visible="legendonly",
            marker=dict(color="black", size=10),
        ),
        row=5,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df["OC"],
            mode="lines",
            name="%OC",
            visible="legendonly",
            marker=dict(color="black", size=10),
        ),
        row=5,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=df.index,
            y=df["OL"],
            mode="lines",
            name="%OL",
            visible="legendonly",
            marker=dict(color="black", size=10),
        ),
        row=5,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[0] * len(df.index),
            mode="lines",
            opacity=0.7,
            visible=True,
            showlegend=False,
            marker=dict(color="#AD8800", size=0.1),
        ),
        row=5,
        col=1,
    )

    if "barCount" in df.columns:
        fig.add_trace(
            go.Scattergl(
                x=df.index,
                y=df["barCount"],
                mode="lines",
                name="barCount",
                visible="legendonly",
                marker=dict(color="black", size=10),
            ),
            row=5,
            col=1,
        )

    if sr:
        for level in sr_levels(df, type="filtered"):
            fig.add_shape(
                type="line",
                x0=df.index[level[0]],
                y0=level[1],
                x1=df.index[-1],
                y1=level[1],
                opacity=0.3,
                line=dict(
                    color="Black",
                    width=6,
                    dash="solid",
                ),
            )

    ###############################################################################
    return fig


def plot_ohlc_variation(df):
    fig = go.Figure()

    fig.update_layout(
        height=500,
        width=1500,
        autosize=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_yaxes(automargin=True)
    fig.update_layout(dict(template="plotly_white"))

    for column in [
        "CO",
        "OH",
        "OL",
        "OC",
    ]:
        fig.add_trace(
            go.Violin(
                y=df[column],
                name=column,
                box_visible=True,
                points="all",
                meanline_visible=True,
            )
        )
    return fig


def plot_volume_analysis(df):
    fig = go.Figure()

    fig.update_layout(
        height=500,
        width=1500,
        autosize=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_yaxes(automargin=True)
    fig.update_layout(dict(template="plotly_white"))

    fig.add_trace(
        go.Violin(
            y=df["volume"],
            name="volume",
            box_visible=True,
            points="all",
            meanline_visible=True,
        )
    )

    fig.add_trace(
        go.Violin(
            y=df["volume"][df["volume"] > df.volume.quantile(0.75)],
            name="volume over 0.75 quantile",
            box_visible=True,
            points="all",
            meanline_visible=True,
        )
    )

    fig.add_trace(
        go.Violin(
            y=df["volume"][df["volume"] > df.volume.quantile(0.95)],
            name="volume over 0.95 quantile",
            box_visible=True,
            points="all",
            meanline_visible=True,
        )
    )

    fig.add_trace(
        go.Violin(
            y=df["volume"][df["volume"] > df.volume.quantile(0.98)],
            name="volume over 0.98 quantile",
            box_visible=True,
            points="all",
            meanline_visible=True,
        )
    )

    fig.add_trace(
        go.Violin(
            y=df["volume"][df["volume"] < df.volume.quantile(0.50)],
            name="volume under 0.5 quantile",
            box_visible=True,
            points="all",
            meanline_visible=True,
        )
    )

    return fig


def add_technicals(df):
    df["fast_ma"] = df.close.ewm(
        span=8, min_periods=8, adjust=False, ignore_na=False
    ).mean()
    # df['fast_ma'] = df.close.rolling(window=9).mean()
    df["slow_ma"] = df.close.ewm(
        span=14, min_periods=14, adjust=False, ignore_na=False
    ).mean()
    df["direction"] = df.close.rolling(window=89).mean()
    df["atr"] = ATR(df[["high", "low", "close"]], timeperiod=10)
    return df


def change_context(df, timeframe):
    df = df.resample(timeframe).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    df.dropna(how="any", inplace=True)
    return df


def heikin_ashi(df):
    df_HA = df.copy()
    df_HA["close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

    for i in range(0, len(df)):
        if i == 0:
            df_HA["open"][i] = (df["open"][i] + df["close"][i]) / 2
        else:
            df_HA["open"][i] = (df["open"][i - 1] + df["close"][i - 1]) / 2

    df_HA["high"] = df[["open", "close", "high"]].max(axis=1)
    df_HA["low"] = df[["open", "close", "low"]].min(axis=1)
    return df_HA


def make_chart(
    df=pd.DataFrame({}),
    levels=pd.DataFrame({}),
    df_2=pd.DataFrame({}),
    levels_2=pd.DataFrame({}),
    size=(1700, 900),
    title="",
):
    fig = make_subplots(
        rows=1 if (df["volume"] == 0).all() else 2,
        cols=2 if not df_2.empty else 1,
        shared_yaxes=False,
        shared_xaxes=True,
        horizontal_spacing=0.05,
        vertical_spacing=0.01,
        row_heights=[1] if (df["volume"] == 0).all() else [0.87, 0.13],
        column_widths=[0.5, 0.5] if not df_2.empty else [1],
    )

    layout = {
        "height": size[1],
        "width": size[0],
        "template": "plotly_dark",
        "margin": {"l": 0, "r": 0, "b": 0, "t": 0},
        "autosize": False,
        "dragmode": "pan",
        "title": title,
        # "dragmode": "drawrect",
        "hovermode": "closest",
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "xanchor": "right",
            "y": 1.05,
            "x": 0.8,
        },
        "xaxis_rangeslider_visible": False,
        "xaxis": {
            "type": "category",
            "showgrid": False,
            "visible": False,
            "showticklabels": False,
            "showspikes": True,
        },
        "yaxis": {"automargin": True, "showspikes": True},
        "yaxis2": {"automargin": True, "showspikes": True},
    }

    if not (df["volume"] == 0).all() or not df_2.empty:
        layout = {
            **layout,
            **{
                "xaxis2_rangeslider_visible": False,
                "xaxis2": {
                    "type": "category",
                    "showgrid": False,
                    "visible": False,
                    "showticklabels": False,
                    "showspikes": True,
                },
                "xaxis3": {
                    "type": "category",
                    "showgrid": False,
                    "visible": False,
                    "showticklabels": False,
                    "showspikes": True,
                },
                "xaxis4": {
                    "type": "category",
                    "showgrid": False,
                    "visible": False,
                    "showticklabels": False,
                    "showspikes": True,
                },
            },
        }

    fig.update_layout(dict(layout))

    fig.add_candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Candles",
        row=1,
        col=1,
    )

    fig.add_scattergl(
        x=df.index,
        y=df["close"],
        mode="lines",
        name="close",
        visible="legendonly",
        marker=dict(color="#000FFF"),
        line=dict(width=5),
        row=1,
        col=1,
    )

    fig.add_scattergl(
        x=df.index,
        y=df["fast_ma"],
        mode="lines",
        name="fast_ma",
        visible="legendonly",
        marker=dict(color="#00ff00"),
        row=1,
        col=1,
    )

    fig.add_scattergl(
        x=df.index,
        y=df["slow_ma"],
        mode="lines",
        name="slow_ma",
        visible="legendonly",
        marker=dict(color="#ff0000"),
        row=1,
        col=1,
    )

    fig.add_scattergl(
        x=df.index,
        y=df["direction"],
        mode="lines",
        name="direction",
        visible="legendonly",
        marker=dict(color="cyan"),
        row=1,
        col=1,
    )

    fig.add_scattergl(
        x=df.index,
        y=(df["close"] + 1 * df["atr"]).shift(1),
        mode="lines",
        name="short_sl",
        visible="legendonly",
        marker=dict(color="white"),
        row=1,
        col=1,
    )

    fig.add_scattergl(
        x=df.index,
        y=(df["close"] - 1 * df["atr"]).shift(1),
        mode="lines",
        name="long_sl",
        visible="legendonly",
        marker=dict(color="white"),
        row=1,
        col=1,
    )

    if not (df["volume"] == 0).all():
        fig.add_bar(
            x=df.index,
            y=df["volume"],
            name="volume",
            marker_color=[
                "green" if x >= 0 else "red" for x in df.close * 100 / df.open - 100
            ],
            row=2,
            col=1,
        )

    for index, level in levels.iterrows():
        fig.add_shape(
            type="rect",
            x0=index if not index < df.index[0] else df.index[0],
            y0=level["end"],
            x1=df.index[-1],
            y1=level["start"],
            opacity=0.2,
            fillcolor="Cyan",
            row=1,
            col=1,
        )
        fig.add_shape(
            type="line",
            x0=index if not index < df.index[0] else df.index[0],
            y0=(level["end"] + level["start"]) / 2,
            x1=df.index[-1],
            y1=(level["end"] + level["start"]) / 2,
            opacity=0.4,
            line=dict(color="White", width=2, dash="solid"),
            row=1,
            col=1,
        )

    if not df_2.empty:
        fig.add_candlestick(
            x=df_2.index,
            open=df_2["open"],
            high=df_2["high"],
            low=df_2["low"],
            close=df_2["close"],
            name="Candles",
            row=1,
            col=2,
        )

        fig.add_scattergl(
            x=df_2.index,
            y=df_2["close"],
            mode="lines",
            name="close",
            visible="legendonly",
            marker=dict(color="#000FFF"),
            line=dict(width=5),
            row=1,
            col=2,
        )

        fig.add_scattergl(
            x=df_2.index,
            y=df_2["fast_ma"],
            mode="lines",
            name="fast_ma",
            visible="legendonly",
            marker=dict(color="#00ff00"),
            row=1,
            col=2,
        )

        fig.add_scattergl(
            x=df_2.index,
            y=df_2["slow_ma"],
            mode="lines",
            name="slow_ma",
            visible="legendonly",
            marker=dict(color="#ff0000"),
            row=1,
            col=2,
        )

        fig.add_scattergl(
            x=df_2.index,
            y=df_2["direction"],
            mode="lines",
            name="direction",
            visible="legendonly",
            marker=dict(color="cyan"),
            row=1,
            col=2,
        )

        fig.add_scattergl(
            x=df_2.index,
            y=(df_2["close"] + 1 * df_2["atr"]).shift(1),
            mode="lines",
            name="short_sl",
            visible="legendonly",
            marker=dict(color="white"),
            row=1,
            col=2,
        )

        fig.add_scattergl(
            x=df_2.index,
            y=(df_2["close"] - 1 * df_2["atr"]).shift(1),
            mode="lines",
            name="long_sl",
            visible="legendonly",
            marker=dict(color="white"),
            row=1,
            col=2,
        )

        if not (df_2["volume"] == 0).all():
            fig.add_bar(
                x=df_2.index,
                y=df_2["volume"],
                name="volume",
                marker_color=[
                    "green" if x >= 0 else "red"
                    for x in df_2.close * 100 / df_2.open - 100
                ],
                row=2,
                col=2,
            )

        for index, level in levels_2.iterrows():
            fig.add_shape(
                type="rect",
                x0=index if not index < df_2.index[0] else df_2.index[0],
                y0=level["end"],
                x1=df_2.index[-1],
                y1=level["start"],
                opacity=0.2,
                fillcolor="Cyan",
                row=1,
                col=2,
            )
            fig.add_shape(
                type="line",
                x0=index if not index < df_2.index[0] else df_2.index[0],
                y0=(level["end"] + level["start"]) / 2,
                x1=df_2.index[-1],
                y1=(level["end"] + level["start"]) / 2,
                opacity=0.4,
                line=dict(color="White", width=2, dash="solid"),
                row=1,
                col=2,
            )

    return fig


def calculate_levels(df, type="filtered", spacing=1.2, thickness=0.01):
    def isSupport(df, i):
        support = (
            df["low"][i] < df["low"][i - 1]
            and df["low"][i] < df["low"][i + 1]
            and df["low"][i + 1] < df["low"][i + 2]
            and df["low"][i - 1] < df["low"][i - 2]
        )
        return support

    def isResistance(df, i):
        resistance = (
            df["high"][i] > df["high"][i - 1]
            and df["high"][i] > df["high"][i + 1]
            and df["high"][i + 1] > df["high"][i + 2]
            and df["high"][i - 1] > df["high"][i - 2]
        )
        return resistance

    def isFarFromLevel(l, s):
        return np.sum([abs(l - x) < s for x in levels]) == 0

    levels = []
    mirror = []
    if type == "raw":
        for i in range(2, df.shape[0] - 2):
            if isSupport(df, i):
                levels.append((i, df["low"][i]))
            elif isResistance(df, i):
                levels.append((i, df["high"][i]))
    elif type == "filtered":
        s = np.mean(df["high"] - df["low"]) / spacing
        for i in range(2, df.shape[0] - 2):
            if isSupport(df, i):
                l = df["low"][i]
                if isFarFromLevel(l, s):
                    levels.append((i, l))
                    mirror.append((i, l, "s"))  # am adaugat astea..
            elif isResistance(df, i):
                l = df["high"][i]
                if isFarFromLevel(l, s):
                    levels.append((i, l))
                    mirror.append((i, l, "r"))

    level_size = (df.high.max() - df.low.min()) * thickness

    temp = []
    for level in mirror:
        #         temp.append((df.index[level[0]],
        #                             level[1]-level_size,
        #                             level[1]+level_size ))
        temp.append(
            (
                df.index[level[0]],
                level[1] - 2 * level_size
                if level[2] == "r"
                else level[1] + 2 * level_size,
                level[1],
            )
        )

    return pd.DataFrame(temp, columns=["Datetime", "start", "end"]).set_index(
        "Datetime"
    )  # thickness=.0125


PandasObject.add_technicals = add_technicals
PandasObject.change_context = change_context
PandasObject.heikin_ashi = heikin_ashi
PandasObject.calculate_levels = calculate_levels
PandasObject.clean = clean
