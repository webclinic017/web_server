from flask import Flask, render_template, request
from flask.json import jsonify
from flask_debugtoolbar import DebugToolbarExtension
from flask_restful import Api
import pandas as pd
from flask_wtf.csrf import CSRFProtect
from GoogleNews import GoogleNews
from pytrends.request import TrendReq
from datetime import datetime
from base import *

pd.options.plotting.backend = "plotly"


app = Flask(__name__)
api = Api(app)
app.debug = True
app.secret_key = "Super1234top567secret$%&password=?()"
csrf = CSRFProtect(app)
toolbar = DebugToolbarExtension(app)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", context={"title": "Scanner"})


@app.route("/watchlist", methods=["GET"])
def watchlist():
    """
    print(F"Received data :\n {request.get_data()}\n")
    received_data = json.loads(request.data)
    """
    df = pd.DataFrame(
        {
            "symbol": ["TSLA", "ETH/USD", "BIOC", "AAPL"],
            "something1": [3, 4, 22, 7],
            "something2": [1, 6, 12, 30],
            "something3": ["a", "b", "c", "d"],
        }
    )
    return df.to_json(orient="records")


@app.route("/news", methods=["GET"])
def get_news():
    googlenews = GoogleNews(
        start=datetime.now().strftime("%m/%d/%Y"),
        end=datetime.now().strftime("%m/%d/%Y"),
        lang="en-US",
    )
    googlenews.search(request.args.get("symbol"))
    df = pd.DataFrame(googlenews.result())
    if not df.empty:
        return (
            df[["title", "date", "desc", "link"]]
            .sort_values(by=["date"], ascending=True)
            .to_json(orient="records")
        )
    else:
        return jsonify(data={})


@app.route("/trends", methods=["GET"])
def get_trends():
    pytrends = TrendReq(hl="en-US", tz=360)
    pytrends.build_payload(
        [f"{request.args.get('symbol')} stock"],
        cat=0,
        timeframe="now 7-d",
        geo="",
        gprop="",
    )
    data = pytrends.interest_over_time()
    data = data.drop(columns=["isPartial"])
    return data.plot().to_json()


@app.route("/plotly", methods=["GET"])
def plot():
    import plotly.express as px

    fig = px.scatter(x=range(10), y=range(10))
    return fig.to_json()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port="8000",
    )
