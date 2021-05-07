Vue.component("tradingview", {
  props: {
    symbol: "NASDAQ:NDAQ"
  },

  render() {
    return new TradingView.widget({
      "autosize": false,
      "symbol": "NASDAQ:NDAQ",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "toolbar_bg": "#f1f3f6",
      "enable_publishing": true,
      "range": "6m",
      "hide_side_toolbar": false,
      "allow_symbol_change": true,
      "studies": [
        "MASimple@tv-basicstudies"
      ],
      "show_popup_button": true,
      "popup_width": "1000",
      "popup_height": "2000",
      "container_id": "tradingview_a1cb1"
    });
  },
  template: `
<!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container">
    <div id="tradingview_a1cb1"></div>
    <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/symbols/NASDAQ-NDAQ/" rel="noopener" target="_blank"><span class="blue-text">NDAQ Chart</span></a> by TradingView</div>

</div>
<!-- TradingView Widget END -->
`
});