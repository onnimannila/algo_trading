
# import the needed packages
from itertools import count

import alpaca_trade_api as trade_api
from alpaca_trade_api import REST
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.stream import TradingStream
from conf import config
from trading import strategies
from trading import execution

# prepare the API and REST order
api = trade_api.REST(config.API_Key, config.Secret_key, config.alpaca_base_URL)

client = TradingClient(config.API_Key, config.Secret_key, paper=True)
account = dict(client.get_account())

# run the trading algo by importing the strategies module
# (the strategies module runs its analysis and prints signals when imported)


# execute the trade based on the final trading execution signal
