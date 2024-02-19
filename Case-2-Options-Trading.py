#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Importing the libraries provided by UChicago

#!/usr/bin/env python

from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto
import asyncio
import json

PARAM_FILE = "params.json"


# In[34]:


# Importing additional libraries 

import pandas as pd
import numpy as np
import math


# In[35]:


# Calculating the greeks

'''
S       is underlying spot price
sigma   is volatility
K       is strike price
r       is interest rate (which is 0)
t       is the time to expiry
'''

def d(sigma, S, K, r, t):
    d1 = 1 / (sigma * np.sqrt(t)) * ( np.log(S / K) + (r + sigma**2 / 2) * t)
    d2 = d1 - sigma * np.sqrt(t)
    return d1, d2

def vega(sigma, S, K, r, t):
    d1, d2 = d(sigma, S, K, r, t)
    v = S * norm.pdf(d1) * np.sqrt(t)
    return v

def delta(d1, option_type):
    if option_type == 'call':
        return norm.cdf(d1)
    if option_type == 'put':
        return -norm.cdf(-d1)
    
def gamma(d2, S, K, sigma, r, t):
    return(K * np.exp(-r * t) * (norm.pdf(d2) / (S**2 * sigma * np.sqrt(t)))) 

def theta(d1, d2, S, K, sigma, r, t, option_type):
    if option_type == 'call':
        theta = -S * sigma * norm.pdf(d1) / (2 * np.sqrt(t)) - r * K * np.exp(-r * t) * norm.cdf(d2)
    if option_type == 'put':
        theta = -S * sigma * norm.pdf(-d1) / (2 * np.sqrt(t)) + r * K * np.exp(-r * t) * norm.cdf(-d2)
    return theta

def call_price(sigma, S, K, r, t, d1, d2):
    C = norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * t)
    return C

def put_price(sigma, S, K, r, t, d1, d2):
    P = -norm.cdf(-d1) * S + norm.cdf(-d2) * K * np.exp(-r * t)
    return P

def implied_vol(sigma, S, K, r, t, bs_price, price):
    val = bs_price - price
    veg = vega(sigma, S, K, r, t)
    vol = -val / veg + sigma
    return vol


# In[36]:


# Defining the strikes

option_strikes = [65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135]


# In[37]:


# Finding initial volatility

price_paths = pd.read_csv("training_pricepaths.csv", index_col = 0)

for i in range(0,599):
    price_paths.loc[i, 'returns'] = ((price_paths.at[i+1, 'underlying'] - price_paths.at[i, 'underlying'])/price_paths.at[i, 'underlying'])*100

spot_returns = price_paths['returns']
spot_returns = spot_returns.dropna()
spot_returns = spot_returns.to_numpy()
std_dev = np.std(spot_returns)
std_dev


# In[39]:


# Defining the class for the Options Trading Bot

class OptionBot(UTCBot):
    """
    The bot reads from a file to set internal parameters during the round
    """

    async def handle_round_started(self):
        """
        This function is called when the round is started. You should do your setup here, and
        start any tasks that should be running for the rest of the round.
        """
        
        # 'positions' is a map from asset names to positions
        self.positions = {}
        self.positions["Underlying"] = 0
        for strike in option_strikes:
            for option_type in ["call", "put"]:
                self.positions[f"{option_type}{strike}"] = 0
                
        # 'option_prices' is a map from asset names to prices
        self.options_prices = {}
        self.options_prices["Underlying"] = 0
        for strike in option_strikes:
            for option_type in ["call", "put"]:
                self.options_prices[f"{option_type}{strike}"] = 0

        # 'i_vol' is a map from strikes to implied volatility
        self.i_vol = {}
        for strike in option_strikes:
            self.i_vol[f"{strike}"] = 0.517 
        
        
        self.implied_vol = 0.517 
        

        self.current_day = 0
        self.pos_delta = 0
        self.pos_gamma = 0
        self.pos_theta = 0
        self.pos_vega = 0
        self.underlying_price = 0
        self.spread = 0
        self.time_to_expiry = 0
        
        
        await asyncio.sleep(0.1)
        asyncio.create_task(self.handle_read_params())
        
        
    def compute_vol_estimate(self) -> float:
        """
        This function is used to provide an estimate of underlying's implied volatility
        """
        
        latest_return = ((self.underlying_price - price_paths.at[price_paths.shape[0]-1, 'underlying'])/price_paths.at[price_paths.shape[0]-1, 'underlying'])*100
        spot_returns.append(latest_return)
        vol = np.std(spot_returns)
        
        return vol
    
    
    def compute_options_price(self, option_type: str, underlying_px: float, strike_px: float, time_to_expiry: float, volatility: float) -> float:
        
        d1, d2 = d(volatility, underlying_px, strike_px, 0, time_to_expiry)
        
        if (option_type == "put"):
            return round(put_price(volatility, underlying_px, strike_px, 0, time_to_expiry, d1, d2), 1)
        elif (option_type == "call"):
            return round(call_price(volatility, underlying_px, strike_px, 0, time_to_expiry, d1, d2), 1)
        
        return 1.0
    
    
    async def update_options_quotes(self):
        """
        This function will update the quotes that the bot has currently put into the market.
        """
        self.time_to_expiry = (63 - self.current_day) / 252
        vol = self.compute_vol_estimate()

        thresh_val = .25/2000
        requests = []
        self.pos_delta = 0
        self.pos_gamma = 0
        self.pos_theta = 0
        self.pos_vega = 0
        
        for strike in option_strikes:
            for option_type in ["call", "put"]:
                d1,d2 = d(vol, self.underlying_price, strike, 0, self.time_to_expiry)
                position = self.positions[f"{option_type}{strike}"]
                self.pos_delta += delta(d1,option_type) * position * 100

        for strike in option_strikes:
            for option_type in ["call", "put"]:
                
                asset_name = f"{option_type}{strike}"
                theo = self.compute_options_price(option_type, self.underlying_price, strike, self.time_to_expiry, vol)
                
                c_bid_p_ask_thresh = round((thresh_val)*(self.pos_delta)+vol,1)
                self.spread = c_bid_p_ask_thresh
                position = self.positions[f"{option_type}{strike}"]
                
                d1,d2 = d(vol, self.underlying_price, strike, 0, self.time_to_expiry)
                

                buy_quantity = 1
                sell_quantity = 1
                if option_type == "put":
                    if self.pos_delta > 500:
                        buy_quantity = min(15,int(15* (self.pos_delta)//2000))
                    elif self.pos_delta < -500:
                        sell_quantity = min(15,int(15 *(-self.pos_delta)//2000))
                else:
                    if self.pos_delta > 500:
                        sell_quantity = min(15,int(15* (self.pos_delta)//2000))
                    elif self.pos_delta < -500:
                        buy_quantity = min(15,int(15 *(-self.pos_delta)//2000))
                

                if (option_type == "call"):
                    requests.append(
                        self.place_order(
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.BID,
                            buy_quantity,
                            theo - c_bid_p_ask_thresh,
                        )
                    )

                    requests.append(
                        self.place_order(
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.ASK,
                            sell_quantity,
                            theo + c_bid_p_ask_thresh,
                        )
                    )
                elif (option_type == "put"):
                    requests.append(
                        self.place_order(
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.BID,
                            buy_quantity,
                            theo - c_bid_p_ask_thresh,
                        )
                    )

                    requests.append(
                        self.place_order(
                            asset_name,
                            pb.OrderSpecType.LIMIT,
                            pb.OrderSpecSide.ASK,
                            sell_quantity,
                            theo + c_bid_p_ask_thresh,
                        )
                    )
                

        responses = await asyncio.gather(*requests)
        for resp in responses:
            assert resp.ok
    
    
    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, _ = betterproto.which_one_of(update, "msg")

        if kind == "pnl_msg":
            print("My PnL:", update.pnl_msg.m2m_pnl)

        elif kind == "fill_msg":
            fill_msg = update.fill_msg

            if fill_msg.order_side == pb.FillMessageSide.BUY:
                self.i_vol[fill_msg.asset[2:-1]] -= (update.fill_msg.filled_qty * self.spread) / 500*vega(self.i_vol[fill_msg.asset[2:-1]],self.underlying_price, int(fill_msg.asset[2:-1]), 0, self.time_to_expiry)
                self.positions[fill_msg.asset] += update.fill_msg.filled_qty
            else:
                self.positions[fill_msg.asset] -= update.fill_msg.filled_qty
                self.i_vol[fill_msg.asset[2:-1]] += (update.fill_msg.filled_qty * self.spread) / 500*vega(self.i_vol[fill_msg.asset[2:-1]],self.underlying_price, int(fill_msg.asset[2:-1]), 0, self.time_to_expiry)

        elif kind == "market_snapshot_msg":
            book = update.market_snapshot_msg.books["underlying"]

            self.underlying_price = (float(book.bids[0].px) + float(book.asks[0].px)) / 2

            await self.update_options_quotes()

        elif (kind == "generic_msg" and update.generic_msg.event_type == pb.GenericMessageType.MESSAGE):
            self.current_day = float(update.generic_msg.message)

    
    async def handle_read_params(self):
        while True:
            try:
                self.params = json.load(open(PARAM_FILE, "r"))
            except:
                print("Unable to read file " + PARAM_FILE)

            await asyncio.sleep(1)


if __name__ == "__main__":
    start_bot(OptionBot)

