#!/usr/bin/env python

from dataclasses import astuple
from datetime import datetime
from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto

import asyncio
##### Additional libraries required #####
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm
import math
#########################################

option_strikes = [90, 95, 100, 105, 110]

#############################################################################
#               calculating the greeks
#
#               S       is underlying spot price
#               sigma   is volatility
#               K       is strike price
#               r       is interest rate (which is 0)
#               t       is th time for expiry
#############################################################################
# defining the greeks with the modularized functions for program-wide usage
def d(sigma, S, K, r, t):
    d1 = 1 / (sigma * np.sqrt(t)) * ( np.log(S / K) + (r + sigma**2 / 2) * t)
    d2 = d1 - sigma * np.sqrt(t)
    return d1, d2

def vega(sigma, S, K, r, t):
    d1, d2 = d(sigma, S, K, r, t)
    v = S * norm.pdf(d1) * np.sqrt(t)
    return v

def delta(d1, option_type):
    if option_type == 'C':
        return norm.cdf(d1)
    if option_type == 'P':
        return -norm.cdf(-d1)
    
def gamma(d2, S, K, sigma, r, t):
    return(K * np.exp(-r * t) * (norm.pdf(d2) / (S**2 * sigma * np.sqrt(t)))) 

def theta(d1, d2, S, K, sigma, r, t, option_type):
    if option_type == 'C':
        theta = -S * sigma * norm.pdf(d1) / (2 * np.sqrt(t)) - r * K * np.exp(-r * t) * norm.cdf(d2)
    if option_type == 'P':
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

#####################################################################################################

class Case2ExampleBot(UTCBot):

    async def handle_round_started(self):
        """
        This function is called when the round is started. You should do your setup here, and
        start any tasks that should be running for the rest of the round.
        """

        # This variable will be a map from asset names to positions
        self.positions = {}
        self.positions["UC"] = 0
        for strike in option_strikes:
            for option_type in ["C", "P"]:
                self.positions[f"UC{strike}{option_type}"] = 0
                
        self.options_prices = {}
        self.options_prices["UC"] = 0
        for strike in option_strikes:
            for option_type in ["C", "P"]:
                self.options_prices[f"UC{strike}{option_type}"] = 0

        self.i_vol = {}
        for strike in option_strikes:
            self.i_vol[f"{strike}"] = .366
        
        
        self.implied_vol = 0.366
        

        self.current_day = 0
        self.pos_delta = 0
        self.pos_gamma = 0
        self.pos_theta = 0
        self.pos_vega = 0
        self.underlying_price = 0
        self.spread = 0
        self.time_to_expiry = 0

    def compute_vol_estimate(self) -> float:
        """
        This function is used to provide an estimate of underlying's volatility. Because this is
        an example bot, we just use a placeholder value here. We recommend that you look into
        different ways of finding what the true volatility of the underlying is.
        """
        # have to stay within stringent risk limits, 
        # specifically regarding extreme delta and vega values.
        # The method below should update delta and vega dictionaries associated with each option in the option chain and also compute total vol.
        # To compute total vol, the method loops through the entire options chain and sums up the implied volatilities using the previously defined implied_vol function.
        
        ######################## CODE ADDED ###############################
        # Still thinking about this stuff, it is not working right now though so I am currently using just constant volatility
        # The thing is some transactions show error due to the risk limit getting exceeded, which I am trying to handle using the code below 
        '''
        
        '''

        ############################################################################
        return 0.366 # (WAS FOR THE EXAMPLE BOT, so I commented it out here)

    def compute_options_price(
        self,
        option_type: str,
        underlying_px: float,
        strike_px: float,
        time_to_expiry: float,
        volatility: float,
    ) -> float:
        """
        This function should compute the price of an option given the provided parameters. Some
        important questions you may want to think about are:
            - What are the units associated with each of these quantities?
            - What formula should you use to compute the price of the option?
            - Are there tricks you can use to do this more quickly?
        You may want to look into the py_vollib library, which is installed by default in your
        virtual environment.
        """
        ##################### CODE ADDED BY DHYEY #######################
        d1, d2 = d(volatility, underlying_px, strike_px, 0, time_to_expiry)
        if (option_type == "P"):
            return round(
                put_price(
                    volatility, underlying_px, strike_px, 0, time_to_expiry, d1, d2
                ), 1)
        elif (option_type == "C"):
            return round(
                call_price(
                    volatility, underlying_px, strike_px, 0, time_to_expiry, d1, d2
                ), 1)
        #################################################################
        return 1.0

    async def update_options_quotes(self):
        """
        This function will update the quotes that the bot has currently put into the market.

        In this example bot, the bot won't bother pulling old quotes, and will instead just set new
        quotes at the new theoretical price every time a price update happens. We don't recommend
        that you do this in the actual competition
        """
        # Because of penalties imposed for violating risk limits.
        # Therefore it is essential to develop a strategy that manages delta and vega risk in a comprehensive manner. 
        # This particular delta hedging strategy adjusts bid and ask prices using thresholds calculated from linear models.
        # The goal is to reduce positive delta approaching the risk limit by 
        # pushing up call prices and pulling down put prices, or doing the opposite to neutralize radically negative delta. 
        # For example, if total delta across the options chain approaches the upper limit, the algorithm should adjust pricing to supply attractive asks on calls and attractive bids on puts. 
        # This will lead to greater sell volume on calls and buy volume on puts, in turn reducing delta. 
        # The variables c_bid_p_ask_threshold and c_ask_p_bid_threshold adjust final bid and ask order prices to reflect this thought process. 
        # The specific linear function was derived through modeling of historical price data and guess check mix.
        
        
        # What should this value actually be?
        # time_to_expiry = 21 / 252 (Given by UChicago)
        self.time_to_expiry = (21 + 5 - self.current_day) / 252 # updated logic
        #vol = self.implied_vol
        vol = self.compute_vol_estimate()

        thresh_val = .25/2000 ##### ADDED #####
        requests = []
        self.pos_delta = 0
        self.pos_gamma = 0
        self.pos_theta = 0
        self.pos_vega = 0
        for strike in option_strikes:
            for option_type in ["C", "P"]:
                d1,d2 = d(vol, self.underlying_price, strike, 0, self.time_to_expiry)
                position = self.positions[f"UC{strike}{option_type}"]
                self.pos_delta += delta(d1,option_type) * position *100

        for strike in option_strikes:
            for option_type in ["C", "P"]:
                asset_name = f"UC{strike}{option_type}"
                theo = self.compute_options_price(
                    option_type, self.underlying_price, strike, self.time_to_expiry, vol#self.i_vol[str(strike)]
                )
                

                ########## ADDED ##########
                # calculate price threshold used in bid and ask orders
                #c_bid_p_ask_thresh = round((thresh_val)*(self.pos_delta)+.30,1)
                c_bid_p_ask_thresh = 0.3
                self.spread = c_bid_p_ask_thresh
                # calculate order quantity based on position held currently
                position = self.positions[f"UC{strike}{option_type}"]
                
                d1,d2 = d(vol, self.underlying_price, strike, 0, self.time_to_expiry)
                
                
                #self.pos_gamma += gamma(d2, self.underlying_price, strike, vol, 0, self.time_to_expiry) * position
                #self.pos_theta += theta(d1, d2, self.underlying_price, strike, vol, 0, self.time_to_expiry, option_type) * position
                #self.pos_vega += vega(vol, self.underlying_price, strike, 0, self.time_to_expiry) * position
                '''
                if (position<0):
                    if (position>-100):
                        buy_quantity = 1
                    else:
                        if round((position**2)/4000) > 15:
                            buy_quantity = 15
                        else:
                            buy_quantity = round((position**2)/4000)
                        #buy_quantity = round((position**2)/4000)
                    sell_quantity = 1
                elif(position>=0):
                    if (position<100):
                        sell_quantity = 1
                    else:
                        if round((position**2)/4000) > 15:
                            sell_quantity = 15
                        else:
                            sell_quantity = round((position**2)/4000)
                        #sell_quantity = round((position**2)/4000)
                    buy_quantity = 1
                '''
                

                buy_quantity = 1
                sell_quantity = 1
                if option_type == "P":
                    if self.pos_delta > 500:
                        buy_quantity = min(15,int(15* (self.pos_delta)//2000))
                    elif self.pos_delta < -500:
                        sell_quantity = min(15,int(15 *(-self.pos_delta)//2000))
                else:
                    if self.pos_delta > 500:
                        sell_quantity = min(15,int(15* (self.pos_delta)//2000))
                    elif self.pos_delta < -500:
                        buy_quantity = min(15,int(15 *(-self.pos_delta)//2000))
                

                #print(position)
                ''' 
                # continuously place bid and ask orders
                if(option_type=="C"):
                    bid_response = await self.place_order(
                        asset_name,
                        pb.OrderSpecType.LIMIT,
                        pb.OrderSpecSide.BID,
                        buy_quantity,
                        theo - c_bid_p_ask_thresh,
                    )
                    assert bid_response.ok
                    ask_response = await self.place_order(
                        asset_name,
                        pb.OrderSpecType.LIMIT,
                        pb.OrderSpecSide.ASK,
                        sell_quantity,
                        theo + c_ask_p_bid_thresh,
                    )
                    assert ask_response.ok
                elif(option_type=="P"):
                    bid_response = await self.place_order(
                        asset_name,
                        pb.OrderSpecType.LIMIT,
                        pb.OrderSpecSide.BID,
                        buy_quantity,
                        theo - c_ask_p_bid_thresh,
                    )
                    assert bid_response.ok
                    ask_response = await self.place_order(
                        asset_name,
                        pb.OrderSpecType.LIMIT,
                        pb.OrderSpecSide.ASK,
                        sell_quantity,
                        theo + c_bid_p_ask_thresh,
                    )
                    assert ask_response.ok
                ###########################
                '''
                # Another Probable way it can be done
                if (option_type == "C"):
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
                elif (option_type == "P"):
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
                

                '''
                ######### STUB CODE PROVIDED BY UCHICAGO #########
                requests.append(
                    self.place_order(
                        asset_name,
                        pb.OrderSpecType.LIMIT,
                        pb.OrderSpecSide.BID,
                        1,  # How should this quantity be chosen?
                        theo - 0.30,  # How should this price be chosen?
                    )
                )

                requests.append(
                    self.place_order(
                        asset_name,
                        pb.OrderSpecType.LIMIT,
                        pb.OrderSpecSide.ASK,
                        1,
                        theo + 0.30,
                    )
                )
                ##################################################
                '''
        print("delta: " + str(self.pos_delta))
        ######################### Provided by UCHICAGO ##############################
        # optimization trick -- use asyncio.gather to send a group of requests at the same time
        # instead of sending them one-by-one
        responses = await asyncio.gather(*requests)
        for resp in responses:
            assert resp.ok
        # self.pos_delta = 0
        ##############################################################################
        

    ######################### Provided by UCHICAGO #########################
    async def handle_exchange_update(self, update: pb.FeedMessage):
        kind, _ = betterproto.which_one_of(update, "msg")

        if kind == "pnl_msg":
            # When you hear from the exchange about your PnL, print it out
            print("My PnL:", update.pnl_msg.m2m_pnl)

        elif kind == "fill_msg":
            # When you hear about a fill you had, update your positions
            fill_msg = update.fill_msg

            if fill_msg.order_side == pb.FillMessageSide.BUY:
                self.i_vol[fill_msg.asset[2:-1]] -= (update.fill_msg.filled_qty * self.spread) / 500*vega(self.i_vol[fill_msg.asset[2:-1]],self.underlying_price, int(fill_msg.asset[2:-1]), 0, self.time_to_expiry)
                # print("filled qty: " + str(update.fill_msg.filled_qty))
                # print("spread: " + str(self.spread))
                # print("change in vol: " + str((update.fill_msg.filled_qty * self.spread) / 500*vega(self.implied_vol,
                #  self.underlying_price, int(fill_msg.asset[2:-1]), 0, self.time_to_expiry)))
                # print("vega:  " + str(vega(self.implied_vol,
                #  self.underlying_price, int(fill_msg.asset[2:-1]), 0, self.time_to_expiry)))
                self.positions[fill_msg.asset] += update.fill_msg.filled_qty
            else:
                self.positions[fill_msg.asset] -= update.fill_msg.filled_qty
                self.i_vol[fill_msg.asset[2:-1]] += (update.fill_msg.filled_qty * self.spread) / 500*vega(self.i_vol[fill_msg.asset[2:-1]],self.underlying_price, int(fill_msg.asset[2:-1]), 0, self.time_to_expiry)

        elif kind == "market_snapshot_msg":
            # When we receive a snapshot of what's going on in the market, update our information
            # about the underlying price.
            book = update.market_snapshot_msg.books["UC"]

            # Compute the mid price of the market and store it
            self.underlying_price = (
                float(book.bids[0].px) + float(book.asks[0].px)
            ) / 2

            await self.update_options_quotes()

        elif (
            kind == "generic_msg"
            and update.generic_msg.event_type == pb.GenericMessageType.MESSAGE
        ):
            # The platform will regularly send out what day it currently is (starting from day 0 at
            # the start of the case) 
            self.current_day = float(update.generic_msg.message)
    #######################################################################


if __name__ == "__main__":
    start_bot(Case2ExampleBot)
