from ...src.agent.agent_thread import agent_thread
from ...src.util.util import *

# This is a very simple Agent, it will buy any time possible
# and sell if the current price is higher than buy-in price
class simple_agent(agent_thread):
    def __init__(self):
        agent_thread.__init__(self)

        self.holding_time = 0

    def _find_decision(self):
        # time.sleep(0.1) # to simulate time for calculation
        if not self.holding:
            self.act = action.BUY
            self.holding = True
            self.holding_time = self.time_counter
            self.buy_in_price = self.market_history[self.time_counter - 1].price
            print("buy  at time " + str(self.time_counter) + "\t price : " + str(
                self.market_history[self.time_counter - 1].price))
        elif self.holding and self.buy_in_price < self.market_history[self.time_counter - 1].price:
            self.act = action.SELL
            print("sell at time " + str(self.time_counter) + "\t price : " + str(
                self.market_history[self.time_counter - 1].price))
            self.holding = False
        elif self.holding_time - self.time_counter > 20:
            self.act = action.SELL
            print("hold 2 long  " + str(self.time_counter) + "\t price : " + str(
                self.market_history[self.time_counter - 1].price))
            self.holding = False
        else:
            self.act = action.HOLD
        return self.act