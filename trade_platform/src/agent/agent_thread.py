import time
from threading import Thread
from ...src.util.util import *
import numpy as np

class agent_thread(Thread):
    def __init__(self, sync=True, currently_holding = False, buy_in_price = 0, time_ = 0, market_history = []):
        Thread.__init__(self)

        if time_ != len(market_history):
            time_ = len(market_history)
        print("Warning: unmatch time and market_history len. set time = market_history len")
        self.time_counter = time_

        self.synchronized = sync
        self.market_history = market_history

        self.act = action.HOLD
        self.offer_price = None

        self.holding = currently_holding
        self.buy_in_price = buy_in_price

        self.next_time_flag = False

        self.exit = False

        # below allow the agent to generate percentage data
        # this in for open, close, high, low version of mrkt_data
        self._percentage_counter = 0; # remember until what time the percentage is accurate
        self.percentage = []
        self._log_percentage_counter = 0;  # remember until what time the percentage is accurate
        self.log_percentage = []

    def _make_decision(self):
        # private function. To signal that a decision is made
        self.next_time_flag = False

    def _need_decision(self):
        return self.next_time_flag

    def _find_decision(self):
        # An agent should overwrite run or _find_decision
        # the main logic
        time.sleep(0.1)
        self.act = action.HOLD
        return self.act

    def get_action(self):
        # return action
        # used by tp
        # return action.HOLD
        return self.act,self.offer_price
    '''
    def set_time(self, value):
        # for debug only
        if (value != self.time_counter):
            if (value - self.time_counter != 1):
                print("wanrning : set_time : increment > 1")
            self.next_time_flag = True
            self.time_counter = value
    '''

    def _find_percentage(self):
        # called by get_percentage.
        while (self._percentage_counter < self.time_counter):
            # need calculatation
            if (self.market_history[self._percentage_counter].high \
                and self.market_history[self._percentage_counter].low \
                and self.market_history[self._percentage_counter].open \
                and self.market_history[self._percentage_counter].close) is None:
                raise Exception("Error: Agent_thread: this version mrkt_data cannot be used to calculate percentage"
                                + "must have high low open close")

            if self._percentage_counter == 0:
                # spacial case: The first line
                base = self.market_history[0].open
                d = {"close": self.market_history[0].close / base,
                     "open": self.market_history[0].open / base,
                     "high": self.market_history[0].high / base,
                     "low": self.market_history[0].low / base,
                     }
                self.percentage.append(d)
                self._percentage_counter += 1
            else:
                base = self.market_history[self._percentage_counter - 1].open
                d = {"close": self.market_history[self._percentage_counter].close / base,
                     "open": self.market_history[self._percentage_counter].open / base,
                     "high": self.market_history[self._percentage_counter].high / base,
                     "low": self.market_history[self._percentage_counter].low / base
                     }
                self.percentage.append(d)
                self._percentage_counter += 1
        if len(self.percentage) != len(self.market_history):
            raise Exception("Error: Agent_thread: Unmatched self.percentage and self.market_history")
        # already up to date

    def get_percentages(self, start=-1, end=None):
        # default return the last one
        # start = - N to return the lately N element
        # [start : end] for ranged elements
        self._find_percentage()
        return self.percentage[start:end]

    def get_log_percentages(self, start=-1, end=None):
        # default return the last one
        # start = - N to return the lately N element
        # [start : end] for ranged elements
        self._find_log_percentage()
        return self.log_percentage[start:end]

    def _find_log_percentage(self):
        # called by get_percentage.
        while (self._log_percentage_counter < self.time_counter):
            # need calculatation
            if (self.market_history[self._log_percentage_counter].high \
                and self.market_history[self._log_percentage_counter].low \
                and self.market_history[self._log_percentage_counter].open \
                and self.market_history[self._log_percentage_counter].close) is None:
                raise Exception("Error: Agent_thread: this version mrkt_data cannot be used to calculate percentage"
                                + "must have high low open close")

            if self._log_percentage_counter == 0:
                # spacial case: The first line
                base = self.market_history[0].open
                d = {"close": np.log(self.market_history[0].close / base),
                     "open": np.log(self.market_history[0].open / base),
                     "high": np.log(self.market_history[0].high / base),
                     "low": np.log(self.market_history[0].low / base),
                     }
                self.log_percentage.append(d)
                self._log_percentage_counter += 1
            else:
                base = self.market_history[self._percentage_counter - 1].open
                d = {"close": np.log(self.market_history[self._log_percentage_counter].close / base),
                     "open": np.log(self.market_history[self._log_percentage_counter].open / base),
                     "high": np.log(self.market_history[self._log_percentage_counter].high / base),
                     "low": np.log(self.market_history[self._log_percentage_counter].low / base)
                     }
                self.log_percentage.append(d)
                self._log_percentage_counter += 1
        if len(self.log_percentage) != len(self.market_history):
            raise Exception("Error: Agent_thread: Unmatched self.percentage and self.market_history")
        # already up to date

    def next_time(self):

        if self.synchronized:
            while (self.next_time_flag):
                pass
                # to prevent clock tick more than once
                # and to prevent time from changing while running
        self.time_counter += 1
        self.next_time_flag = True
        self.act = action.BLOCK

    ''' #USE WITH DISCRETION -_- DEBUG ONLY
    def set_market_history(self, time, data):
        #set market history at one point
        #USE WITH DISCRETION -_- DEBUG ONLY

        self.market_history[time] = data

    def set_all_market_history(self, data):
        # set entire market history
        #USE WITH DISCRETION -_- DEBUG ONLY
        self.market_history = data
        self.time_counter = len(data)
    '''

    def update_market_history(self, data):
        # undate for 1 unit of time
        self.next_time()
        self.market_history.append(data)

    def set_exit(self, value=True):
        # used to exit this agent thread
        self.exit = value

    def run(self):
        # An agent should overwrite run or _find_decision
        print("agent started")
        last_time = 0
        while True:
            if self.time_counter - last_time > 1:
                raise Exception("ERROR: Agent: not Sync ")
            if self.exit:
                print("client terminated")
                break
            if not self._need_decision():
                # the market has not updated. i.e. the time not changed
                continue
            # make a simply move

            # first we set action = block, so the market knows we need more time
            # self.act = action.BLOCK
            #This line was moved to next_time function

            # _find_decision
            self.act = self._find_decision()

            self._make_decision()
            last_time = self.time_counter
