from threading import Thread,Lock
import time
import numpy as np
from ...src.util.mrkt_data import mrkt_data
from trade_platform.src.util.Data_parsing.data_parsing import parse

class market():
    '''/*
     * This is the base object for market
     * it will only consider price
    */'''
    def __init__(self ,data_path = '', random=False,length=0 , type = None):
        #type is for future use, if there is a specific way of parsing needed

        #Make sure that is some sort of data
        if not random and data_path == '':
            print("warning: no data was given to generate a market")

        self.type = type

        self.time_counter = 0
        self.market_values = list()
        if random:
            self.length = length
        else:
            self.length = 0
        self.ended = False # status of simulation
        #randomly generate price
        if random:
            r = np.random.random(self.length) * 5 + 20
            for i,d in enumerate(r):
                self.market_values.append(mrkt_data([d],time=i))
        else:
            self.data_path = data_path
            self.read_data()
            print("Info: Market: imported "+str(self.length) + "lines of data")

    def read_data(self):
        if self.type == "minute":
            values = parse(self.data_path).values
            for i, val in enumerate(values):
                self.market_values.append(mrkt_data(val, time = i));
            self.length = i
        else:
            data = np.genfromtxt(self.data_path, dtype = float, delimiter = ',')
            data = np.reshape(data, (1, np.product(data.shape)))[0]
            for i, val in enumerate(data):
                self.market_values.append(mrkt_data([val], time = i))
            self.len = data.shape[0]

    '''
    Get current value gives the current value of the market

    If it has gotten minute data, then the current value will just be open
    If it has gotten tick data, the currnt value will be the value at said tick

    Get_current_open/high/low/close only works for minute data
    '''
    def get_current_value(self):
        try:
            return self.market_values[self.time_counter][0]
        except (TypeError):
            return self.market_values[self.time_counter]

    def get_current_open(self):
        return self.market_values[self.time_counter][0]

    def get_current_high(self):
        return self.market_values[self.time_counter][1]

    def get_current_low(self):
        return self.market_values[self.time_counter][2]

    def get_current_close(self):
        return self.market_values[self.time_counter][3]

    def get_all_value(self):
        # historical price, including current
        return list(self.market_values)


    def get_all_value(self):
        # historical price, including current
        return list(self.market_values)

    def get_ranged_value(self, range):
        # return market value from current time_counter to time_counter-range
        # if range = 0, return entire list
        if range < 0:
            raise Exception("get_ranged_value : negative")
        if range == 0:
            return self.get_all_value()
        if(range >= self.time_counter + 1):
            return np.append(np.array([mrkt_data([0], None)]*(range-self.time_counter-1)), np.array(self.market_values[0:self.time_counter+1]))
        return np.array(self.market_values[self.time_counter - range : self.time_counter])

    def set_time(self,value):
        # to skip forward, meant for debug, no guarantee
        # be careful for the mutex lock
        self.time_counter = value

    def get_time(self):
        return  self.time_counter

    def next_time(self):
        # step into next unit of time
        if not self.ended:
            self.time_counter += 1
        else:
            print("warning: market: simulation ended time = ", self.time_counter)
        if self.time_counter == self.length - 1:
            print("warning: market: simulation ended time = ", self.time_counter)
            self.ended = True



class market_thread(market, Thread):
    def __init__(self, sync = True,data_path = '', animation_speed = 1.0, random=False,length=0, graph=False, graph_span=50, cos_mrkt_data = None, type = None):
        '''/*
         * sync: True: wait for all agent finish to step into next unit of time
         * number_of_agent : in case you want more than one agent

         * animation_speed: if not sync, the market will move forward animation_speed unit per second
         * eg: if animation_speed = 0.5 , then market will move forward 1 unit of time every 2 seconds.

         * graph: enable real time market grpah
         * graph_span: How many past units of time are included in the graph. 0 : Entire history

         * cos_mrkt_data: import customized mrkt_data
        */'''
        if(cos_mrkt_data is not None):
            mrkt_data.__init__ = cos_mrkt_data
        print("market type is:", type)
        market.__init__(self, random=random, length=length,data_path = data_path, type = type)
        Thread.__init__(self)
        self.sync = sync
        #self.next_time_status = False
        self.next_time_mutex = Lock()

        self.exit = False

        self.animation_speed = animation_speed


    def set_next_time_status(self, value=True):

        self.next_time_mutex.acquire()
        while (self.next_time_mutex.locked()):
            pass
        #self.next_time_status = value

    def set_exit(self, value=True):
        # used to end the thread
        self.exit = value

    def run(self) -> None:
        while True:
            if self.ended: # if the simulation runs out
                break
            if self.exit: # be terminated
                break
            if(self.sync):
                if(self.next_time_mutex.locked()):
                    self.next_time()
                    self.next_time_mutex.release()

            else:
                time.sleep(1/self.animation_speed)
                self.next_time()