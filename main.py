from trade_platform.src.agent.tcn_agent import tcn_agent
from trade_platform.src.agent.simple_agent import simple_agent
from trade_platform.src.agent.agent_thread import agent_thread
from trade_platform.src.trade_platform.trade_platform import trade_platform

if __name__ == "__main__":
    t = trade_platform(length=5000, data_path='data/US1.ABT_190504_200519.txt', enable_plot=False,random=False, type = "minute")
    # when inport csv or other data file, be sure to remove the headers.

    trained_agent = tcn_agent(trainset = 0, arima = False, model='2', moments=17)
    trained_agent.train('data/USDJPY.csv')
    t.add_agent(trained_agent)
    t.start()


# If you want to use your own mrkt_data format:
# check CS175-Trade-Platform/src/util/mrkt_data.py first
# for example, the order of your CSV is different from mrkt_data
# first define a function
def mrkt_data_example(self, args,time = 0 ):
    self.price = args[0]#hi
    self.spacial_data1 = args[1]
    self.spacial_data2 = args[2]
    # ......
    self.time  = time # time here is only as a reference. Use with discretion
# pass this function to cos_mrkt_data parameter of trade_platform
# example = trade_platform(length=5000, data_path='[PATH HERE]', enable_plot=False,random=False,cos_mrkt_data= mrkt_data_example)

'''/*
 * Synchronous(default) / Asynchronous(deprecated)
 * Market / Agent
 *
 * Trade platform
 *
*/'''
