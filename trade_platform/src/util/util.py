from enum import Enum

class action(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2
    BLOCK = 3



# unrelated
LENGTH = 1000  # Duration of test.
TIME_PER_TRADE = 1  # the intervel between each test
# If TIME_PER_TRADE = 1, then it can make 1 trade every unit of time.


# Macros for testing
RANDOM_MARKET_VALUE = True  # market will generate ranndam value
IMPORT_VALUE_FROM_FILE = False  # if not RANDOM_MARKET_VALUE, market object will import
IMPORT_PATH = ''

'''/*
 *
 *
 *
*/'''
