class mrkt_data:
    # store all the data at a unit of time
    def __init__(self, args,time = 0 ):
        self.price = args[0]
        if len(args)> 1:
            # Back ward compatibility for plotting
            self.open = args[0]
            self.close= args[1]
            self.low  = args[2]
            self.high = args[3]
        else:
            self.open = None
            self.close = None
            self.low = None
            self.high = None
        self.time  = time # time here is only as a reference. Use with discretion