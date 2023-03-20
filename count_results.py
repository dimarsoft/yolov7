class Deviation(object):
    def __init__(self, start, end, status):
        self.start_frame = start
        self.end_frame = end
        self.status_id = status


class Result:
    def __init__(self, humans, c_in, c_out, deviations):
        self.file = ""
        self.humans = humans
        self.counter_in = c_in
        self.counter_out = c_out
        self.deviations = deviations
