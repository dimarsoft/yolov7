class Deviation(object):
    def __init__(self, start, end, status):
        self.start_frame = int(start)
        self.end_frame = int(end)
        self.status_id = int(status)

    def __str__(self):
        return f"{self.status_id}: [{self.start_frame}, {self.end_frame}]"


class Result:
    def __init__(self, humans, c_in, c_out, deviations: list):
        self.file = ""
        self.humans = int(humans)
        self.counter_in = int(c_in)
        self.counter_out = int(c_out)
        self.deviations = deviations

    def __str__(self):
        return f"file = {self.file}, in = {self.counter_in}, " \
               f"out = {self.counter_out}, deviations = {len(self.deviations)}"
