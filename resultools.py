import json
from pathlib import Path
from types import SimpleNamespace


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


class TestResults:
    def __init__(self, test_file):
        self.test_file = test_file

        self.test_items = TestResults.read_info(test_file)
        self.result_items = []

    @staticmethod
    def read_info(json_file):
        with open(json_file, "r") as read_file:
            return json.loads(read_file.read(), object_hook=lambda d: SimpleNamespace(**d))

    @staticmethod
    def get_for(x, file):
        for item in x:
            if item.file == file:
                return item
        return None

    @staticmethod
    def compare_item_count(test, my):
        return (test.counter_in == my.counter_in) and (test.counter_out == my.counter_out)

    def print_info(self):
        for item in self.test_items:
            print(
                f"file = {item.file}, in = {item.counter_in}, out = {item.counter_out}, "
                f"deviations = {len(item.deviations)} ")
            for div in item.deviations:
                print(f"\t div = {div.status_id}, [{div.start_frame} - {div.end_frame}]")

    def add_test(self, test_info):
        self.result_items.append(test_info)

    def save_results(self, output_folder):
        result_json_file = Path(output_folder) / "current_all_track_results.json"
        with open(result_json_file, "w") as write_file:
            write_file.write(json.dumps(self.result_items, indent=4, sort_keys=True, default=lambda o: o.__dict__))
