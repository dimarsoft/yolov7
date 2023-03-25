import json
from pathlib import Path
from types import SimpleNamespace

from configs import TEST_TRACKS_PATH
from count_results import Result


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
            for i, div in enumerate(item.deviations):
                print(f"\t{i + 1}, status = {div.status_id}, frame: [{div.start_frame} - {div.end_frame}]")

    def add_test(self, test_info):
        if isinstance(test_info, Result):
            self.result_items.append(test_info)
        else:
            print(f"not a Result type: {test_info}")

    def save_results(self, output_folder):
        result_json_file = Path(output_folder) / "current_all_track_results.json"
        print(f"Save result_items '{str(result_json_file)}'")
        with open(result_json_file, "w") as write_file:
            write_file.write(json.dumps(self.result_items, indent=4, sort_keys=True, default=lambda o: o.__dict__))

    def compare_to_file(self, output_folder):
        self.compare_list_to_file(output_folder, self.test_items)

    def compare_one_to_file(self, output_folder):

        new_test = []
        for item in self.result_items:
            result_item = TestResults.get_for(self.test_items, item.file)
            new_test.append(result_item)

        self.compare_list_to_file(output_folder, new_test)

    def compare_list_to_file(self, output_folder, test_items):

        # 1 версия считаем вход/выход

        in_equals = 0  # количество не совпадений
        out_equals = 0

        sum_delta_in = 0
        sum_delta_out = 0

        by_item_info = []

        total = len(test_items)
        total_equal = 0

        for item in test_items:
            result_item = TestResults.get_for(self.result_items, item.file)

            if result_item is not None:
                actual_counter_in = result_item.counter_in
                actual_counter_out = result_item.counter_out
            else:
                actual_counter_in = 0
                actual_counter_out = 0

            delta_in = item.counter_in - actual_counter_in
            delta_out = item.counter_out - actual_counter_out

            if delta_in == 0 and delta_out == 0:
                total_equal += 1
            if delta_in == 0:
                in_equals += 1
            if delta_out == 0:
                out_equals += 1

            if delta_in != 0:
                item_info = dict()

                item_info["file"] = item.file
                item_info["expected_in"] = item.counter_in
                item_info["actual_in"] = actual_counter_in

                by_item_info.append(item_info)

            if delta_out != 0:
                item_info = dict()

                item_info["file"] = item.file
                item_info["expected_out"] = item.counter_out
                item_info["actual_out"] = actual_counter_out

                by_item_info.append(item_info)

            sum_delta_in += abs(delta_in)
            sum_delta_out += abs(delta_out)

        results_info = dict()

        results_info['equals_in'] = in_equals
        results_info['equals_out'] = out_equals

        results_info['delta_in_sum'] = sum_delta_in
        results_info['delta_out_sum'] = sum_delta_out

        results_info['not_equal_items'] = by_item_info

        results_info['total_records'] = total
        results_info['total_equal'] = total_equal
        results_info['total_equal_percent'] = (100.0 * total_equal) / total

        result_json_file = Path(output_folder) / "compare_track_results.json"

        print(f"Save compare results info '{str(result_json_file)}'")

        with open(result_json_file, "w") as write_file:
            write_file.write(json.dumps(results_info, indent=4, sort_keys=True, default=lambda o: o.__dict__))

        # 2 версия считаем дополнительно совпадения инцидентов

    def compare_to_file_v2(self, output_folder):
        self.compare_list_to_file_v2(output_folder, self.test_items)

    def compare_list_to_file_v2(self, output_folder, test_items):

        # 1 версия считаем вход/выход

        in_equals = 0  # количество не совпадений
        out_equals = 0

        sum_delta_in = 0
        sum_delta_out = 0

        by_item_info = []

        total = len(self.result_items)
        total_equal = 0

        for result_item in self.result_items:
            item = TestResults.get_for(test_items, result_item.file)

            actual_counter_in = result_item.counter_in
            actual_counter_out = result_item.counter_out

            if item is not None:
                expected_counter_in = item.counter_in
                expected_counter_out = item.counter_out
            else:
                expected_counter_in = 0
                expected_counter_out = 0

            delta_in = expected_counter_in - actual_counter_in
            delta_out = expected_counter_out - actual_counter_out

            if delta_in == 0 and delta_out == 0:
                total_equal += 1
            if delta_in == 0:
                in_equals += 1
            if delta_out == 0:
                out_equals += 1

            if delta_in != 0:
                item_info = dict()

                item_info["file"] = result_item.file
                item_info["expected_in"] = expected_counter_in
                item_info["actual_in"] = actual_counter_in

                by_item_info.append(item_info)

            if delta_out != 0:
                item_info = dict()

                item_info["file"] = result_item.file
                item_info["expected_out"] = expected_counter_out
                item_info["actual_out"] = actual_counter_out

                by_item_info.append(item_info)

            sum_delta_in += abs(delta_in)
            sum_delta_out += abs(delta_out)

        results_info = dict()

        results_info['equals_in'] = in_equals
        results_info['equals_out'] = out_equals

        results_info['delta_in_sum'] = sum_delta_in
        results_info['delta_out_sum'] = sum_delta_out

        results_info['not_equal_items'] = by_item_info

        results_info['total_records'] = total
        results_info['total_equal'] = total_equal
        if total > 0:
            results_info['total_equal_percent'] = (100.0 * total_equal) / total
        else:
            results_info['total_equal_percent'] = 0

        result_json_file = Path(output_folder) / "compare_track_results.json"

        print(f"Save compare results info '{str(result_json_file)}'")

        with open(result_json_file, "w") as write_file:
            write_file.write(json.dumps(results_info, indent=4, sort_keys=True, default=lambda o: o.__dict__))

        # 2 версия считаем дополнительно совпадения инцидентов


def test_tracks_file(test_file):
    """
Тест содержимого тестового файла.
1. читаем показываем. если что-то не так, то будет ошибка
2. создаем словарь по file, оно должно быть уникально, иначе ошибка по ключу
    """
    print(f"test_tracks_file: {test_file}")
    result = TestResults(test_file)
    result.print_info()

    test = dict()

    already_present_count = 0

    for i, item in enumerate(result.test_items):
        if item.file in test:
            print(f"{i}, {item.file} already present")
            print(f"{test[item.file]}, {item}")

            already_present_count += 1

        test[item.file] = item

    if already_present_count > 0:
        print(f"Error: File has duplicated file names, count error {already_present_count}!!!")
    else:
        print(f"Good: File has unique file name keys")

    for key, item in test.items():
        print(
            f"key = {key}, file = {item.file}, in = {item.counter_in}, out = {item.counter_out}, "
            f"deviations = {len(item.deviations)} ")
        for i, div in enumerate(item.deviations):
            print(f"\t{i + 1}, status = {div.status_id}, [{div.start_frame} - {div.end_frame}]")

    # result_json_file = "testinfo/tmp_track_results.json"
    # print(f"Save compare results info '{str(result_json_file)}'")

    # with open(result_json_file, "w") as write_file:
    #     write_file.write(json.dumps(test, indent=4, sort_keys=True, default=lambda o: o.__dict__))

    # result.result_items = TestResults.read_info(test_file)

    # result.compare_to_file("testinfo")


if __name__ == '__main__':
    test_tracks_file(test_file=TEST_TRACKS_PATH)
