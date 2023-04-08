import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pandas import DataFrame

from configs import TEST_TRACKS_PATH, TEST_ROOT
from count_results import Result, Deviation, get_status
from exception_tools import save_exception, print_exception


def save_test_result(test_results, session_folder, source_path):
    try:
        test_results.save_results(session_folder)
    except Exception as e:
        text_ex_path = Path(session_folder) / f"{source_path.stem}_ex_result.log"
        with open(text_ex_path, "w") as write_file:
            write_file.write("Exception in save_results!!!")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            for line in lines:
                write_file.write(line)
            for item in test_results.result_items:
                write_file.write(f"{str(item)}\n")

        print(f"Exception in save_results {str(e)}! details in {str(text_ex_path)} ")

    compare_result = None
    try:
        compare_result = test_results.compare_to_file_v2(session_folder)
    except Exception as e:
        text_ex_path = Path(session_folder) / f"{source_path.stem}_ex_compare.log"
        save_exception(e, text_ex_path, "compare_to_file_v2")

    return compare_result


def get_test_result(test_results, session_folder):
    compare_result = None
    try:
        compare_result = test_results.compare_to_file_v2(session_folder)
    except Exception as e:
        print_exception(e, "get_test_result")

    return compare_result


def results_to_json(result_items: Result):
    return json.dumps(result_items, indent=4, sort_keys=True, default=lambda o: o.__dict__)


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

        return results_info

    def compare_to_file_v2(self, output_folder):
        return self.compare_list_to_file_v2(output_folder, self.test_items)

    @staticmethod
    def intersect_deviation(dev_1: Deviation, dev_2: Deviation) -> bool:
        if (dev_1.start_frame >= dev_2.start_frame) and (dev_1.start_frame <= dev_2.end_frame):
            return True

        if (dev_1.end_frame >= dev_2.start_frame) and (dev_1.end_frame <= dev_2.end_frame):
            return True
        return False

    def compare_deviations(self, actual_deviations: list, expected_deviations: list) -> int:

        count_equal = 0

        for a_div in actual_deviations:

            for e_div in expected_deviations:
                if self.intersect_deviation(a_div, e_div):
                    count_equal += 1
                    break

        return count_equal

    def compare_list_to_file_v2(self, output_folder, test_items):

        # 1 версия считаем вход/выход

        in_equals = 0  # количество не совпадений
        out_equals = 0

        sum_delta_in = 0
        sum_delta_out = 0

        by_item_info = []
        by_item_dev_info = []

        total_records = len(self.result_items)
        total_equal = 0

        total_count_correct = 0
        total_actual_devs = 0
        total_expected_devs = 0

        for result_item in self.result_items:
            item = TestResults.get_for(test_items, result_item.file)

            actual_counter_in = result_item.counter_in
            actual_counter_out = result_item.counter_out

            actual_deviations = result_item.deviations

            if item is not None:
                expected_counter_in = item.counter_in
                expected_counter_out = item.counter_out

                expected_deviations = item.deviations
            else:
                expected_counter_in = 0
                expected_counter_out = 0
                expected_deviations = []

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

            count_correct = self.compare_deviations(actual_deviations, expected_deviations)

            actual_devs = len(actual_deviations)
            expected_devs = len(expected_deviations)

            # if count_correct != expected_devs or actual_devs != expected_devs:
            dev_info = dict()
            dev_info["file"] = result_item.file
            dev_info["count_correct"] = count_correct
            dev_info["actual_devs"] = actual_devs
            dev_info["expected_devs"] = expected_devs

            dev_info["expected_in"] = expected_counter_in
            dev_info["expected_out"] = expected_counter_out

            dev_info["actual_in"] = actual_counter_in
            dev_info["actual_out"] = actual_counter_out

            if actual_devs > 0:
                dev_info["dev_precision"] = count_correct / actual_devs
            else:
                dev_info["dev_precision"] = 0

            if expected_devs > 0:
                dev_info["dev_recall"] = count_correct / expected_devs
            else:
                dev_info["dev_recall"] = 0

            by_item_dev_info.append(dev_info)

            total_count_correct += count_correct
            total_actual_devs += actual_devs
            total_expected_devs += expected_devs

            sum_delta_in += abs(delta_in)
            sum_delta_out += abs(delta_out)

        results_info = dict()

        results_info['total_count_correct'] = total_count_correct
        results_info['total_actual_devs'] = total_actual_devs
        results_info['total_expected_devs'] = total_expected_devs

        results_info['equals_in'] = in_equals
        results_info['equals_out'] = out_equals

        results_info['delta_in_sum'] = sum_delta_in
        results_info['delta_out_sum'] = sum_delta_out

        results_info['not_equal_items'] = by_item_info
        results_info['dev_items'] = by_item_dev_info

        results_info['total_records'] = total_records
        results_info['total_equal'] = total_equal

        if total_records > 0:
            results_info['total_equal_percent'] = (100.0 * total_equal) / total_records
        else:
            results_info['total_equal_percent'] = 0

        if total_expected_devs > 0:
            results_info['total_dev_recall'] = (100.0 * total_count_correct) / total_expected_devs
            results_info['total_dev_actual_percent'] = (100.0 * total_actual_devs) / total_expected_devs
        else:
            results_info['total_dev_recall'] = 0
            results_info['total_dev_actual_percent'] = 0

        if total_actual_devs > 0:
            results_info['total_dev_precision'] = (100.0 * total_count_correct) / total_actual_devs
        else:
            results_info['total_dev_precision'] = 0

        if output_folder is not None:

            result_json_file = Path(output_folder) / "compare_track_results.json"

            print(f"Save compare results info '{str(result_json_file)}'")

            with open(result_json_file, "w") as write_file:
                write_file.write(json.dumps(results_info, indent=4, sort_keys=True, default=lambda o: o.__dict__))

            result_csv_file = Path(output_folder) / "compare_track_results.csv"
            result_xlsx_file = Path(output_folder) / "compare_track_results.xlsx"

            test_dev_results_to_table(by_item_dev_info, result_csv_file, result_xlsx_file)

        return results_info


def test_tracks_file(test_file):
    """
Тест содержимого тестового файла.

1. Читаем показываем. Если, что-то не так, то будет ошибка

2. Создаем словарь по file, оно должно быть уникально, иначе ошибка по ключу

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


def unique(list1):
    x = np.array(list1)
    return np.unique(x)


def get_files_str(items: list[dict]) -> str:
    files: list[int] = []
    for nc in items:
        f = int(Path(nc['file']).stem)
        files.append(f)

    files = sorted(unique(files))

    files_str = ""
    for item in files:
        f = str(item)
        files_str += f"{f},"

    return files_str


def get_no_correct_dev(all_dev: list[dict]) -> list[dict]:
    no_correct_dev = []

    for dev_info in all_dev:

        count_correct = dev_info["count_correct"]
        actual_devs = dev_info["actual_devs"]
        expected_devs = dev_info["expected_devs"]

        if count_correct != expected_devs or actual_devs != expected_devs:
            no_correct_dev.append(dev_info)

    return no_correct_dev


def save_results_to_csv(results: dict, file_path, sep=";") -> None:
    """
    Сохранение результатов сравнение в csv файл.
    Данные в таблично виде и упорядоченны по total_equal_percent

    Args:
        results (dict): Словарь с результатами сравнения
        file_path: Путь к файлу, для сохранения данных
        sep: Разделитель в csv файле
    """
    table = []
    for key in results.keys():
        results_info = results[key]

        total_equal_percent = results[key]["total_equal_percent"]
        total_equal = results[key]["total_equal"]
        total_records = results[key]["total_records"]

        not_equal_items = results[key]["not_equal_items"]

        total_count_correct = results_info['total_count_correct']
        total_actual_devs = results_info['total_actual_devs']
        total_expected_devs = results_info['total_expected_devs']

        # total_dev_correct_percent = results_info['total_dev_correct_percent']
        total_dev_actual_percent = results_info['total_dev_actual_percent']

        total_dev_precision = results_info['total_dev_precision']
        total_dev_recall = results_info['total_dev_recall']

        by_item_dev_info = results_info['dev_items']

        files_str = get_files_str(not_equal_items)
        files_dev_str = get_files_str(get_no_correct_dev(by_item_dev_info))

        print(f"{key} = {total_equal_percent}, {files_str}")

        table.append([key, total_equal_percent, total_equal, total_records, str(files_str),
                      total_dev_precision, total_dev_recall,
                      total_count_correct, total_actual_devs, total_expected_devs,
                      total_dev_actual_percent, files_dev_str])

    df = DataFrame(table, columns=["tracker_name", "total_equal_percent",
                                   "total_equal", "total_records", "not_equal_items",
                                   "total_dev_precision", "total_dev_recall",
                                   "total_count_correct", "total_actual_devs", "total_expected_devs",
                                   "total_dev_actual_percent",
                                   "no_correct_dev"])
    df.sort_values(by=['total_equal_percent'], inplace=True, ascending=False)

    # print(df)
    df.to_csv(file_path, sep=sep, index=False)


def results_to_table():
    file_path = "D:\\AI\\2023\\corridors\\dataset-v1.1\\2023_04_02_07_43_54_yolo_tracks_by_txt" \
                "\\all_compare_track_results.json"

    file_path_tbl = "D:\\AI\\2023\\corridors\\dataset-v1.1\\2023_04_02_07_43_54_yolo_tracks_by_txt" \
                    "\\all_compare_track_results.csv"

    with open(file_path, "r") as read_file:
        results = json.loads(read_file.read())

    save_results_to_csv(results, file_path_tbl)

    table = []
    for key in results.keys():
        total_equal_percent = results[key]["total_equal_percent"]
        total_equal = results[key]["total_equal"]
        total_records = results[key]["total_records"]

        print(f"{key} = {total_equal_percent}")

        table.append([key, total_equal_percent, total_equal, total_records])

    df = DataFrame(table, columns=["tracker_name", "total_equal_percent", "total_equal", "total_records"])
    df.sort_values(by=['total_equal_percent'], inplace=True, ascending=False)
    print(df)

    df = DataFrame.from_dict(results, orient="index")
    print(df)

    # print(results)


def test_dev_results_to_table(results: list, csv_file_path, excel_file_path, sep: str = ";"):
    table = []

    for dev_info in results:
        actual_devs = dev_info["actual_devs"]
        count_correct = dev_info["count_correct"]
        expected_devs = dev_info["expected_devs"]

        file = int(Path(dev_info["file"]).stem)

        precision = dev_info["dev_precision"]
        recall = dev_info["dev_recall"]

        expected_counter_in = dev_info["expected_in"]
        expected_counter_out = dev_info["expected_out"]

        actual_counter_in = dev_info["actual_in"]
        actual_counter_out = dev_info["actual_out"]

        table.append([file, actual_counter_in, actual_counter_out,
                      expected_counter_in,
                      expected_counter_out,
                      precision, recall, actual_devs, count_correct, expected_devs])

    df = DataFrame(table, columns=["file",
                                   "Вход", "Выход",
                                   "Вход(эталон)", "Выход(эталон)",
                                   "precision", "recall",
                                   "actual_devs",
                                   "count_correct",
                                   "expected_devs"])

    df.sort_values(by=['file'], inplace=True, ascending=True)
    # csv
    df.to_csv(csv_file_path, sep=sep, index=False)
    # excel
    df.to_excel(excel_file_path, index=False)


def test_results_to_table(results, csv_file_path, excel_file_path, sep: str = ";"):
    """
        Сохранение результатов сравнение в csv файл.
        Данные в таблично виде и упорядоченны по total_equal_percent

        Args:
            results (dict): Словарь с результатами сравнения
            csv_file_path: Путь к файлу, для сохранения данных в csv
            excel_file_path: Путь к файлу, для сохранения данных в excel
            sep: Разделитель в csv файле
        """
    table = []
    for key in results:
        results_info = key

        file_name = results_info["file"]
        file_id = int(Path(file_name).stem)

        counter_in = results_info["counter_in"]
        counter_out = results_info["counter_out"]

        deviations = results_info['deviations']

        deviations_count = len(deviations)

        if deviations_count > 0:
            table.append([file_id, file_name, counter_in, counter_out, deviations_count, "", "", "", ""])
            for i, dev in enumerate(deviations):
                start_frame = dev["start_frame"]
                end_frame = dev["end_frame"]
                status_id = dev["status_id"]

                table.append(
                    ["", "", "", "", "", start_frame, end_frame, status_id, get_status(status_id)])
        else:
            table.append([file_id, file_name, counter_in, counter_out, deviations_count, 0, 0, 0])

    df = DataFrame(table, columns=["file_id", "file",
                                   "counter_in", "counter_out",
                                   "deviations",
                                   "start_frame",
                                   "end_frame", "status_id", "Тип"])
    # csv
    df.to_csv(csv_file_path, sep=sep, index=False)
    # excel
    df.to_excel(excel_file_path, index=False)


def convert_test_json_to_csv():
    json_file_path = TEST_TRACKS_PATH
    csv_file_path = TEST_ROOT / "all_track_results.csv"
    excel_file_path = TEST_ROOT / "all_track_results.xlsx"

    with open(json_file_path, "r") as read_file:
        results = json.loads(read_file.read())

    test_results_to_table(results, csv_file_path, excel_file_path)


if __name__ == '__main__':
    test_tracks_file(test_file=TEST_TRACKS_PATH)
    # convert_test_json_to_csv()

    # results_to_table()
