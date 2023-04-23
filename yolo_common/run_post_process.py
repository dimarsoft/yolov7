from pathlib import Path

from post_processing.alex import alex_count_humans
from post_processing.group_1 import group_1_count_humans
from post_processing.group_3 import group_3_count
from post_processing.stanislav import stanislav_count_humans
from post_processing.timur import timur_count_humans
from tools.count_results import Result
from tools.exception_tools import print_exception


def get_post_process_results(test_func, track, num, w, h, fps, bound_line, source, log: bool) -> Result:
    # count humans

    humans_result = Result(0, 0, 0, [])

    # count humans
    if test_func is not None:
        try:
            tracks_new = []
            for item in track:
                tracks_new.append([item[0], item[5], item[6], item[1], item[2], item[3], item[4], item[7]])

            if isinstance(test_func, str):

                humans_result = None

                if test_func == "popov_alex":
                    humans_result = alex_count_humans(tracks_new, num, w, h, bound_line, log=log)
                    pass
                if test_func == "stanislav":
                    humans_result = stanislav_count_humans(tracks_new, num, w, h, bound_line, log=log)
                    pass
                if test_func == "group_1":
                    humans_result = group_1_count_humans(tracks_new, num, w, h, bound_line, log=log)
                    pass
                if test_func == "timur":
                    humans_result = timur_count_humans(tracks_new, source, bound_line, log=log)
                    pass
                if test_func == "group_3":
                    humans_result = group_3_count(tracks_new, num, w, h, fps)
                    pass
            else:
                #  info = [frame_id,
                #  left, top,
                #  width, height,
                #  int(detection[4]), int(detection[5]), float(detection[6])]
                # [frame_index, track_id, cls, bbox_left, bbox_top, bbox_w, bbox_h, box.conf]
                # humans_result = test_func(tracks_new)
                # bound_line =  [[490, 662], [907, 613]]
                # num(str), w(int), h(int)

                humans_result = test_func(tracks_new, num, w, h, bound_line)

        except Exception as e:
            print_exception(e, "post processing")

    humans_result.file = Path(source).name

    return humans_result
