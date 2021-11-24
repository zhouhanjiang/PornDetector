# -*- coding: utf-8 -*-

# @Time    : 2021/09/02 10:44
# @Author  : zhouhanjiang@xunlei.com
# @FileName: detector_test.py
# @Software: PyCharm
"""
  detector_test.py
"""

import os
# # import inspect
# import copy
# # import pytest
# import time
# import subprocess

from xldllib_common.until.logger import get_logger
# from xldllib_common.until.logger import set_global_logger_level
# from xldllib_common.until.logger import set_es_logger_report
import xldllib_common.until.global_var as global_var

# logger = get_logger(os.path.basename(__file__),__file__,level="INFO",log_file_max_bytes=1024*1024,log_to_file = True)
logger = get_logger(os.path.basename(os.path.abspath(__file__)), os.path.abspath(__file__), level="INFO",
                    log_file_max_bytes=1024 * 1024, log_to_file=True)

# import xldllib_common.until.basic_tool as basic_tool
# import xldllib_common.until.basic_tool_base as basic_tool_base
# import xldllib_common.networkobj.dllib_http_client as dllib_http_client
# import xldllib_common.until.system_tool_base as system_tool_base
# import xldllib_common.networkobj.network_tool as network_tool
import xldllib_common.fileobj.file_tool as file_tool
# import xldllib_common.networkobj.ftp_tool as ftp_tool

global_var.set_global_var_dict("need_report_es_result", True)


class PornDetectorHandler:
    """
      PornDetectorHandler
    """

    def __init__(self, current_dir=""):
        logger.debug("current_dir=" + str(current_dir))
        if str(current_dir).strip() == "":
            current_dir = file_tool.get_current_dir()
            self.parent_dir = os.path.dirname(current_dir)
        else:
            if file_tool.is_a_dir(current_dir):
                self.parent_dir = current_dir
            else:
                current_dir = file_tool.get_current_dir()
                self.parent_dir = os.path.dirname(current_dir)
        self.current_dir = current_dir
        global_var.set_global_var_dict("current_dir", self.current_dir)
        global_var.set_global_var_dict("parent_dir", self.parent_dir)

        self.model_bin_file_path = os.path.join(self.current_dir, "model.bin")
        self.model_bin_file_path = str(self.model_bin_file_path).replace("\\", "/")
        if not file_tool.is_a_file(self.model_bin_file_path):
            raise Exception("error.PornDetectorHandler.__init__.model_bin_file_path not exists."
                            "self.model_bin_file_path=" + str(self.model_bin_file_path))

        self.nnmodel_bin_file_path = os.path.join(self.current_dir, "nnmodel.bin")
        self.nnmodel_bin_file_path = str(self.nnmodel_bin_file_path).replace("\\", "/")
        if not file_tool.is_a_file(self.nnmodel_bin_file_path):
            raise Exception("error.PornDetectorHandler.__init__.nnmodel_bin_file_path not exists."
                            "self.nnmodel_bin_file_path=" + str(self.nnmodel_bin_file_path))

        self.test_resource_dir = os.path.join(self.current_dir, "test_resource")
        file_tool.ensure_folder_exists(self.test_resource_dir)

        self.tranning_img_dir = os.path.join(self.test_resource_dir, "tranning_img")
        file_tool.ensure_folder_exists(self.tranning_img_dir)

        self.tranning_normal_img_dir = os.path.join(self.tranning_img_dir, "normal")
        file_tool.ensure_folder_exists(self.tranning_normal_img_dir)

        self.tranning_porn_img_dir = os.path.join(self.tranning_img_dir, "porn")
        file_tool.ensure_folder_exists(self.tranning_porn_img_dir)

    def pcr_porn_predict(self, file_path=""):
        """
          pcr_porn_predict
        """
        porn_predictions = -999
        try:
            logger.debug("current_dir=" + str(self.current_dir))

            file_path = str(file_path).strip()
            if not file_tool.is_a_file(file_path):
                return porn_predictions

            file_path_list = [file_path]

            from pcr import PCR
            model = PCR()
            model.loadModel('model.bin')
            porn_predictions = model.predict(file_path_list)

            return porn_predictions
        except Exception as msg:
            logger.warning("err,msg=" + str(msg))
            return porn_predictions

    def nudenet_detector(self, file_path=""):
        """
          https://github.com/notAI-tech/NudeNet
          pcr_porn_predict
        """
        porn_processed_boxes_dict = {}
        try:
            logger.debug("current_dir=" + str(self.current_dir))
            file_path = str(file_path).strip()
            if not file_tool.is_a_file(file_path):
                return porn_processed_boxes_dict

            # Import module
            from nudenet import NudeDetector

            # Microsoft Visual C++ Redistributable for Visual Studio 2019 not installed on the machine.
            # 
            # initialize detector (downloads the checkpoint file automatically the first time)
            detector = NudeDetector()  # detector = NudeDetector('base') for the "base" version of detector.

            # Detect single image
            porn_processed_boxes_dict = detector.detect(file_path)

            return porn_processed_boxes_dict
        except Exception as msg:
            logger.warning("err,msg=" + str(msg))
            return porn_processed_boxes_dict


def test_porn_detector():
    p_d_cls = PornDetectorHandler()
    logger.debug("test_porn_detector.p_d_cls=" + str(p_d_cls))

    file_fullpath_list, dir_filelist_dict, sub_folder_list, file_fullpath_index_dict = file_tool.traverse_folder_files(p_d_cls.tranning_normal_img_dir)
    logger.debug("test_porn_detector.len(file_fullpath_index_dict)=" + str(len(file_fullpath_index_dict)))
    logger.debug("test_porn_detector.len(dir_filelist_dict)=" + str(len(dir_filelist_dict)))
    logger.debug("test_porn_detector.len(file_fullpath_list)=" + str(len(file_fullpath_list)))
    logger.debug("test_porn_detector.len(sub_folder_list)=" + str(len(sub_folder_list)))

    need_detect_file_fullpath_list = file_fullpath_list

    file_fullpath_list, dir_filelist_dict, sub_folder_list, file_fullpath_index_dict = file_tool.traverse_folder_files(p_d_cls.tranning_porn_img_dir)
    logger.debug("test_porn_detector.len(file_fullpath_index_dict)=" + str(len(file_fullpath_index_dict)))
    logger.debug("test_porn_detector.len(dir_filelist_dict)=" + str(len(dir_filelist_dict)))
    logger.debug("test_porn_detector.len(file_fullpath_list)=" + str(len(file_fullpath_list)))
    logger.debug("test_porn_detector.len(sub_folder_list)=" + str(len(sub_folder_list)))

    need_detect_file_fullpath_list += file_fullpath_list

    for file_fullpath in need_detect_file_fullpath_list:
        prediction = p_d_cls.nudenet_detector(file_fullpath)
        logger.info("test_porn_detector.pcr_porn_predict.prediction=" + str(prediction) + ";file_fullpath=" + str(file_fullpath))


if __name__ == '__main__':
    test_porn_detector()
