__all__ = ["LoadDataframe"]

import numpy as np
import pandas as pd
import os
import glob
import re

"""
load_dataframe
auth: Methodfunc - Kwak Piljong
date: 2021.08.03
modify date: 2021.08.29
version: 0.3
describe: 데이터 폴더 안 csv, excel 외 다른 파일이 들어가 있어도 불러올 수 있게 변경
"""


class LoadDataframe:
    """
    Description: 판다스 데이터프레임을 불러옵니다. 단일 파일, 폴더 단위로 불러오며, 폴더 단위로 불러올 시
    폴더 안에 있는 모든 데이터를 합친 후 데이터를 내보냅니다.

    path      = file or folder path type: str
    index_col = default None type: str
    header    = default 0 type: str

    사용법
    1. 클래스 선언하기.
    2. get_df()로 데이터프레임 불러오기

    for example:
    path = '../data'
    load_df = LoadDataframe(path)
    raw_df = load_df.get_df()


    """

    def __init__(self, path, index_col=None, header=0):
        self._path = path
        self.extensions = None
        self._index_col = index_col
        self._header = header
        self.df = pd.DataFrame()

        self.ext = ["csv", "xls", "xlsx"]

        self.__load_df()

    def __repr__(self):
        if self.df.empty:
            return "Empty DataFrame"
        else:
            return f"DataFrame size: {self.df.shape}"

    def __call__(self):
        self.df.reset_index(inplace=True, drop=True)
        return self.df

    def get_df(self):
        self.df.reset_index(inplace=True, drop=True)
        return self.df

    def __load_df(self):  # 함수 숨기기
        try:
            if self.__check_path() == 0:
                self.__load_folder_df()

            if self.__check_path() == 1:
                self.__load_file_df()
        except UnboundLocalError:
            print("파일 경로 및 폴더 경로가 잘못 되었습니다.")
            exit()

    def __check_path(self):
        keys = None
        if os.path.isdir(self._path):
            keys = 0

        if os.path.isfile(self._path):
            keys = 1

        return keys

    def __load_file_df(self):
        temp = self.__read_df(self._path)
        self.df = self.df.append(temp)

    def __load_folder_df(self):
        for i, data in enumerate(os.listdir(self._path)):
            if os.path.isdir(f"{self._path}/{data}"):
                continue
            else:
                self.extensions = os.path.splitext(os.listdir(self._path)[i])[1]

                p_value = [i in str.lower(self.extensions) for i in self.ext]
                if not np.max(p_value):
                    continue

                break

        file_list = glob.iglob(f"{self._path}/*{self.extensions}")

        for file_path in file_list:
            temp = self.__read_df(file_path)
            self.df = self.df.append(temp)

    def __read_df(self, file_path):
        extension_type = None
        regex = r"[^A-Za-z]"
        if self.__check_path() == 0:
            extension_type = re.sub(regex, "", self.extensions)
        elif self.__check_path() == 1:
            extension_type = os.path.splitext(self._path)[1]
            extension_type = re.sub(regex, "", extension_type)

        check = str.lower(extension_type)
        if check == "csv":
            temp = pd.read_csv(
                file_path, index_col=self._index_col, header=self._header
            )
        elif (check == "xls") or (check == "xlsx"):
            temp = pd.read_excel(
                file_path, index_col=self._index_col, header=self._header
            )
        else:
            raise "Not support extension type"

        return temp
