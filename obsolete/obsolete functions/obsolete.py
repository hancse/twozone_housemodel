import numpy as np
import pandas as pd
from pathlib import Path


def nen5060_to_dataframe(xl_tab_name: str = "nen5060 - energie") -> pd.DataFrame:
    """ conversion from NEN5060 spreadsheet tab into Dataframe.
        SUPERSEDED by function read_nen_weather_from_xl
        in weather_solar subpackage
    Args:
        xl_tab_name: (str) tabname from NEN5060 spreadsheet ("nen5060 - energie", "ontwerp 1%" or "ontwerp 5%")

    Returns:
        pandas Dataframe with contents of NEN5060 tabsheet

    """
    # print(Path.cwd())
    data_dir = Path.cwd() / 'NEN_data'
    if not data_dir.exists():
        data_dir = Path.cwd().parent / 'NEN_data'
    output_dir = Path.cwd() / 'working' / 'submit'
    NENdata_path = data_dir / 'NEN5060-2018.xlsx'
    print(NENdata_path)
    xls = pd.ExcelFile(NENdata_path)
    print(xls.sheet_names)  # Check sheet names

    # select sheet "nen5060 - energie" by NEN default
    df5060 = pd.DataFrame()
    # df5060 = pd.read_excel(xls, 'nen5060 - energie')  # this file is part of NEN 5060 20018
    # NEN5060-2018.xlsx has two lines with column headers
    # first line is column name, second line is measurement unit
    df5060 = pd.read_excel(xls, xl_tab_name, header=[0, 1])  # this file is part of NEN 5060 2018
    ind = df5060.index
    print(ind.values)
    print(df5060.head())
    print(df5060.columns)

    return df5060  # pandas Dataframe
