# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:16:43 2020

@author: TrungNguyen
"""

# Replace week python date time function
import numpy as np
import pandas as pd
from pathlib import Path

from typing import List

def read_week(dir_name,xl_tab_name: str="DATA"):
    
    """ conversion from weeks spreadsheet tab into Dataframe.

    Args:
        xl_tab_name: (str) tabname from weeknummers2020 spreadsheet ("Pivot", "Data")

    Returns:
        pandas Dataframe with contents of weeknummers2020 tabsheet
        and holidays arrays

    """

    data_dir  = Path.cwd() /dir_name
    data_path = data_dir/'weeknummers2020.xlsx'
    #print(data_path)
    xls = pd.ExcelFile(data_path)
    print(xls.sheet_names)  # Check sheet names
    #data = pd.read_excel(xls,'DATA')
    
    # select sheet "DATA" 
    dfweek = pd.DataFrame()
    # df5060 = pd.read_excel(xls, 'nen5060 - energie')  # this file is part of NEN 5060 20018
    # NEN5060-2018.xlsx has two lines with column headers
    # first line is column name, second line is measurement unit
    dfweek = pd.read_excel(xls, xl_tab_name, header=[0])  # this file is part of NEN 5060 20018
    dfweek['Datum'] = pd.to_datetime(dfweek['Datum'] ,errors = 'coerce'
                                       ,format = '%Y-%m-%d').dt.strftime('%Y-%m-%d')
    #ind = dfweek.index
    #print(ind.values)
    #print(dfweek.head())
    #print(dfweek.columns)
    #print(data)
    '''
    Input Netherlands national holidays
    '''
    
    holidays = ['2020-01-01','2020-04-10','2020-04-12',
                '2020-04-13','2020-04-27','2020-05-05','2020-05-21','2020-06-01',
                '2020-12-25','2020-12-26','2020-12-31']
    
    #df_holidays = pd.DataFrame()
    #df_holidays['holidays'] = holidays#["holidays"].astype("|S")
    '''
    Get date, time column from data frame and convert to str
    '''
    dfweek['holidays'] = pd.Series(holidays)
	
    return dfweek #holidays


if __name__ == "__main__":
    df_week = read_week('NEN_data')
    print(df_week.Dag[0])
    #print(df_week[0].head())
    #print(df_week.head())
    #print(df_week[0].loc[1, 'Dag'])
    #print(df_week[1].loc[1])
    #print(df_week.iloc[:,]['holidays'])


    