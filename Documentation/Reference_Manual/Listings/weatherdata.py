#!/bin/env python3

import pandas as pd
import numpy as np
import datetime as dt
from pytz import timezone

import requests
from io import StringIO  # , BytesIO
from typing import Tuple

# Determine the directory where the weather data files are located:
from pathlib import Path

WEATHERFILEDIR = Path(__file__).absolute().parent
# The directory where the weather data files are located is the same
# as where this Python file sits
NEN5060DIR = Path(__file__).absolute().parent.parent.parent / 'NEN_data'


# The directory where the NEN 5060 data files are located is two levels higher
# than where this Python file sits


def get_hourly_knmi_weather_from_api(stations: str = '260', vars_to_get: str = 'ALL',
                                     start: str = '2020010101', end: str = '2020020124') -> Tuple:
    """get weather data from KNMI hourly data website.

    Args:
        stations (str):      (list of) weather stations separated by colon (260 = De Bilt).
        vars_to_get (str):   (list of) weather variables i.e. TEMP (°C), SUNR (solar radiation) or ALL.
        start (str):         start date and time in compact format (2020010101)
        end (str):           end date and time in compact format (2020082024)
    Note: for each date, the weather data is given for the hourly interval given (01-24)

    Returns:
        (pandas Dataframe):  data columns:  # STN  YYYYMMDD     HH      T   T10N     TD.
                                            # STN  YYYYMMDD     HH      SQ  Q
        list (str):          explanatory header lines from api call

    Refs:
        - https://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script
        - https://daggegevens.knmi.nl/klimatologie/uurgegevens
        - https://stackoverflow.com/questions/2018026/what-are-the-differences-between-the-urllib-urllib2-urllib3-and-requests-modul
        - https://www.geeksforgeeks.org/get-post-requests-using-python/
    """

    # url knmi hourly data
    URL = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"

    # defining a params dict for the parameters to be sent to the API
    # stns=235:280:260&vars=VICL:PRCP&start=197001001&end=2009081824
    PARAMS = {'stns': stations, 'vars': vars_to_get, 'start': start, 'end': end}

    # sending get request and saving the response as response object
    response = requests.get(url=URL, params=PARAMS)

    # split response.text in explanatory header lines and spreadsheet
    # https://stackoverflow.com/questions/4664850/how-to-find-all-occurrences-of-a-substring
    # https://stackoverflow.com/questions/3873361/finding-multiple-occurrences-of-a-string-within-a-string-in-python
    lastcomment = response.text.rfind('#')  # find # before "STN" in header line for Pandas DataFrame
    explanation = response.text[:lastcomment - 1].splitlines()
    spreadsheet = response.text[lastcomment + 1:]

    # extracting data
    df = pd.read_csv(StringIO(spreadsheet), header=0, sep=r'\s*,\s*',
                     engine='python')  # sep: avoid spaces in the column labels, e.g. '    T' -> 'T'

    # conversion to float
    if vars_to_get == 'TEMP' or vars_to_get == 'ALL':
        df.loc[:, 'T'] /= 10  # Convert the temperature from 0.1°C -> °C
        df.loc[:, 'T10N'] /= 10  # Convert the temperature from 0.1°C -> °C
        df.loc[:, 'TD'] /= 10  # Convert the temperature from 0.1°C -> °C
    if vars_to_get == 'SUNR' or vars_to_get == 'ALL':
        df.loc[:, 'Q'] *= 10000 / 3600  # Convert the global radiation from J/cm^2 -> W/m^2:
        df.loc[:, 'SQ'] /= 10  # convert sunshine from tenths to a fraction between 0 and 1
    if vars_to_get == 'ALL':
        df.loc[:, 'P'] /= 10
    return df, explanation


def read_hourly_knmi_weather_from_csv(filename, start_date, end_date):
    """Reads the measured KNMI weather data for a particular station from file.
    
    Reads the CSV files provided on the KNMI website for the weather data over the last decade from a chosen
    station. This function reads the weather data from a chosen start date to the chosen end date, which is
    NOT included.  For example, for the whole of the year 2020, specify 2020,1,1, 2021,1,1.
    
    Parameters:
        filename (str): filename to select the specific station, this is "uurgeg_260_2011-2020_Bilt.csv" for De Bilt
    
        start_date (datetime):  start date:  date of the first data entry to be returned (timezone aware).
        end_date   (datetime):  end date:    date of the first data entry NOT to be returned (timezone aware).
    
    Returns:
        pandas.DataFrame:  Table containing 26 columns with weather data, most relevant YYYYMMDD, HH, T, SQ, Q and datetime:
    
        - YYYYMMDD (int):      Date (YYYY=year,MM=month,DD=day).
        - HH (int):            Time in hours, the hourly division 05 runs from 04.00 UT to 5.00 UT.
        - T (float):           Temperature at 1.50 m at the time of observation (°C).
        - SQ (float):          Fraction of the hour with sunshine (0-1).
        - Q (float):           Global horizontal radiation (W/m2).
        - datetime (datetime)  Date and time of the entry (UT).
          
    """

    # Read the data as a Pandas DataFrame with entries from the start to the end date:
    try:
        knmi_data = pd.read_csv(WEATHERFILEDIR.joinpath(filename), header=28, sep=r'\s*,\s*', engine='python')
    except Exception as e:
        print(e)
        exit(1)

    # Convert the YYYYMMDD and HH columns into a datetime-object column with hours in [0,23] rather than
    # in [1,24]:
    knmi_data = KNMIdatehour2datetime(knmi_data)

    # Select the data for the desired date range.  Add one hour, since datetime contains the time at which the hour ENDS!
    knmi_data = knmi_data[knmi_data['datetime'] >= start_date + dt.timedelta(hours=1)]  # Date must be >= start_date
    knmi_data = knmi_data[knmi_data['datetime'] < end_date + dt.timedelta(hours=1)]  # Date must be < end_date
    knmi_data = knmi_data.reset_index(
        drop=True)  # Reset the index so that it starts at 0 again.  Don't keep the original index as a column.

    # Slicing below produces a warning telling us to use .loc[], which we are using already(!)
    # - https://stackoverflow.com/a/20627316/1386750
    pd.options.mode.chained_assignment = None  # Default='warn'
    knmi_data.loc[:, 'T'] /= 10  # Convert the temperature at 1.50 m from 0.1°C -> °C.
    knmi_data.loc[:, 'SQ'][knmi_data.loc[:,
                           'SQ'] == -1] = 0.25  # Sunshine time == -1 indicates 0 - 0.05 hours, so assume 0.025 hours, expressed in [0.1 hours]
    knmi_data.loc[:, 'SQ'] /= 10  # Convert the sunshine time from [0.1 hours] to a fraction (0-1)
    knmi_data.loc[:, 'Q'] /= 0.36  # Convert the global horizontal radiation from [J/cm2/h] to [W/m2]
    pd.options.mode.chained_assignment = 'warn'  # Back to default

    return knmi_data


def read_nen_weather_from_xl(xl_tab_name: str = "nen5060 - energie") -> pd.DataFrame:
    """ conversion from NEN5060 spreadsheet tab into Dataframe.

    Args:
        xl_tab_name: (str) tabname from NEN5060 spreadsheet NEN5060-2018.xlsx
        ("nen5060 - energie", "ontwerp 1%" or "ontwerp 5%")
        select sheet "nen5060 - energie" by NEN default

    Returns:
        pandas Dataframe with contents of NEN5060 tabsheet

    # NEN5060-2018.xlsx has two lines with column headers,first line is column name, second line is measurement unit
    """
    print(f"\nNEN_5060 folder: {NEN5060DIR}")
    xls = pd.ExcelFile(NEN5060DIR.joinpath('NEN5060-2018.xlsx'))
    print(xls.sheet_names)  # Check sheet names

    # df5060 = pd.read_excel(xls, sheet_name=xl_tab_name, header=[0, 1])
    df5060 = pd.read_excel(xls, sheet_name=xl_tab_name, header=0, skiprows=[1], usecols="A:S")

    df5060['temperatuur'] /= 10.0
    df5060['absolute_vochtigheid'] /= 10.0
    df5060['neerslaghoeveelheid'] /= 10.0
    df5060['windsnelheid'] /= 10.0
    df5060['bewolkingsgraad'] /= 8.0
    df5060['zonneschijnduur'] /= 10.0
    df5060['luchtdruk'] /= 10.0

    print(df5060.head())
    print(df5060.columns)

    return df5060  # pandas Dataframe


def read_nen_weather_from_csv():
    """Read the weather data from the NEN 5060 standard.
    
    Reads the CSV file provided from the NEN 5060 standard year. This function reads the weather data from
    a chosen start date to the chosen end date and adds the year, month and day columns together
    into the YYYYMMDD format.
    
    Returns:
        Array containing 26 columns with weather data, most relevant YYYYMMDD, HH, T:
    
        - YYYYMMDD  (int):    Date (YYYY=year,MM=month,DD=day).
        - H         (int):    Time in hours, the hourly division 05 runs from 04.00 UT to 5.00 UT.
        - T         (float):  Temperature at 1.50 m at the time of observation (°C).
    """

    try:
        nen_weather_data = pd.read_csv(WEATHERFILEDIR.joinpath('NEN5060-A2a.csv'), header=5, sep=r'\s*,\s*',
                                       engine='python')
    except Exception as e:
        print(e)
        exit(1)

    # Read the data and convert into an array from the start till end date
    nen_weather_data["YYYYMD"] = (nen_weather_data['Y'].astype(str) + nen_weather_data['M'].astype(str)
                                  + nen_weather_data['D'].astype(str))
    nen_weather = nen_weather_data.set_index("YYYYMD", drop=False)

    nen_weather['T'] /= 10.0

    return nen_weather


def calculate_degree_days(mode, temperature_base, weather_range):
    """Calculates the degree days in terms of heating or cooling.

    Calculates the degree hours in terms of the chosen mode of either heating or cooling compared to a base temperature
    and then converts these to degree days. This is done for either the NEN 5060 standard year with nen_weather_range
    as argument or for the KNMI weather data with knmi_weather_range.

    Parameters:
        mode              (str):    Mode for which degree days are calculated, either "heating" or "cooling".
        temperature_base  (float):  Base temperature to which the cooling or heating is compared (°C).
        weather_range     (str):    Selection of either NEN data with nen_weather_range or KNMI with knmi_weather_range.
    
    Returns:
        float:  Number of degree days for the selected period.
    """

    # Put the temperature in an array:
    temperature = weather_range['T'].values  # Air temperature at 1.50 m in °C.

    # Calculate the heating degree days if a temperature is above the base temperature:
    assert mode == "heating" or "cooling", "heating or cooling mode"

    if mode == "heating":
        degree_days = sum(temperature_base - temperature[temperature < temperature_base]) / 24  # Number of degree days

    # Calculate the cooling degree hour if a temperature is below the input base temperature
    elif mode == "cooling":

        degree_days = sum(temperature[temperature > temperature_base] - temperature_base) / 24  # Number of degree days

    return degree_days


def KNMIdatehour2datetime(knmi_data):
    """Convert the KNMI date and hour columns to a single datetime column.
    
    The KNMI date is expressed as an integer formatted as YYYYMMDD, while the hours run from 1-24 rather than
    from 0-23.  This causes problems when converting to Python or Pandas datetime objects.
    
    Parameters:
      knmi_data (Pandas df):  KNMI weather dataframe.
    
    Returns:
      (Pandas df):  KNMI weather dataframe.
    
    """

    from astrotool.date_time import fix_date_time

    # Split the YYYYMMDD column into separate numpy arrays:
    ymd = knmi_data['YYYYMMDD'].values  # Numpy array
    years = np.floor(ymd / 1e4).astype(int)
    months = np.floor((ymd - years * 1e4) / 100).astype(int)
    days = np.floor(ymd - years * 1e4 - months * 100).astype(int)

    # Create numpy arrays for the time variables:
    hours = knmi_data['HH'].values  # Numpy array
    minutes = np.zeros(hours.size)
    seconds = np.zeros(
        hours.size) + 0.001  # 1 ms past the hour, to ensure no negative round-off values occur (e.g. 2021,1,1, 0,0,-1e-5 -> 2020,12,31, 23,59,59.99999)

    # Fix the dates, e.g. turning 2020-12-31 24:00:00 to 2021-01-01 00:00:00:
    years, months, days, hours, minutes, seconds = fix_date_time(years, months, days, hours, minutes, seconds)

    # Combine the 1D numpy arrays into a single 2D array with the original arrays as COLUMNS, and convert it to a Pandas df:
    dts = pd.DataFrame(np.vstack([years, months, days, hours]).transpose(), columns=['year', 'month', 'day', 'hour'])
    dts = pd.to_datetime(dts, utc=True)  # Turn the columns in the df into a single datetime64[ns] column

    # Add the datetime column to the KNMI weather dataframe:
    knmi_data['datetime'] = dts

    return knmi_data


def NENdatehour2datetime(nen_df: pd.DataFrame):
    """Convert the NEN date and hour columns to a single datetime column.

    The NEN datetime information is expressed in the first four columns of the NEN5060-2018 spreadsheet,
    with names: 'jaar', 'MONTH(datum)', 'DAY(datum)', 'HOUR(uur)'.
    The hour numbering follows the KNMI convention, running from 1-24 rather than from 0-23.
    KNMI uses NAIVE UTC timestamps. See: https://www.knmidata.nl/data-services/knmi-producten-overzicht
    Remedy:
    - construct a NAIVE UTC Pandas DateTimeIndex from the first four columns in the spreadsheet
       Thereby:
       - subtracting one hour to convert from KNMI convention (backward average)
         to forward average needed for modelling with ODE solver
       Thus:
       2001 1 1 1 -> 2001-01-01 01:00:00 (NAIVE UTC, backward)
       2001-01-01 01:00:00 -> 2001-01-01 00:00:00 (NAIVE UTC, forward)
       In the KNMI tables, using the UTC timestamp at the END of the interval,
       this occurs at YYYYMMDD=20001231 and H=24, averageing the weather from
       2000-12-31 23:00:00 to 2001-01-01 00:00:00
       a "forward looking" UTC timestamp thus becomes 2000-12-31 23:00:00
    - add timezone-awareness to the forward looking NAIVE UTC timestamp
       2000-12-31 23:00:00 -> 2000-12-31 23:00:00 +00:00 (AWARE UTC)
    - convert to local time in timezone "Europe/Amsterdam"
       2000-12-31 23:00:00 +00:00 -> 2001-01-01 00:00:00 +01:00 (AWARE CET)
       The last step also covers conversion to DST (CEST, +02:00) in the summer period for the AWARE local time.

    Args:
        nen_df (Pandas df):  NEN5060 weather dataframe.

    Returns:
        (Pandas df):  NEN5060 weather dataframe.
    """
    # define timezones
    utz = timezone('UTC')
    nltz = timezone('Europe/Amsterdam')

    # convert columns 'jaar', 'MONTH(datum)', 'DAY(datum)', 'HOUR(uur)' into Pandas timestamps
    # subtracting 1 hour from the 'HOUR(uur)' values (works automatically!)
    pdt_naive = pd.to_datetime(dict(year=nen_df['jaar'],
                                    month=nen_df['MONTH(datum)'],
                                    day=nen_df['DAY(datum)'],
                                    hour=nen_df['HOUR(uur)'] - 1))
    # make NAIVE UTC forward-looking timestamp AWARE
    # Note: this cannot be done inplace because Timestamps are IMMUTABLE
    # Note2: since pdt_naive is a pandas Series object, use Series.dt.tz_localize and Series.dt.tz_convert
    pdt_utc = pdt_naive.dt.tz_localize(tz=utz)
    # convert AWARE UTC to AWARE local time
    pdt_local = pdt_utc.dt.tz_convert(tz=nltz)

    # insert AWARE UTC and AWARE LOCAL DateTimeIndex as first columns in DataFrame
    nen_df.insert(loc=0, column='utc', value=pdt_utc)
    nen_df.insert(loc=1, column='local_time', value=pdt_local)
    return nen_df


if __name__ == "__main__":
    nltz = timezone('Europe/Amsterdam')
    start_time = nltz.localize(dt.datetime(2019, 1, 1))
    end_time = nltz.localize(dt.datetime(2019, 1, 2))
    weather_knmi = read_hourly_knmi_weather_from_csv("uurgeg_260_2011-2020_Bilt.csv",
                                                     start_time, end_time)
    print(weather_knmi)
    print(calculate_degree_days("heating", 18, weather_knmi))

    weather_nen = read_nen_weather_from_csv()
    print(weather_nen)

    # # test get_hourly_knmi_weather_from_api
    station_DeBilt = '260'
    vars_from_station = 'ALL'
    dfout, exp = get_hourly_knmi_weather_from_api(station_DeBilt, vars_from_station, '2002123101', '2003010224')
    print(dfout)
    [print(line) for line in exp]

    df5060 = read_nen_weather_from_xl()
    print(df5060)
