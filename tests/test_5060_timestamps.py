
import context

from housemodel.weather_solar.weatherdata import (read_nen_weather_from_xl,
                                                  NENdatehour2datetime)
from housemodel.sourcesink.NEN5060 import run_qsun_new
import numpy as np


def test_timestamps():
    # read NEN5060 data from spreadsheet NEN5060-2018.xlsx "as is" into pandas DataFrame
    # use first header line as column names
    df_xl = read_nen_weather_from_xl()
    # generate and insert timezone-aware UTC and local timestamps (with DST)
    df_xl = NENdatehour2datetime(df_xl)


run_qsun_new()

if __name__ == "__main__":
    test_timestamps()
