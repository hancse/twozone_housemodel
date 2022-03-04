
import context

from housemodel.weather_solar.weatherdata import (read_nen_weather_from_xl,
                                                  NENdatehour2datetime,
                                                  get_hourly_knmi_weather_from_api)
import pandas as pd
from pytz import timezone
import numpy as np

def test_5060_timestamps():
    # read NEN5060 data from spreadsheet NEN5060-2018.xlsx "as is" into pandas DataFrame
    # use first header line as column names
    df_xl = read_nen_weather_from_xl()

    # generate and insert timezone-aware UTC and local timestamps (with DST)
    df_xl = NENdatehour2datetime(df_xl)

    # collect data from KNMI and compare with NEN5060
    df_api, exp = get_hourly_knmi_weather_from_api(260, 'ALL',
                                                   start=2001010101,
                                                   end=2001010124)

    np.testing.assert_allclose(df_xl.loc[0:23, 'temperatuur'], df_api.loc[0:23, 'T'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[0:23, 'luchtdruk'], df_api.loc[0:23, 'P'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[0:23, 'relatieve_vochtigheid'], df_api.loc[0:23, 'U'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[0:23, 'globale_zonnestraling'], df_api.loc[0:23, 'Q'], atol=0.5)

    df_api, exp = get_hourly_knmi_weather_from_api(260, 'ALL',
                                                   start=2004032801,
                                                   end=2004032824)

    np.testing.assert_allclose(df_xl.loc[2064:2064+23, 'temperatuur'], df_api.loc[0:23, 'T'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[2064:2064+23, 'luchtdruk'], df_api.loc[0:23, 'P'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[2064:2064+23, 'relatieve_vochtigheid'], df_api.loc[0:23, 'U'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[2064:2064+23, 'globale_zonnestraling'], df_api.loc[0:23, 'Q'], atol=0.5)

    df_api, exp = get_hourly_knmi_weather_from_api(260, 'ALL',
                                                   start=2010103101,
                                                   end=2010103124)

    # np.testing.assert_allclose(df_xl.loc[7272:7272+23, 'temperatuur'], df_api.loc[0:23, 'T'], rtol=1e-3)
    # np.testing.assert_allclose(df_xl.loc[7272:7272+23, 'luchtdruk'], df_api.loc[0:23, 'P'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[7272:7272+23, 'relatieve_vochtigheid'], df_api.loc[0:23, 'U'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[7272:7272+23, 'globale_zonnestraling'], df_api.loc[0:23, 'Q'], atol=0.5)

    df_api, exp = get_hourly_knmi_weather_from_api(260, 'ALL',
                                                   start=2003123101,
                                                   end=2003123124)

    np.testing.assert_allclose(df_xl.loc[8736:8736+23, 'temperatuur'], df_api.loc[0:23, 'T'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[8736:8736+23, 'luchtdruk'], df_api.loc[0:23, 'P'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[8736:8736+233, 'relatieve_vochtigheid'], df_api.loc[0:23, 'U'], rtol=1e-3)
    np.testing.assert_allclose(df_xl.loc[8736:8736+23, 'globale_zonnestraling'], df_api.loc[0:23, 'Q'], atol=0.5)

if __name__ == "__main__":
     test5060_timestamps()