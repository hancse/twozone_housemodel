import numpy as np
import pandas as pd
from pathlib import Path

from housemodel.sourcesink.qsun import qsun


def run_qsun(df5060: pd.DataFrame, verbose=False):
    """

    Args:
        df5060:   dataframe from NEN5060 spreadsheet

    Returns:

    """
    # copying dataframe columns to numpy array
    # calculation using dataframe columns is much slower than using numpy arrays
    qglob_hor = df5060.loc[:, 'globale_zonnestraling'].values  # global irradiation
    qdiff_hor = df5060.loc[:, 'diffuse_zonnestraling'].values  # diffuse irradiation
    qdir_hor = df5060.loc[:, 'directe_zonnestraling'].values  # direct irradiation
    qdir_nor = df5060.loc[:, 'directe_normale_zonnestraling'].values  # DNI

    t_s = df5060.index.values * 3600
    iday = 1 + np.floor(t_s / (24 * 3600))
    LST = np.floor((t_s / 3600) % 24)
    if verbose:
        print(type(qdiff_hor))

    # dfout = pd.DataFrame(t, columns = list('t'))
    # dfout['iday'] = 1 + np.floor(t / (24 * 3600))
    # dfout['LST'] = np.floor((t / 3600) % 24)

    # Define an empty matrix for the result of qsun
    # this result is a numpy stack with
    # 8760 rows (hours per year)
    # 9 columns (compass directions -90(E), -45(SE), 0(S), 45(SW), 90(W), 135(NW), 180(N), 225 (NE) plus horizontal)
    # 4 decks (diffuse, direct, global and total solar irradiation)

    E = np.zeros((4, 8760, 9))

    # ground albedo is ignored, hence the  input parameter for qsun is zero
    ground_albedo = 0

    # (scalar) beta is the tilt (inclination) angle of the surface, horizontal is 0, vertical is +90
    # beta is 90 for k = -1 to 7 (vertical walls of house) and 0 for the horizontal flat roof surface
    # (scalar) gamma is the azimuth of the surface, starting with -90(E), loop over k = -1 to 7 to 225(NE)
    # for k = 8 (horizontal surface) gamma is arbitrary (set to 90 degrees here), since beta = 0

    k = -1  # k starts at -1 because of "East comes First" convention
    for j in range(9):
        print(j)
        if k < 7:
            gamma = 45 * (k - 1)  # gamma -90 (E), -45 (SE), 0 (S), 45 (SW), 90 (W), 135 (NW), 180 (N), 225 (NE)
            beta = 90
        else:
            gamma = 90
            beta = 0
        k = k + 1

        for row in range(8760):
            diffuse_irr, direct_irr, \
            total_irr, global_irr = qsun(qdiff_hor[row],
                                         qdir_nor[row],
                                         gamma, beta,
                                         ground_albedo,
                                         iday[row],
                                         LST[row])

            E[0, row, j] = diffuse_irr
            E[1, row, j] = global_irr
            E[2, row, j] = total_irr
            E[3, row, j] = global_irr

    # When you want to set an index or columns in data frame you should define it as a list
    dfout = pd.DataFrame(t_s, columns=list('t'))
    dfout['iday'] = 1 + np.floor(t_s / (24 * 3600))
    dfout['LST'] = np.floor((t_s / 3600) % 24)
    dfout['total_E'] = E[2, :, 0]
    dfout['total_SE'] = E[2, :, 1]
    dfout['total_S'] = E[2, :, 2]
    dfout['total_SW'] = E[2, :, 3]
    dfout['total_W'] = E[2, :, 4]
    dfout['total_NW'] = E[2, :, 5]
    dfout['total_N'] = E[2, :, 6]
    dfout['total_NE'] = E[2, :, 7]
    dfout['total_hor'] = E[2, :, 8]

    if verbose:
        print(dfout.columns)
    return dfout


def run_qsun_new(df5060: pd.DataFrame, azimuth, tilt, north_is_zero=False, verbose=False):
    """

    Args:
        df5060:          dataframe from NEN5060 spreadsheet
        azimuth:         azimuth of solar panel
        tilt:            tilt (inclination) of solar panel
        north_is_zero:   azimuth convention
                         if TRUE, north=0, east=90, south=180, west=270
                         if FALSE, north=180, east=-90, south=0, west=+90

    Returns:
        dataframe with total irradiation on surface in [W/m^2]
    """
    # copying dataframe columns to numpy array
    # calculation using dataframe columns is much slower than using numpy arrays
    qglob_hor = df5060.loc[:, 'globale_zonnestraling'].values  # global irradiation
    qdiff_hor = df5060.loc[:, 'diffuse_zonnestraling'].values  # diffuse irradiation
    qdir_hor = df5060.loc[:, 'directe_zonnestraling'].values  # direct irradiation
    qdir_nor = df5060.loc[:, 'directe_normale_zonnestraling'].values  # DNI

    t_s = df5060.index.values * 3600
    iday = 1 + np.floor(t_s / (24 * 3600))
    LST = np.floor((t_s / 3600) % 24)
    if verbose:
        print(type(qdiff_hor))

    # dfout = pd.DataFrame(t, columns = list('t'))
    # dfout['iday'] = 1 + np.floor(t / (24 * 3600))
    # dfout['LST'] = np.floor((t / 3600) % 24)

    # Define an empty matrix for the result of qsun
    # this result is a numpy stack with
    # 8760 rows (hours per year)
    # 4 columns (diffuse, direct, global and total solar irradiation)

    E = np.zeros((8760, 4))

    # ground albedo is ignored, hence the  input parameter for qsun is zero
    ground_albedo = 0

    if north_is_zero:
        azimuth -= 180.0

    for row in range(8760):
        diffuse_irr, direct_irr, \
        total_irr, global_irr = qsun(qdiff_hor[row],
                                     qdir_nor[row],
                                     azimuth, tilt,
                                     ground_albedo,
                                     iday[row],
                                     LST[row])

        E[row, 0] = diffuse_irr
        E[row, 1] = direct_irr
        E[row, 2] = total_irr
        E[row, 3] = global_irr

    # When you want to set an index or columns in data frame you should define it as a list
    dfout = pd.DataFrame(t_s, columns=list('t'))
    dfout['iday'] = 1 + np.floor(t_s / (24 * 3600))
    dfout['LST'] = np.floor((t_s / 3600) % 24)
    dfout['total_irr'] = E[:, 2]

    if verbose:
        print(dfout.columns)
    return dfout


if __name__ == "__main__":
    pass
