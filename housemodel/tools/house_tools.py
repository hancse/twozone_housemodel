
import numpy as np
import csv


def LMTD(T_hot_in, T_hot_out, T_cold_in, T_cold_out, flowpattern='parallel', corrfact=1.0):
    """calculates log mean temperature difference

    representative value in case of varying temperature difference along heat exchanger
    https://checalc.com/solved/LMTD_Chart.html
    Args:
        T_hot_in:     entry temperature hot fluid or gas
        T_hot_out:    exit temperature hot fluid or gas
        T_cold_in:    entry temperature cold fluid or gas
        T_cold_out:   exit temperature cold fluid or gas
        flowpattern: 'parallel', 'counter', 'multipass' or 'cross' flow
        corrfact:    see:     https://checalc.com/solved/LMTD_Chart.html
                              https://cheguide.com/lmtd_charts.html
                              https://excelcalculations.blogspot.com/2011/06/lmtd-correction-factor.html
                              http://fchart.com/ees/heat_transfer_library/heat_exchangers/hs2000.htm
                              https://yjresources.files.wordpress.com/2009/05/4-3-lmtd-with-tutorial.pdf
                              https://www.engineeringtoolbox.com/arithmetic-logarithmic-mean-temperature-d_436.html
    Returns:
        LMTD temperature
        ( Delta T 1 - Delta T 2 ) / ln (Delta t 1 / Delta T 2)
    """
    eps = 1e-6
    if flowpattern == 'parallel':
        DeltaT_in = T_hot_in - T_cold_in
        DeltaT_out = T_hot_out - T_cold_out

    else:
        DeltaT_in = T_hot_in - T_cold_out
        DeltaT_out = T_hot_out - T_cold_in

    # assert (DeltaT_out > 0), "Output temperature difference $\Delta T_1$ is negative"
    # assert DeltaT_in > DeltaT_out, "Input temperature difference $\Delta T_1$ is smaller than output "
    denominator = np.log(DeltaT_in) - np.log(DeltaT_out)
    nominator = DeltaT_in - DeltaT_out
    # assert denominator > eps, "Ratio of input/output temperature difference too large"
    log_mean_diff = nominator / denominator

    if flowpattern == 'cross' or flowpattern == 'multipass':
        assert corrfact >= 0.5, "Correction factor lower than 0.5"
        log_mean_diff *= corrfact

    return log_mean_diff


def LMTD_radiator(T_feed, T_return, T_amb, corrfact=1.0):
    """calculates log mean temperature difference

    representative value in case of varying temperature difference along heat exchanger
    https://checalc.com/solved/LMTD_Chart.html
    Args:
        T_feed:     entry temperature hot fluid or gas
        T_return:    exit temperature hot fluid or gas
        T_amb:    entry temperature cold fluid or gas
        corrfact:    see:     https://checalc.com/solved/LMTD_Chart.html
                              https://cheguide.com/lmtd_charts.html
                              https://excelcalculations.blogspot.com/2011/06/lmtd-correction-factor.html
                              http://fchart.com/ees/heat_transfer_library/heat_exchangers/hs2000.htm
                              https://yjresources.files.wordpress.com/2009/05/4-3-lmtd-with-tutorial.pdf
                              https://www.engineeringtoolbox.com/arithmetic-logarithmic-mean-temperature-d_436.html
    Returns:
        LMTD temperature
        ( Delta T 1 - Delta T 2 ) / ln (Delta t 1 / Delta T 2)
    """
    eps = 1e-9
    DeltaT_fr = T_feed - T_return
    DeltaT_feed = T_feed - T_amb
    DeltaT_ret = T_return- T_amb

    # assert (DeltaT_fr > 0), "Output temperature difference $\Delta T_1$ is negative"
    # assert DeltaT_in > DeltaT_out, "Input temperature difference $\Delta T_1$ is smaller than output "
    denominator = np.log(DeltaT_feed) - np.log(DeltaT_ret)
    nominator = DeltaT_fr
    # assert denominator > eps, "Ratio of input/output temperature difference too large"
    log_mean_diff = nominator / denominator
    log_mean_diff *= corrfact

    return log_mean_diff

def house_to_csv():
    with open('new_yaml_xl/simulation_output.csv', mode='w') as output_file:
        output_writer = csv.writer(output_file, delimiter=';',
                                   quotechar='"', quoting=csv.QUOTE_MINIMAL)

        output_writer.writerow(['Description',
                                'Resultaten HAN Dynamic Model Heat Built Environment'])
        output_writer.writerow(['Chain number', '1'])
        output_writer.writerow(['Designation', '2R-2C-1-zone' '2R-2C-1-zone'])
        output_writer.writerow(['Node number', '0', '1', '2'])
        output_writer.writerow(['Designation', 'Outside Temperature',
                               'Internals', 'Load bearing construction'])
        output_writer.writerow(['TIMESTEP', 'TEMPERATURE (GRADEN C)', 'TEMPERATURE (GRADEN C)',
        'SOLAR (J)', 'SOURCES (J)', 'HEATING (J)', 'TEMPERATURE', 'SOLAR', 'SOURCES', 'HEATING'])


if __name__ == "__main__":
    deg = u"\u00b0"
    pattern = 'parallel'
    log_mean_td = LMTD(80, 50, 20, 40, pattern)
    print("LMTD : %s %f" % (pattern, log_mean_td))

    pattern = 'counter'
    log_mean_td = LMTD(80, 50, 20, 40, pattern)
    print("LMTD : %s %f" % (pattern, log_mean_td))

    pattern = 'counter'
    log_mean_td = LMTD(80, 50, 20, 20, pattern)
    print("LMTD : %s %f" % (pattern, log_mean_td))

    log_mean_rad = LMTD_radiator(80, 50, 20)
    print(f"LMTD_radiator : {log_mean_rad} {deg}C")

    # house_to_csv()


