
import numpy as np

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

    assert (DeltaT_out > 0), "Output temperature difference $\Delta T_1$ is negative"
    assert DeltaT_in > DeltaT_out, "Input temperature difference $\Delta T_1$ is smaller than output "
    denominator = np.log(DeltaT_in) - np.log(DeltaT_out)
    nominator = DeltaT_in - DeltaT_out
    assert denominator > eps, "Ratio of input/output temperature difference too large"
    log_mean_diff = nominator / denominator

    if flowpattern == 'cross' or flowpattern == 'multipass':
        assert corrfact >= 0.5, "Correction factor lower than 0.5"
        log_mean_diff *= corrfact

    return log_mean_diff

if __name__ == "__main__":
    pattern = 'parallel'
    log_mean_td = LMTD(80, 50, 20, 40, pattern)
    print("LMTD : %s %f" % (pattern, log_mean_td))

    pattern = 'counter'
    log_mean_td = LMTD(80, 50, 20, 40, pattern)
    print("LMTD : %s %f" % (pattern, log_mean_td))