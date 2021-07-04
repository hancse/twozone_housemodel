import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

import logging

logging.basicConfig()
logger = logging.getLogger('test_buffervessel')
logger.setLevel(logging.INFO)


def my_assert(condition, fail_str, suc_str):
    assert condition, fail_str
    print(suc_str)


import Simulation2R2C_buffervessel
import Simulation2R2C_buffervessel_matrix

logging.info("Testing Simulation2R2C_buffervessel")
t_sim, sp_sim, Tamb_sim, olddata = Simulation2R2C_buffervessel.main(show=False)
logging.info("Testing Simulation2R2C_buffervessel_matrix")
t_sim_new, sp_sim_new, Tamb_sim_new, newdata = Simulation2R2C_buffervessel_matrix.main(show=False)

time_hr = t_sim / 3600

np.testing.assert_allclose(t_sim, t_sim_new)
np.testing.assert_allclose(sp_sim, sp_sim_new)
np.testing.assert_allclose(Tamb_sim, Tamb_sim_new)
np.testing.assert_allclose(olddata[0], newdata[0])
np.testing.assert_allclose(olddata[1], newdata[1])
np.testing.assert_allclose(olddata[2], newdata[2], rtol=5e-7)
np.testing.assert_allclose(olddata[3], newdata[3])
np.testing.assert_allclose(olddata[4], newdata[4])
logging.info("all tests OK")

# plot the results
plt.figure(figsize=(15, 5))  # key-value pair: no spaces
plt.plot(olddata[4], olddata[0], label='Tair')
plt.plot(olddata[4], olddata[1], label='Twall')
plt.plot(olddata[4], olddata[2], label='Treturn')
plt.plot(olddata[4], olddata[3], label='Tbuffervessel')
plt.plot(t_sim, sp_sim, label='SP_Temperature')
plt.plot(t_sim, Tamb_sim, label='Toutdoor')
plt.legend(loc='best')
plt.show()

# plot the results
plt.figure(figsize=(15, 5))  # key-value pair: no spaces
plt.plot(newdata[4], newdata[0], label='Tair')
plt.plot(newdata[4], newdata[1], label='Twall')
plt.plot(newdata[4], newdata[2], label='Treturn')
plt.plot(newdata[4], newdata[3], label='Tbuffervessel')
plt.plot(t_sim_new, sp_sim_new, label='SP_Temperature')
plt.plot(t_sim_new, Tamb_sim_new, label='Toutdoor')
plt.legend(loc='best')
plt.show()
