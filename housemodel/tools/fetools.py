
import numpy as np
"""
function M = MassAssembler1D(x)
n = length(x)-1; % number of subintervals
M = zeros(n+1,n+1); % allocate mass matrix
for i = 1:n % loop over subintervals
h = x(i+1) - x(i); % interval length
M(i,i) = M(i,i) + h/3; % add h/3 to M(i,i)
M(i,i+1) = M(i,i+1) + h/6;
M(i+1,i) = M(i+1,i) + h/6;
M(i+1,i+1) = M(i+1,i+1) + h/3;
end
"""

def assemble_mass_matrix():
    # for n in
    return M


def make_local_mass_matrix(heatcap):
    """

    Args:
        heatcap: heat capacity of element

    Returns:

    """
    m = np.identity(2)
    m *= 0.5*heatcap
    return m


def make_local_stiffness_matrix(conductivity: float):
    """

    Args:
        heatcap: heat capacity of element

    Returns:

    """
    m = np.full((2, 2), -1)
    np.fill_diagonal(m, 1)
    m *= conductivity
    return m


if __name__ == "__main__":
    M = make_local_mass_matrix(1e3)
    print(M)
    K = make_local_stiffness_matrix(10)
    print(K)
