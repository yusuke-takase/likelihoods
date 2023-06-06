import numpy as np

def _cl2dl_(cl, ell_start=2.):
    """ The function to convert C_ell to D_ell
    
    Parameters
    ----------
    cl: 1d-array
        Power spectrum
    ell_start:float (default = 2.)
        The multi-pole ell value of first index of the `cl`.
        
    Return
    ------
    dl: 1d-array
    """
    ell = np.arange(ell_start, len(cl)+ell_start)
    return cl*ell*(ell+1.)/(2.*np.pi)

def _dl2cl_(dl, ell_start=2.):
    """ The function to convert D_ell to C_ell
    
    Parameters
    ----------
    dl: 1d-array
        (Reduced) Power spectrum
    ell_start:float (default = 2.)
        The multi-pole ell value of first index of the `dl`.
        
    Return
    ------
    cl: 1d-array
    """
    ell = np.arange(ell_start, len(dl)+ell_start)
    return dl*(2.*np.pi)/(ell*(ell+1.))

def forecast(lmax, cl_sys, path="/home/cmb/yusuket/program/simdata/output_camb_for_PTEP", 
                rmin=1e-8, rmax=1e-1, rresol=1e5, iter=0, verbose=False, test=False, bias=1e-5):
    """ The function to estimate the bias of tensor-to-scalar ratio.
    This function based on the paper: https://academic.oup.com/ptep/article/2023/4/042F01/6835420
    P88, Sec. (5.3.2)
    
    Usage and detail of the function
    --------------------------------
    The user is required to download the model of power spectrum from wiki page: 
    https://wiki.kek.jp/display/cmb/Cosmological+Parameters%2C+sky+maps+and+power+spectra+to+use
    and put the directory `output_camb_for_PTEP` in your machine. The path to reach the directory is requireed as the argument `path`.
    
    The argument `rmin` and `rmax` represent a range for a first r survery. `rresol` is the resulution of the grid of r within the range.
    If the argument `iter` does not equal 0, the estimation is continued around the r which is estimated on a previous survey. 
    You can see the survey log with `verbose` makes True.
    If the argument `test=True` we can verify the correctness of this function. The estimation result should be same with the value we set as `bias`.

    
    Parameters
    ----------
    lmax   : int
    cl_sys : 1d-array
    path   : str
    rmin   : float
    rmax   : float
    rresol : float
    iter   : int
    verbose: bool
    test   : bool
    bias   : float
    
    Return
    ------
    data: Dict
    """
    rresol = int(rresol)
    gridOfr = np.linspace(rmin, rmax, num=rresol)
    # Load the dat file which includes model power spectrum
    dl_tens = np.loadtxt(path + "/PTEP_tens_dls.dat")
    dl_lens = np.loadtxt(path + "/PTEP_lensed_dls.dat")
    # Convert D_ell to C_ell
    cl_tens = _dl2cl_(dl_tens.T[3], ell_start=2) # dl_tens.T[3] : tensor B-mode (r=1)
    cl_lens = _dl2cl_(dl_lens.T[3], ell_start=2) # dl_lens.T[3] : lensiong B-mode
    
    # Note that the dat file has a power spectrum value from ell = 2 to ~4000.
    # In order to keep the formalism, we insert zeros at ell=1,2 of power spectrum.
    cl_tens = np.insert(cl_tens, 0, [0.,0.])
    cl_lens = np.insert(cl_lens, 0, [0.,0.])
    
    if test == True:
        # cl_sys is replaced to cl_tens which is maltiplyed `bias`
        cl_sys[0:lmax+1] = cl_tens[0:lmax+1] * bias
        
    ell = np.arange(2, lmax+1)
    Nell = len(ell)
    delta_r = 0.
    likelihood = 0.
    gridOfr4LH= 0.

    for j in range(iter + 1):
        Nr = len(gridOfr)
        likelihood = np.zeros(Nr)
        
        for i, grid_val in enumerate(gridOfr):
            Cl_hat = cl_sys[ell] + cl_lens[ell]
            Cl = grid_val * cl_tens[ell] + cl_lens[ell]
            likelihood[i] = np.sum((-0.5) * (2.*ell + 1.) * ((Cl_hat / Cl) + np.log(Cl) - ((2.*ell - 1.) / (2.*ell + 1.)) * np.log(Cl_hat)))
        
        likelihood = np.exp(likelihood - np.max(likelihood))
        maxid = np.argmax(likelihood)
        delta_r = gridOfr[maxid]
        survey_range = [delta_r - delta_r*(0.5/(j+1.)), delta_r + delta_r*(0.5/(j+1.))]
        gridOfr_old = gridOfr
        gridOfr = np.linspace(survey_range[0], survey_range[1], num=int(1e4))
        
        if verbose == True:
            print("*--------------------------- iter =", j, "---------------------------*")
            print("Δr                :", delta_r)
            print("Next survey range :", survey_range)
    
    # Calcurate the likelihood function again in the range that is delta_r*1e-3 < delta_r < delta_r*3.
    # Note that the delta_r has already been estimated, this likelihood is used for display. 
    gridOfr4LH = np.linspace(delta_r*1e-3, delta_r*3., num=int(1e4))
    Nr = len(gridOfr4LH)
    likelihood = np.zeros(Nr)
    
    for i, grid_val in enumerate(gridOfr4LH):
        Cl_hat = cl_sys[ell] + cl_lens[ell]
        Cl = grid_val * cl_tens[ell] + cl_lens[ell]
        likelihood[i] = np.sum((-0.5) * (2.*ell + 1.) * ((Cl_hat / Cl) + np.log(Cl) - ((2.*ell - 1.) / (2.*ell + 1.)) * np.log(Cl_hat)))
    
    likelihood = np.exp(likelihood - np.max(likelihood))
    data = {"delta_r":delta_r, "grid_r":gridOfr4LH, "likelihood":likelihood}
    return data
