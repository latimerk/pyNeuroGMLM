import numpy as np
from pyGMLM.basisFunctions import ModifiedCardinalSpline

# get whole set of basis functions
def setupRCTBasis(bin_size_ms : float = 1.0, orthogonalize : bool = True, rescale : bool = True) -> dict:
    """
    Builds a set of basis functions for the RCT task using the modified Cardinal Spline basis.

    Args:
      bin_size_ms:     Time bin size in milliseconds
      orthogonalize:   Orthonormalize the basis functions (if false, returns the normal basis)
      rescale:         Rescales the basis functions by a constant for shrinkage priors

    Returns:
      A dictionary containing keys ["stimulus", "response", "spkHistory", "spkCoupling"].
      Each value is a ModifiedCardinalSpline object.
        
    """
    dd = 1e-12;
    cp = np.hstack((np.arange(0,200,20), np.arange(250,1600,250)));
    cp[0 ] -= dd;
    cp[-1] += dd;
    stimBasis = ModifiedCardinalSpline((0, 1600), c_pt=cp, \
        bin_size_ms=bin_size_ms, zero_first=True, zero_last=False, rescale_goal = 1 if rescale else None, orthogonalize=orthogonalize);
        
    cp = np.hstack((np.arange(-300,100,50), np.arange(-60,40,20), 50));
    cp[0 ] -= dd;
    cp[-1] += dd;
    responseBasis = ModifiedCardinalSpline((-300, 50), c_pt = cp, \
        bin_size_ms=bin_size_ms, zero_first=False, zero_last=False, rescale_goal = 1 if rescale else None, orthogonalize=orthogonalize);
        
    if(bin_size_ms < 2):
        cp = np.array([0, 2, 4, 10, 20, 40, 60, 80, 120, 160, 240, 320, 400, 480, 560, 640]);
    elif(bin_size_ms < 4):
        cp = np.array([0, 4, 10, 20, 40, 60, 80, 120, 160, 240, 320, 400, 480, 560, 640]);
    elif(bin_size_ms < 10):
        cp = np.array([0, 10, 20, 40, 60, 80, 120, 160, 240, 320, 400, 480, 560, 640]);
    else:
        raise ValueError(f"Cannot build spike history for bin size of {bin_size_ms} ms");
    cp[-1] += dd;
    spkHistBasis = ModifiedCardinalSpline(640, c_pt = cp, \
        bin_size_ms=bin_size_ms, zero_first=False, zero_last=False, rescale_goal = 10 if rescale else None, orthogonalize=orthogonalize);

    spkCouplingBasis = spkHistBasis;

    bases = dict{"stimulus" : stimBasis, "response" : responseBasis, "spkHistory" : spkHistBasis, "spkCoupling" : spkCouplingBasis};
    return bases;

# gets model setup for the stimulus

# gets model setup for the response

# build design matrices for spike history (& spike coupling)

# build design matrices for stimulus

# build design matrices for response


