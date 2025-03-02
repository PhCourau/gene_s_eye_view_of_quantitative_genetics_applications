import numpy as np
import scipy.special as special
import warnings

def gss(f, a0, b0, tolerance=1e-5):
  """
  Golden-section search
  to find the minimum of f on [a,b]
  * f: a strictly unimodal function on [a,b]

  Example:
  >>> def f(x): return (x - 2) ** 2
  >>> x = gss(f, 1, 5)
  >>> print(f"{x:.5f}")
  2.00000

  Credit to Wikipedia
  """
  a=a0
  b=b0
  invphi = (np.sqrt(5) - 1) / 2  # 1 / phi (inverse golden ratio)
  while b - a > tolerance:
    c = b - (b - a) * invphi
    d = a + (b - a) * invphi
    if f(c) < f(d):
      b = d
    else:  # f(c) > f(d) to find the maximum
      a = c
  if ((a-a0 < 2*tolerance) or (b0-b< 2*tolerance)):
    warnings.warn("The golden-section search found an answer close to the boundaries, maybe widen up the domain of the search")
  return (b + a) / 2

def mean_X(sN,thetaN):
  """Finds the mean X associated with a modified beta solution
  to a WF with selection sN, mutation thetaN"""
  return (thetaN[0]/(thetaN[0]+thetaN[1])
          *special.hyp1f1(2*thetaN[0]+1,2*(thetaN[0]+thetaN[1])+1,2*sN)
          /special.hyp1f1(2*thetaN[0],2*(thetaN[0]+thetaN[1]),2*sN))

def mean_sigma2(sN,thetaN):
  """Finds the mean X(1-X) associated with a modified beta solution
  to a WF with selection sN, mutation thetaN"""
  return (thetaN[0]*thetaN[1]/((thetaN[0]+thetaN[1])*(thetaN[0]+thetaN[1]+1/2))
          *special.hyp1f1(2*thetaN[0]+1,2*(thetaN[0]+thetaN[1])+2,2*sN)
          /special.hyp1f1(2*thetaN[0],2*(thetaN[0]+thetaN[1]),2*sN))

def mean_phenotype(sN,thetaN,L,list_alpha,list_proba_alpha):
  """Finds the mean phenotype.
  Arguments:
  ----------
  thetaN: mutation rates (tuple of two positive floats)

  sN: effective selection coefficient s^*

  list_alpha: list of classes of values for gene effect. Typically np.linspace(a,b)

  list_proba_alpha: the list of probabilities of each class in list_alpha
  """
  return L*np.sum(2*list_alpha*mean_X(sN*list_alpha,thetaN)*list_proba_alpha)

def genetic_variance(sN,thetaN,L,list_alpha,list_proba_alpha):
  """Finds the mean E[2alpha**2 X(1-X)].
  Arguments:
  ----------
  thetaN: mutation rates (tuple of two positive floats)

  sN: effective selection coefficient s^*

  list_alpha: list of classes of values for gene effect. Typically np.linspace(a,b)

  list_proba_alpha: the list of probabilities of each class in list_alpha
  """
  return L*np.sum(2*list_alpha**2*mean_sigma2(sN*list_alpha,thetaN)*list_proba_alpha)

############################ WEAK SELECTION ##################################
def match_equation_weak(sN,thetaN,eta,N,L,om2,list_alpha,list_proba_alpha):
  """Finds how well the equation for weak selection is satisfied
  L: the number of loci
  om2: the inverse strength of stabilizing selection
  """
  return np.abs( (mean_phenotype(sN,thetaN,L,list_alpha,list_proba_alpha)-eta)
                 /om2*N + sN)

def find_sN_weak(om2,
                 thetaN,
                 eta,
                 N,
                 L,
                 list_alpha,
                 list_proba_alpha,
                 tolerance=1e-5,
                 a0=-1000,
                 b0=1000):
  """Finds the value of s^* given by the equation for weak selection. The search
     is made on the interval (a0,b0)"""
  return gss(lambda sN: match_equation_weak(sN,
                                           thetaN,
                                           eta,
                                           N,L,
                                           om2,
                                           list_alpha,
                                           list_proba_alpha),
             a0,b0, #Boundaries of the golden-section search
             tolerance=tolerance
  )


############################ MODERATE SELECTION ##################################
def match_equation_moderate(sN,thetaN,eta,L,list_alpha,list_proba_alpha):
  """Finds the absolute value of the distance between the mean phenotype and the
  optimum eta"""
  return np.abs(mean_phenotype(sN,thetaN,L,list_alpha,list_proba_alpha)-eta)


def find_sN_moderate(thetaN,
                     eta,
                     L,
                     list_alpha,
                     list_proba_alpha,
                     tolerance=1e-5,
                     a0=-1000,
                     b0=1000):
  """Finds the value of s^* given by the equation for moderate selection, searches
     on the interval (a0,b0)"""
  return gss(lambda s: match_equation_moderate(s,
                                               thetaN,
                                               eta,
                                               L,
                                               list_alpha,
                                               list_proba_alpha),
             a0,b0, #Boundaries of the golden-section search
             tolerance=tolerance
  )


############################ STRONG SELECTION ##################################
# First we need to compute the integral of x^{a-1} (1-x)^{b - 1} e^{cx + dx(1-x)}
def integral_WF_FD(a,b,c,d,klim):
  """
  Computes
  sum_{k} c^k / (k!)   a(a+1)...(a+k-1)/((a+b)(a+b+1)...(a+b+k-1)) 1F1(a+k,b;a+b+k;d)
  One can check that this is proportionnal to the integral of
  x^{a-1} (1-x)^{b-1} e^{cx + dx(1-x)}
  with proportionnality coefficient independent of c and d.
  klim is the precision of the approximation
  """
  return np.sum([d**j/special.factorial(j) * special.poch(a,j) * special.poch(b,j)
                 /special.poch(a+b,2*j) * special.hyp1f1(a+j,a+b+2*j,c)
                for j in range(klim)],axis=0)

def mean_X_FD(sN,dN,thetaN,klim=20):
  """
  Computes the mean of a WF diffusion with frequency-dependent (FD) selection
  xi(x) = s + d(1-2x)
  and mutation rates given by theta.

  Parameters
  ----------
  thetaN: tuple of two positive floats: mutation rates
  sN: float: directional selection
  dN: float: dominance coefficient
  klim: int: the precision of the approximation
  """
  return (thetaN[0]/(thetaN[0]+thetaN[1])
         *integral_WF_FD(2*thetaN[0]+1,2*thetaN[1],2*sN,2*dN,klim)
         /integral_WF_FD(2*thetaN[0],2*thetaN[1],2*sN,2*dN,klim))

def var_WF_FD(sN,dN,thetaN,klim=20):
  """
  Computes E[X(1-X)] where X is a WF diffusion with frequency-dependent (FD) selection
  xi(x) = s + d(1-2x)
  and mutation rates given by theta.

  Parameters
  ----------
  thetaN: tuple of two positive floats: mutation rates
  sN: float: directional selection
  dN: float: dominance coefficient
  klim: int: the precision of the approximation
  """
  return (thetaN[0]*thetaN[1]/((thetaN[0]+thetaN[1])*(thetaN[0]+thetaN[1]+1/2))
         *integral_WF_FD(2*thetaN[0]+1,2*thetaN[1]+1,2*sN,2*dN,klim)
         /integral_WF_FD(2*thetaN[0],2*thetaN[1],2*sN,2*dN,klim))



def mean_phenotype_strong(sN,dN,thetaN,L,list_alpha,list_proba_alpha,klim=20):
  """Finds the mean phenotype.
  Arguments:
  ----------
  thetaN: mutation rates (tuple of two positive floats)

  sN: effective selection coefficient s^*

  list_alpha: list of classes of values for gene effect. Typically np.linspace(a,b)

  list_proba_alpha: the list of probabilities of each class in list_alpha
  """
  return L*np.sum(2*list_alpha*mean_X_FD(sN*list_alpha,
                                         dN*list_alpha**2,
                                         thetaN,
                                         klim=20)
                  *list_proba_alpha)

def genetic_variance_strong(sN,om2,thetaN,N,L,list_alpha,list_proba_alpha,klim=20):
  """Finds the mean E[2 alpha**2 X(1-X)].
  Arguments:
  ----------
  thetaN: mutation rates (tuple of two positive floats)

  sN: effective selection coefficient s^*

  list_alpha: list of classes of values for gene effect. Typically np.linspace(a,b)

  list_proba_alpha: the list of probabilities of each class in list_alpha
  """
  return L*np.sum(2*list_alpha**2*var_WF_FD(sN*list_alpha,
                                            -N/(2*om2)*list_alpha**2,
                                            thetaN,
                                            klim=20)
                  *list_proba_alpha)

def match_equation_strong(sN,dN,thetaN,eta,L,list_alpha,list_proba_alpha,klim=20):
  """Finds the absolute value of the distance between the mean phenotype and the
  optimum eta for strong selection"""
  return np.abs(mean_phenotype_strong(sN,dN,thetaN,L,list_alpha,list_proba_alpha,klim)
                -eta)


def find_sN_strong(om2,
                   thetaN,
                   eta,
                   N,
                   L,
                   list_alpha,
                   list_proba_alpha,
                   tolerance=1e-5,
                   a0=-1000,
                   b0=1000,
                   klim=20):
  """Finds the value of s^* given by the equation for strong selection for a WF
  diffusion with frequency-dependent selection
  xi(x,alpha) = s^* alpha +  N alpha**2/(2 om2)  (1-2x)

  Parameters:
  ----------
  om2: inverse strength of selection
  thetaN: mutation rates (the probability of mutation at one locus from 0 to +1 is
          thetaN[0] per organism per generation)
  eta: optimum
  N: population size (number of diploid organisms)
  L: number of loci
  list_alpha: list of values of additive effects
  list_proba_alpha: list of probabilities of the additive effects
  tolerance: precision
  a0,b0: domain in which the search is made
  """
  return gss(lambda sN: match_equation_strong(sN,
                                              -N/(2*om2),
                                              thetaN,
                                              eta,
                                              L,
                                              list_alpha,
                                              list_proba_alpha,
                                              klim=20),
             a0,b0, #Boundaries of the golden-section search
             tolerance=tolerance
  )
