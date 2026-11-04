"""Exact rational scalar bounds only: no NPZ, model, reference, or quadrature calls."""
from fractions import Fraction as F
from math import factorial
from decimal import Decimal, localcontext
import json
from pathlib import Path

TOL = F(47,250000000000000)

def encoded(x):
    with localcontext() as c:
        c.prec=70
        dec=str(Decimal(x.numerator)/Decimal(x.denominator))
    return dict(numerator=x.numerator,denominator=x.denominator,decimal=dec)

def gl_upper(n,s):
    # exp(2s) <= 1/(1-2s), 0<=2s<1, termwise exponential/geometric series.
    C=F(2**(2*n+1)*factorial(n)**4,(2*n+1)*factorial(2*n)**3)
    polynomial=sum((s**(2*n-k)/F(2**k*factorial(k)*factorial(2*n-2*k)) for k in range(n+1)),F())
    return C*factorial(2*n)*polynomial/(2*(1-2*s))

rows=[]
for n in (1,2,3,4,6,8):
    k=next(k for k in range(2,100) if gl_upper(n,F(1,2**k))<=TOL)
    next_upper = encoded(gl_upper(n,F(1,2**(k-1)))) if k>2 else None
    rows.append(dict(order=n,threshold=f'2^-{k}',threshold_exact=encoded(F(1,2**k)),upper=encoded(gl_upper(n,F(1,2**k))),next_dyadic_upper=next_upper))

def trim(p):
    while len(p)>1 and p[-1]==0:p.pop()
    return p

def plus(p,q):
    out=[F()]*max(len(p),len(q))
    for i,v in enumerate(p):out[i]+=v
    for i,v in enumerate(q):out[i]+=v
    return trim(out)

def scale(p,x):return trim([x*v for v in p])
def legendre(n):
    old,new=[F(1)],[F(0),F(1)]
    if n==0:return old
    for j in range(1,n):old,new=new,scale(plus(scale([F()]+new,F(2*j+1)),scale(old,F(-j))),F(1,j+1))
    return new

def remainder(p,divisor):
    p=p[:]
    while len(p)>=len(divisor):
        mult=p[-1]/divisor[-1];offset=len(p)-len(divisor)
        for i,x in enumerate(divisor):p[i+offset]-=mult*x
        trim(p)
    return p

def integral(p):return sum((2*x/F(i+1) for i,x in enumerate(p) if i%2==0),F())

# Uniform GL4/GL6 counterexamples via exact polynomial remainders, not integration.
# q=1/8 corresponds only to local m=0,h=1/2; exp(-q t^2), t in [-1,1].
# At each rule's roots Q(poly)=Q(poly mod Pn)=integral(poly mod Pn).
# The Taylor remainder contributes <=4*q^(K+1)/(K+1)! to I-Q.
q=F(1,8);K=20;counterexamples=[]
for n in (4,6,8):
    pn=legendre(n);I=F();E=F()
    for k in range(K+1):
        p=[F()]*(2*k)+[F(1)]
        weight=(-q)**k/F(factorial(k))
        exact=F(2,2*k+1)
        I+=weight*exact
        E+=weight*(exact-integral(remainder(p,pn)))
    single=2*q**(K+1)/F(factorial(K+1));double=2*single
    lower=max(F(),abs(E)-double)/(I+single)
    upper=(abs(E)+double)/(I-single)
    counterexamples.append(dict(order=n,relative_error_lower=encoded(lower),relative_error_upper=encoded(upper),exceeds_local_ceiling=lower>TOL,below_local_ceiling=upper<TOL))

# Positive-series bound: q=a+c <= 1/8 under b+2a <=1/4,
# a=h^2/2,b=abs(m*h),c=b^2/4.
series=[]
for N in (4,5,6,7,8):
    upper=F(1)
    for r in range(1,N+2):upper*=F(1,8)/F(2*r+1,2)
    upper/=1-F(1,8)/F(2*N+5,2)
    series.append(dict(last_included_degree=N,relative_tail_upper=encoded(upper),passes=upper<=TOL))

output=dict(scope='Exact rational symbolic bounds on local scalar synthetic parameters only; no arrays or integration.',local_ceiling=encoded(TOL),dyadic_thresholds=rows,uniform_span_quarter_bounds=[dict(order=n,upper=encoded(gl_upper(n,F(1,4)))) for n in (4,6,8)],legendre_polynomial_counterexample=counterexamples,positive_series_uniform_remainders=series,integrator_calls=0,npz_decodes=0)
Path(__file__).with_name('results.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
for r in rows:print('GL',r['order'],r['threshold'],'upper',r['upper']['decimal'])
for r in counterexamples:print('GL counterexample',r['order'],r['relative_error_lower']['decimal'],r['relative_error_upper']['decimal'])
for r in series:print('positive series',r['last_included_degree'],r['relative_tail_upper']['decimal'])
