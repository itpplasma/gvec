#!/usr/bin/env python3
"""Exact independent local energy difference; source-matched GVEC lambda force."""
from fractions import Fraction as Q
import json,hashlib,pathlib
fixtures=[(Q(5,4),Q(1,3),Q(13,5),Q(7,3),Q(11,7),Q(-5,4),Q(1,9),Q(-2,11),Q(2,5),Q(-3,8)),(Q(3),Q(-1),Q(2),Q(5),Q(-2),Q(3),Q(-1,4),Q(1,6),Q(-2,3),Q(1,5)),(Q(7),Q(2),Q(4),Q(11,3),Q(4),Q(-2),Q(1,2),Q(-1,5),Q(1,3),Q(2,7))]
results=[]
for k,(gtt,gtz,gzz,jac,chip,phip,lt,lz,yt,yz) in enumerate(fixtures):
 assert gtt>0 and gtt*gzz-gtz*gtz>0 and jac>0
 def energy(t):
  bt=chip-phip*(lz+t*yz);bz=phip*(1+lt+t*yt)
  return (gtt*bt*bt+2*gtz*bt*bz+gzz*bz*bz)/(2*jac)
 h=Q(1,97);exact_derivative=(energy(h)-energy(-h))/(2*h)
 bt=chip-phip*lz;bz=phip*(1+lt);Btheta=(gtt*bt+gtz*bz)/jac;Bzeta=(gtz*bt+gzz*bz)/jac
 native_negative_derivative=phip*(Btheta*yz-Bzeta*yt)
 corrected_document_derivative=Btheta*(-phip*yz)+Bzeta*(phip*yt)
 old_document_derivative=Btheta*(phip*yz)+Bzeta*(-phip*yt)
 assert exact_derivative==corrected_document_derivative
 assert -exact_derivative==native_negative_derivative
 assert exact_derivative!=old_document_derivative
 results.append({'fixture':k,'exact_energy_derivative':str(exact_derivative),'native_negative_derivative':str(native_negative_derivative),'old_document_derivative':str(old_document_derivative),'pass':True})
print(json.dumps({'source_pin':'c0dc66fe2b9faa147c76a728e5cd053ce693d472','source_mapping':'mhd3d_evalfunc.F90:796–797 native coefficients +PhiPrime*Bcov_theta for test_zeta and −PhiPrime*Bcov_zeta for test_theta','method':'Fraction exact central difference of quadratic energy,3 SPD metrics with nonzero cross term; no CAS or upstream dependency','fixtures':results,'scope':'Local lambda variation, not full discrete projection/convergence proof','oracle_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()},indent=2))
