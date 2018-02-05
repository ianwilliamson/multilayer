from numpy import (pi, cos, sin, sqrt, zeros, sum, meshgrid, transpose, reshape)

C_CONST = 299792458

def _m11_tm(d,k):
	return cos(k*d,dtype=complex)

def _m22_tm(d,k):
	return cos(k*d,dtype=complex)

def _m12_tm(d,k):
	return 1j/k*sin(k*d,dtype=complex)

def _m21_tm(d,k):
	return 1j*k*sin(k*d,dtype=complex)

def _m11_te(d,k):
	return cos(k*d,dtype=complex)

def _m22_te(d,k):
	return cos(k*d,dtype=complex)

def _m12_te(d,k):
	return 1j/k*sin(k*d,dtype=complex)

def _m21_te(d,k):
	return 1j*k*sin(k*d,dtype=complex)

# This function is based on 
# https://jameshensman.wordpress.com/2010/06/14/
# multiple-matrix-multiplication-in-numpy/
def _product(A,B):
	N = A.shape[0]
	assert(N == B.shape[0])
	return sum(transpose(A,(0,2,1)).reshape(N,2,2,1)*B.reshape(N,2,1,2),-3)

def _Mi_tm(d,k):
	N = k.size
	M = zeros((N,2,2),dtype=complex)

	M[:,0,0] = _m11_tm(d,k)
	M[:,0,1] = _m12_tm(d,k)
	M[:,1,0] = _m21_tm(d,k)
	M[:,1,1] = _m22_tm(d,k)
	return M

def _Mi_te(d,k):
	N = k.size
	M = zeros((N,2,2),dtype=complex)

	M[:,0,0] = _m11_te(d,k)
	M[:,0,1] = _m12_te(d,k)
	M[:,1,0] = _m21_te(d,k)
	M[:,1,1] = _m22_te(d,k)
	return M

def build_M(d,n,k0,beta,te=True):
	assert(d.size == n.size)
	assert(k0.size == beta.size)
	N = k0.size
	if te:
		_Mi = lambda d,k: _Mi_te(d,k)
	else:
		_Mi = lambda d,k: _Mi_tm(d,k)

	ki = sqrt( (k0*n[1])**2 - beta**2,dtype=complex )
	M  = _Mi(d[1],ki)
	for i in range(2,d.size-1):
		ki = sqrt( (k0*n[i])**2 - beta**2,dtype=complex)
		Mi = _Mi(d[i],ki)
		M  = _product(M,Mi)
	return M

def calc_F(d,n,omega,beta,te=True):
	Nb = beta.size
	Nw = omega.size
	(k0,beta) = meshgrid(omega/C_CONST,beta)
	k0 = k0.ravel()
	beta = beta.ravel()
	M = build_M(d,n,k0,beta,te)
	m11 = M[:,0,0]
	m12 = M[:,0,1]
	m21 = M[:,1,0]
	m22 = M[:,1,1]
	gamma0 = sqrt((k0*n[0])**2-beta**2,dtype=complex)
	gammas = sqrt((k0*n[-1])**2-beta**2,dtype=complex)
	F = gamma0*m11+gamma0*gammas*m12+m21+gammas*m22
	return F.reshape((Nb,Nw)).transpose()