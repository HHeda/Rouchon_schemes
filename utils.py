from email.mime import message
import itertools
import sympy as sp
import pickle
import time
import os
from multiprocessing import Pool


dt = sp.symbols("dt", positive=True)
G = sp.symbols("G", commutative=False)
L = sp.symbols("L", commutative=False)

def partitions(n, k): # generates all partitions of n into k non-negative integers
    for c in itertools.combinations(range(n+k-1), k-1):
        yield [b-a-1 for a, b in zip((-1,)+c, c+(n+k-1,))]

def partition_order(N): # generates all partitions of N into (N = n + 2m) with n, m >=0
    for c in partitions(int(2*N), 2):
        if c[1]%2:
            yield [c[0], c[1]//2]

def variable(i, k): # returns the variable t_i, replacing t_0 = dt and t_{k+1} = 0
    if i == 0:
        return dt
    elif i >= k+1:
        return 0
    return sp.symbols(f"t_{i}", positive=True)

def products(n, k): # generates all products of n times G and k times L, corresponding to partitions of n into k+1 parts
    for p in partitions(n, k+1):
        res = 1
        for c in p[:-1]:
            res *=  G**c*L if c >= 1 else L
        res*= G**p[-1]
        yield res

def functions(n, k): # generates all k variable functions of total degree n, corresponding to partitions of n into k+1 parts
    for p in partitions(n, k+1):
        res = dt/dt # proxy for 1 in sympy
        for i, c in enumerate(p):
            res *= (variable(i, k)-variable(i+1, k))**c/sp.factorial(c)
        yield state(res, k)

def average(exp, i, k): # integrates over variable i between variable i-1 and i+1, substituting variable i+1 by variable i afterwards
    return sp.integrate(exp, (variable(i, k), variable(i+1, k), variable(i-1, k))).subs([(variable(j+1, k), variable(j, k)) for j in range(i, k)])

def destroy(exp, k): # annihilation operator acting on k-photon state
    return 1/sp.sqrt(dt)*sum([average(exp, i, k) for i in range(1, k+1)])

def create(exp, k): # creation operator acting on k-photon state
    return 1/sp.sqrt(dt)*sum([exp.subs([(variable(j, k+1), variable(j+1, k+1)) for j in range(k, i-1, -1)]) for i in range(1, k+2)])

def create_n(exp, k, n): # creation operator acting n times on k-photon state
    res = exp
    for i in range(n):
        res = create(res, k+i)
    return res

def destroy_n(exp, k, n): # annihilation operator acting n times on k-photon state
    res = exp
    for i in range(n):
        res = destroy(res, k-i)
    return res

# def decomposition(exp, k):
#     ais = [exp]
#     for i in range(0, k):
#         ais.append(destroy(ais[-1], k-i))
#     ois = (k+1)*[0]
#     for i in range(k, -1, -1):
#         ois[k-i] = ais[i]/(sp.sqrt(sp.factorial(i))) - sum([create_n(ois[k-i-j], k-i-j, j)*sp.sqrt(sp.factorial(j+i)/sp.factorial(i))/sp.factorial(j) for j in range(1, k-i+1)])
#     return ois

def scalprod(exp1, exp2, k): # scalar product between two k-photon states
    if k == 0:
        return exp1*exp2
    conditions = ([(variable(i, k), 0, variable(i-1, k)) for i in range(k, 0, -1)])
    return sp.integrate(exp1*exp2, *conditions)


def generate_terms(order): # generates all terms of given order
    to = order+0.5
    res = [[]]*int(to*2)
    for c in partition_order(to):
        res[c[0]] = [d for d in functions(c[1], c[0])]
    return res

def generate_operators(order): # generates all operators of given order (in the same order as terms)
    to = order+0.5
    res = [[]]*int(to*2)
    for c in partition_order(to):
        res[c[0]] = [d for d in products(c[1], c[0])]
    return res

def Gram_Schmidt(l): # Gram-Schmidt orthonormalization of list of states l
    if l == []:
        return l
    res = [l[0]]
    for c in l[1:]:
        to_append = (c - sum([(scalprod_states(c, d)/scalprod_states(d, d)).simplify()*d for d in res]))
        if to_append.simplify().f != 0:
            res.append(to_append)
    res_normalized = [d.normalized() for d in res]
    return res_normalized

def proj_state(s, s0): #projects s onto s0
    return (scalprod_states(s, s0)/scalprod_states(s0, s0)*s0).simplify()

def poolmap_proj(l1, l2, pool = None): # projects all elements of l1 onto corresponding elements of l2
    if pool is None:
        return [proj_state(c, d) for c, d in zip(l1, l2)]
    else:
        return pool.starmap(proj_state, [(c, d) for c, d in zip(l1, l2)])


def poolmap_proj_async(l1, l2, pool = None): 
    # projects all elements of l1 onto corresponding elements of l2 using asynchronous pool mapping
    if pool is None:
        return [proj_state(c, d) for c, d in zip(l1, l2)]
    else:
        return pool.starmap_async(proj_state, [(c, d) for c, d in zip(l1, l2)])

def decomposition(s): # decomposes state s into photon number basis
    return s.decomposition()

# def extend_Gram_Schmidt(l, new, pool=None):
#     res = l[:]
#     for c in new:
#         to_append = (c - sum(poolmap_proj(c, res, pool=pool))).simplify()
#         if to_append.f != 0:
#             res.append(to_append)
#     res_normalized = [d.normalized() for d in res[len(l):]]
#     return res_normalized

def extend_Gram_Schmidt(basis, new, pool=None): # Gram-Schmidt extension of basis with new elements new using pool for parallelization
    # First computes all new projections onto the old basis asynchronously, and then applies the Gram-Schmidt process. For each new element, if a new basis element is found, the projections onto it are computed synchronously.
    res = []
    projs_l = [poolmap_proj_async([c]*len(basis), basis, pool=pool) for c in new]        
    followup_projectors = []
    isfirsttime = True
    if pool is None:
        isfirsttime = False # to never call get()
        matprojs = projs_l
    for i, c in enumerate(new):
        if isfirsttime: # it means that there are now new element, we stay asynchronous
            to_append = (c - sum(projs_l[i].get())).simplify() 
        else: # if there is a new element in basis, we use matrpojs which is already computed
            to_append = (c - sum(matprojs[i])).simplify()

        if to_append.f != 0: 
            if isfirsttime: # only the first time we need to get the projectors from the async calls
                matprojs = [c.get() for c in projs_l]
                isfirsttime = False
            res0 = to_append.normalized()
            res.append(res0)  
            followup_projector = [state(0, 0)]*i+[res0]+poolmap_proj(new[i+1:], [res0]*len(new[i+1:]), pool=pool)
            for j in range(len(new)):
                matprojs[j].append(followup_projector[j])
    return res

# def extend_Gram_Schmidt(l, new, pool=None):
#     res = []
#     projs_l = [poolmap_proj_async(c, l, pool=pool) for c in new]
#     for i, c in enumerate(new):
#         to_append_old = (c - sum(projs_l[i].get())).simplify()
#         to_append_new = (to_append_old - sum(poolmap_proj(c, res, pool=pool))).simplify()
#         if to_append_new.f != 0:
#             res.append(to_append_new.normalized())
#     #res_normalized = [d.normalized() for d in res]
#     return res


class state(): # a class for a state with a given number of photons nphotons and a function exp
    def __init__(self, exp, nphotons):
        self.nphotons = nphotons
        self.f = sp.simplify(exp)
        if self.f == 0:
            self.nphotons = 0

    def __mul__(self, scalar):
        if scalar == 0:
            return state(0, 0)
        if type(scalar) == type(self):
            if scalar.nphotons == 0:
                return state(self.f*scalar.f, self.nphotons)
            else:
                raise ValueError("Cannot multiply two states with photons")
        return state(self.f*scalar, self.nphotons)
    
    __rmul__ = __mul__

    def __add__(self, other):
        if other == 0:
            return self
        if other.f == 0:
            return self
        if self.f == 0:
            return other
        if self.nphotons != other.nphotons:
            raise ValueError("Cannot add states with different number of photons")
        return state(self.f + other.f, self.nphotons)
    __radd__ = __add__

    def __sub__(self, other):
        if other == 0:
            return self
        if other.f == 0:
            return self
        if self.nphotons != other.nphotons:
            raise ValueError("Cannot add states with different number of photons")
        return state(self.f - other.f, self.nphotons)
    
    def __rsub__(self, other):
        if other == 0:
            return self*-1
        if other.f == 0:
            return self*-1
        if self.nphotons != other.nphotons:
            raise ValueError("Cannot add states " \
            " different number of photons")
        return state(other.f - self.f, self.nphotons)

    def __truediv__(self, scalar):
        return state(self.f/scalar, self.nphotons)

    def __repr__(self):
        return self.f.__repr__()
    def __str__(self):
        return self.f.__str__()
    

    def norm(self):
        return sp.sqrt(scalprod(self.f, self.f, self.nphotons))
    
    def create(self):
        return state(create(self.f, self.nphotons), self.nphotons+1)
    
    def create_n(self, n):
        return state(create_n(self.f, self.nphotons, n), self.nphotons+n)

    def destroy(self):
        if self.nphotons == 0:
            return state(0, 0)
        return state(destroy(self.f, self.nphotons), self.nphotons-1)

    def destroy_n(self, n):
        return state(destroy_n(self.f, self.nphotons, n), self.nphotons-n)

    def decomposition(self): 
        ais = [self]
        for i in range(0, self.nphotons):
            ais.append(ais[-1].destroy())
        ois = (self.nphotons+1)*[state(0, 0)]
        adois = [[] for _ in range(self.nphotons+1)]
        for i in range(self.nphotons, -1, -1):
            ois[self.nphotons-i] = ais[i]/(sp.sqrt(sp.factorial(i))) - sum([adois[self.nphotons-i-j][j]*sp.sqrt(sp.factorial(j+i)/sp.factorial(i))/sp.factorial(j) for j in range(1, self.nphotons-i+1)])
            adois[self.nphotons-i].append(ois[self.nphotons-i])
            for _ in range(self.nphotons-i,self.nphotons):
                adois[self.nphotons-i].append(adois[self.nphotons-i][-1].create())
        return ois
    
    def normalized(self): # returns normalized state
        norm  = self.norm()
        if norm == 0:
            return state(0, 0)
        return state(self.f/norm, self.nphotons)

    def simplify(self): # simplifies the function of the state
        return state(sp.simplify(self.f), self.nphotons)


def scalprod_states(s1, s2): # scalar product between two states
    if s1.nphotons != s2.nphotons:
        return 0
    # log_write(scalprod(s1.f, s2.f, s1.nphotons))
    return scalprod(s1.f, s2.f, s1.nphotons)

class Lindblad_scheme: # a class for the scheme at a given order. Computes all terms and operators in the unitary expansion, and updates the environment states for the Lindblad scheme
    def __init__(self, logfile_name = "log.txt"): # initializes the scheme at order 0
        self.logfile_name = logfile_name
        self.order = 0 # the order of the unitary expansion
        self.Lindblad_orders = [0] # the orders of the Lindblad Kraus operators
        self.term_order = 0 # the expansion order of the Kraus maps (twice the order of the unitary expansion)
        self.term_orders = [0] # the list of expansion orders of the Kraus maps
        self.terms = [[x for xs in generate_terms(self.order) for x in xs]] # the list of terms in the unitary expansion
        self.operators = [[x for xs in generate_operators(self.order) for x in xs]] # corresponding operators
        self.Lindblad_states = Gram_Schmidt(self.terms[0]) # the states of the environment for the Lindblad scheme

    def increase_order(self, pool=None): # give a pool for parallelization
        ttot = time.time()
        t0= time.time()
        self.order += 0.5 # increasing the order of the unitary expansion
        self.term_order += 0.5 # increasing the order of the Kraus map
        self.term_orders.append(self.term_order) # updating the list of Kraus map orders
        new_terms = [x for xs in generate_terms(self.term_order) for x in xs] # generating new terms of the unitary expansion
        new_operators = [x for xs in generate_operators(self.term_order) for x in xs] # generating corresponding operators
        self.terms.append(new_terms)
        self.operators.append(new_operators)
        self.term_order += 0.5 # doing it again to reach the next integer order
        self.term_orders.append(self.term_order)
        new_terms = [x for xs in generate_terms(self.term_order) for x in xs]
        new_operators = [x for xs in generate_operators(self.term_order) for x in xs]
        self.terms.append(new_terms)
        self.operators.append(new_operators)
        log_write(f"New terms and operators generated in {round(time.time()-t0, 3)} seconds", logfile_name = self.logfile_name)
        # new_decomposed_terms = [[d.decomposition() for d in c] for c in new_terms]
        indice_half_order = int(2*self.order) # index of the terms to add to the Gram-Schmidt process
        t0 = time.time()
        new_Lindblad_states = extend_Gram_Schmidt(self.Lindblad_states, self.terms[indice_half_order], pool=pool) # extend the basis of Lindblad states
        log_write(f"Gram-Schmidt extended in {round(time.time()-t0, 3)} seconds", logfile_name = self.logfile_name)
        t0 = time.time()
        self.new_Lindblad_states = [c.simplify() for c in new_Lindblad_states] # simplifying the new Lindblad states
        self.Lindblad_states = self.Lindblad_states + self.new_Lindblad_states # updating the Lindblad states

        log_write(f"Simplification of new terms done in {round(time.time()-t0, 3)} seconds", logfile_name = self.logfile_name)
        self.Lindblad_orders.extend([self.order]*(len(new_Lindblad_states))) # update the orders of the Lindblad states
        return ttot, new_Lindblad_states


class SME_scheme(Lindblad_scheme): # a class for the scheme at a given order. Computes all terms and operators in the unitary expansion, and updates the environment states for the Lindblad and SME schemes
    def __init__(self, logfile_name = "log.txt"): # initializes the scheme at order 0
        super().__init__(logfile_name = logfile_name)
        decomposed_Lindblad_states = [d.decomposition() for d in self.Lindblad_states] # decomposition of the Lindblad states on the integration mode
        self.SME_states  = Gram_Schmidt([x for xs in decomposed_Lindblad_states for x in xs]) # the states of the environment for the SME scheme
        self.SME_orders = [0] # the list of expansion orders of the SME maps
        self.basis = [[c.create_n(i)/sp.sqrt(sp.factorial(i)) for i in range(int(4*self.order)-c.nphotons+1)] for c in self.SME_states] # the basis for the SME scheme

    def increase_order(self, pool=None): # give a pool for parallelization
        ttot, new_Lindblad_states = super().increase_order(pool=pool)
        t0 = time.time()
        new_decomposed_Lindblad_states = pool.map(decomposition, new_Lindblad_states) if pool is not None else [x.decomposition() for x in new_Lindblad_states] # decomposing the new Lindblad states on the integration mode
        new_decomposed_Lindblad_states = [x for xs in new_decomposed_Lindblad_states for x in xs]

        log_write(f"Decomposition of new basis done in {round(time.time()-t0, 3)} seconds", logfile_name = self.logfile_name)
        t0 = time.time()
        new_SME_states = extend_Gram_Schmidt(self.SME_states, new_decomposed_Lindblad_states, pool = pool) # update the SME basis states
        self.SME_states.extend(new_SME_states)
        self.SME_orders.extend([self.order]*(len(new_SME_states)))

        log_write(f"Gram-Schmidt of new SME states done in {round(time.time()-t0, 3)} seconds, number of SME Kraus operators: {len(self.SME_states)}", logfile_name = self.logfile_name)

        # Updating the basis
        t0 = time.time()
        for i in range(len(self.basis)): # increasing the number of photons in the existing basis elements
            self.basis[i].append(self.basis[i][-1].create()/sp.sqrt(len(self.basis[i])))
            self.basis[i].append(self.basis[i][-1].create()/sp.sqrt(len(self.basis[i])))
        max_nphotons = (self.order-1)*2
        new_basis = [[c] for c in new_SME_states]
        t0 = time.time()
        # Creating the right number of photons in the basis
        con = True
        while con:
            con = False
            for b in new_basis:
                if b[-1].nphotons < max_nphotons:
                    b.append(b[-1].create()/sp.sqrt(len(b)))
                    con = True
        self.basis.extend(new_basis)
        log_write(f"Basis updated in {round(time.time()-t0, 3)} seconds", logfile_name = self.logfile_name)
        log_write(f"Order increased to {int(2*self.order)} in {round(time.time()-ttot, 3)} seconds", logfile_name = self.logfile_name)


def Lindblad_Kraus_ops(s, pool = None): # computes the Lindblad Kraus operators from the scheme s. Give pool for multiprocesing
    t0 = time.time()
    Mmus = [0 for _ in s.Lindblad_states]
    all_scalprods = [[] for _ in s.Lindblad_states]
    # return basis
    # return basis, Mmus
    for i, el in enumerate(s.Lindblad_states[:]):
        # log_write(len(b))
        for lterms, loperators, lorders in zip(s.terms[:], s.operators[:], s.term_orders[:]):
            if s.Lindblad_orders[i]+lorders <= 2*s.order:
                if pool is None:
                    Mmus[i] = Mmus[i] + sum([scalprod_states(el, term)*op for term, op in zip(lterms, loperators)])
                else:
                    all_scalprods[i].append(pool.starmap_async(scalprod_states, [(el, term) for term in lterms]))
    if pool is not None:
        for i, el in enumerate(s.Lindblad_states[:]):
            for lterms, loperators, lorders, scalprod in zip(s.terms[:], s.operators[:], s.term_orders[:], all_scalprods[i]):
                if s.Lindblad_orders[i]+lorders <= 2*s.order:
                    Mmus[i] = Mmus[i] + sum([c*op for c, op in zip(scalprod.get(), loperators)])
    log_write(f"Lindblad Kraus operators computed in {round((time.time()-t0)/60, 2)} minutes", logfile_name = s.logfile_name)
    return Mmus

def SME_Kraus_ops(s, pool = None): # computes the SME Kraus operators from the scheme s. Give pool for multiprocesing
    t0 = time.time()
    Mmus = [[0 for _ in b] for b in s.basis]
    all_scalprods = [[[] for _ in b] for b in s.basis]
    # return basis
    # return basis, Mmus
    for i, b in enumerate(s.basis[:]):
        # log_write(len(b))
        for j, el in enumerate(b):
            # log_write(sum([scalprod_states(el, term)*op for term, op in zip(terms, operators)]))
            for lterms, loperators, lorders in zip(s.terms[:], s.operators[:], s.term_orders[:]):
                if s.SME_orders[i]+lorders <= 2*s.order:
                    if pool is None:
                        Mmus[i][j] = Mmus[i][j] + sum([scalprod_states(el, term)*op for term, op in zip(lterms, loperators)])
                    else:
                        all_scalprods[i][j].append(pool.starmap_async(scalprod_states, [(el, term) for term in lterms]))
                        
    if pool is not None:
        for i, b in enumerate(s.basis[:]):
            for j, el in enumerate(b):
                for lterms, loperators, lorders, scalprod in zip(s.terms[:], s.operators[:], s.term_orders[:], all_scalprods[i][j]):
                    if s.SME_orders[i]+lorders <= 2*s.order:
                        Mmus[i][j] = Mmus[i][j] + sum([c*op for c, op in zip(scalprod.get(), loperators)])
    log_write(f"SME Kraus operators computed in {round((time.time()-t0)/60, 2)} minutes", logfile_name=s.logfile_name)
    return Mmus


def log_write(message, init = False, logfile_name='log.txt'):
    if init:
        with open(logfile_name, 'w') as logfile:
            logfile.write(message+'\n')
            logfile.close()
    else:
        with open(logfile_name, 'a') as logfile:
            logfile.write(message+'\n')
            logfile.close()