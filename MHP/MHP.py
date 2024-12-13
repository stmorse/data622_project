import time as T
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

class MHP:
    def __init__(self, alpha=[[0.5]], mu=[0.1], omega=1.0):
        '''params should be of form:
        alpha: numpy.array((u,u)), mu: numpy.array((,u)), omega: float'''
        
        self.alpha, self.mu, self.omega = np.array(alpha), np.array(mu), omega
        self.dim = self.mu.shape[0]
        self.check_stability()

    def check_stability(self):
        ''' check stability of process (max alpha eigenvalue < 1)'''
        w,v = np.linalg.eig(self.alpha)
        me = np.amax(np.abs(w.real))
        print(f'Max eigenvalue: {me:1.5f}')
        if me >= 1.:
            print('(WARNING) Unstable.')
        else:
            print('Appears stable')

    def generate_seq(self, horizon, data=[[0,0]]):
        '''Generate a sequence based on mu, alpha, omega values. 
        Uses Ogata's thinning method, with some speedups, noted below'''

        data = np.array(data)

        Istar = np.sum(self.mu)
        s = np.random.exponential(scale=1./Istar)

        # attribute (weighted random sample, since sum(mu)==Istar)
        n0 = np.random.choice(np.arange(self.dim), 
                              1, 
                              p=(self.mu / Istar))[0]
        data = np.append(data, [[s, n0]], axis=0)

        # value of \lambda(t_k) where k is most recent event
        # starts with just the base rate
        lastrates = self.mu.copy()

        decIstar = False
        while True:
            tj, uj = data[-1,0], int(data[-1,1])

            if decIstar:
                # if last event was rejected, decrease Istar
                Istar = np.sum(rates)
                decIstar = False
            else:
                # otherwise, we just had an event, so recalc Istar (inclusive of last event)
                Istar = np.sum(lastrates) + \
                        self.omega * np.sum(self.alpha[:,uj])

            # generate new event
            s += np.random.exponential(scale=1./Istar)

            # calc rates at time s (use trick to take advantage of rates at last event)
            rates = self.mu + np.exp(-self.omega * (s - tj)) * \
                    (self.alpha[:,uj].flatten() * self.omega + lastrates - self.mu)

            # attribution/rejection test
            # handle attribution and thinning in one step as weighted random sample
            diff = Istar - np.sum(rates)
            try:
                n0 = np.random.choice(np.arange(self.dim+1), 1, 
                                      p=(np.append(rates, diff) / Istar))[0]
            except ValueError:
                # by construction this should not happen
                print('Probabilities do not sum to one.')
                return data

            if n0 < self.dim:
                data = np.append(data, [[s, n0]], axis=0)
                # update lastrates
                lastrates = rates.copy()
            else:
                decIstar = True

            # if past horizon, done
            if s >= horizon:
                return data

    #-----------
    # EM LEARNING
    #-----------

    def EM(self, Ahat=None, mhat=None, omega=None, seq=[], 
           smx=None, tmx=None, regularize=False, 
           Tm=-1, maxiter=100, epsilon=0.01, verbose=True, return_p=False):
        '''implements MAP EM. Optional to regularize with `smx` and `tmx` matrix (shape=(dim,dim)).
        In general, the `tmx` matrix is a pseudocount of parent events from column j,
        and the `smx` matrix is a pseudocount of child events from column j -> i, 
        however, for more details/usage see https://stmorse.github.io/docs/orc-thesis.pdf'''
        
        # use stored values unless something passed
        Ahat = Ahat if Ahat is not None else self.alpha
        mhat = mhat if mhat is not None else self.mu
        omega = omega if omega is not None else self.omega

        N = len(seq)
        dim = mhat.shape[0]
        Tm = float(seq[-1,0]) if Tm < 0 else float(Tm)
        sequ = seq[:,1].astype(int)

        p_ii = np.random.uniform(0.01, 0.99, size=N)
        p_ij = np.random.uniform(0.01, 0.99, size=(N, N))

        t0 = T.time()

        # PRECOMPUTATIONS

        # indicator matrix for events
        S = np.zeros((N, dim), dtype=np.float32)
        S[np.arange(N), sequ] = 1

        # diffs[i,j] = t_i - t_j for j < i (o.w. zero)
        diffs = pairwise_distances(np.array([seq[:,0]]).T, metric = 'euclidean')
        diffs[np.triu_indices(N)] = 0

        # kern[i,j] = omega*np.exp(-omega*diffs[i,j])
        kern = omega*np.exp(-omega*diffs)

        colidx = np.tile(sequ.reshape((1,N)), (N,1))
        rowidx = np.tile(sequ.reshape((N,1)), (1,N))

        # approx of Gt sum in a_{uu'} denom
        p_ones = np.ones((N, N))
        p_ones[np.triu_indices(N)] = 0
        seqcnts = (S.T @ p_ones) @ S   # (dim, dim)
        seqcnts[np.where(seqcnts == 0)] = 1  # hack

        k = 0
        old_LL = -10000
        START = T.time()
        while k < maxiter:
            Auu = Ahat[rowidx, colidx]
            ag = np.multiply(Auu, kern)
            ag[np.triu_indices(N)] = 0

            # compute m_{u_i}
            mu = mhat[sequ]

            # compute total rates of u_i at time i
            rates = mu + np.sum(ag, axis=1)

            # compute matrix of p_ii and p_ij  (keep separate for later computations)
            p_ij = np.divide(ag, np.tile(np.array([rates]).T, (1,N)))
            p_ii = np.divide(mu, rates)

            # compute mhat:  mhat_u = (\sum_{u_i=u} p_ii) / T
            mhat = np.array([np.sum(p_ii[np.where(seq[:,1]==i)]) \
                             for i in range(dim)]) / Tm
            
            if regularize:
                Ahat = np.divide((S.T @ p_ij @ S) + (smx - 1),
                                 seqcnts + tmx)
            else:
                Ahat = np.divide(S.T @ p_ij @ S, seqcnts)

            if k % 10 == 0:
                try:
                    term1 = np.sum(np.log(rates))
                except:
                    print('Log error!')
                term2 = Tm * np.sum(mhat)
                term3 = np.sum([
                    np.sum([Ahat[u,int(seq[j,1])] for j in range(N)]) 
                    for u in range(dim)])
                new_LL = (1./N) * (term1 - term2 - term3)

            k += 1

        if verbose:
            print(f'Reached max iter {maxiter} (LL: {new_LL}) ... {T.time()-t0:.3f}')

        self.Ahat = Ahat
        self.mhat = mhat
        
        if return_p:
            return Ahat, mhat, p_ii, p_ij
        else:
            return Ahat, mhat