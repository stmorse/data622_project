import pickle
import time
import numpy as np
from MHP import MHP

DATA_PATH = '../data/data_ts/'

def main():
    t0 = time.time()

    print(f'Loading data ... {time.time()-t0:.3f}')

    with open(f'{DATA_PATH}token_dicts_mhp.pkl', 'rb') as f:
        token_dicts = pickle.load(f)

    dim = len(set(token_dicts['token_dict'].keys()))

    with open(f'{DATA_PATH}data_mhp.pkl', 'rb') as f:
        tokenized = pickle.load(f)

    print(f'Computing prior ... {time.time()-t0:.3f}')

    # t_ij = total counts where token j precedes token i
    tmx = np.ones((dim, dim))

    # s_ij = total counts where token j *directly* precedes token i
    smx = np.ones((dim, dim))

    for story in tokenized['train']:
        story = np.array(story)
        sequ = story[:,1].astype(int)
        N = sequ.shape[0]

        # indicator matrix s_ij = 1 if event_i is type j
        S = np.zeros((N, dim), dtype=np.float32)
        S[np.arange(N), sequ] = 1

        # compute tmx for this story
        p_ones = np.ones((N, N))
        p_ones[np.triu_indices(N)] = 0
        tmx += (S.T @ p_ones) @ S  

        # compute smx for this story
        pairs = zip(sequ[:-1], sequ[1:])
        for j, i in pairs:
            smx[i, j] += 1

    print(f'Training model ... {time.time()-t0:.3f}')

    # initialize parameters
    mshape = (dim, dim)
    Ahat = np.zeros(mshape)
    num_entries = int(np.prod(mshape) * 0.1)
    random_indices = np.unravel_index(
        np.random.choice(Ahat.size, num_entries, replace=False), mshape
    )
    Ahat[random_indices] = np.random.uniform(0.0001, 0.001, size=num_entries)

    mhat = np.random.uniform(0.1, 0.5, dim)
    omega = 0.9

    mhp = MHP(alpha=Ahat, mu=mhat, omega=omega)

    for k, story in enumerate(tokenized['train']):
        verbose = False
        if k % 10 == 0: 
            verbose = True
            print(k)

        _, _ = mhp.EM(seq=np.array(story), maxiter=30, 
                    regularize=True, smx=smx, tmx=tmx,
                    verbose=verbose)
        
    print(f'Save model parameters ... {time.time()-t0:.3f}')

    with open(f'mhp_params.pkl', 'wb') as f:
        pickle.dump({
            'tmx': tmx, 'smx': smx,
            'alpha': mhp.alpha, 'mu': mhp.mu, 'omega': mhp.omega
        }, f)

    print('Complete. {time.time()-t0:.3f}')

if __name__=="__main__":
    main()