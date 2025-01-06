import matplotlib.pyplot as plt
import numpy as np
import quimb.tensor as qtn
import time
import thewalrus as tw
import script as gALS
import torch
import os
import json
import pandas as pd
import copy
import warnings

###########################################################################
N = 5 #Number of modes
D = 10 #Basis cutoff
kappa = 10.0 #Strength of the non-Gaussian gate.
seeds = [0] #seeds for the experiments.

bond_dims = [2,2,4,4,8,8,10,16,32,64,100] #Bond dimensions for the DMRG sweeps.

assert bond_dims[6] == D #Hacky way to ensure that the bond dimensions match. (See: "dmrg.solve(verbosity=True,max_sweeps=7, tol = 0)")
############################################################################

if not os.path.exists("GeneratedData"):
    os.makedirs("GeneratedData")

def gauss_overlap(cov1,mu1,cov2,mu2):
    '''
    Some prefactors and so on have been left out. This is simply used as a heuristic to select "easy" GBS problems.
    '''
    cov1_inv = np.linalg.inv(cov1)
    cov2_inv = np.linalg.inv(cov2)
    mu = mu1 - mu2
    if len(mu.shape) == 1:
        mu = mu.reshape(-1,1)
    assert cov1.shape == cov2.shape, "Covariance matrices must have the same shape."
    assert mu1.shape == mu2.shape, "Means must have the same shape."
    assert cov1.shape[0] == mu.shape[0], "Covariance matrix and mean vector must have the same dimensionality."

    exponent = - 0.5 * mu.T@cov1_inv@np.linalg.inv(cov1_inv + cov2_inv)@cov2_inv@mu
    denominator = np.sqrt(np.linalg.det(cov1_inv + cov2_inv)*np.linalg.det(cov1)*np.linalg.det(cov2))
    return np.exp(exponent)/denominator

def gen_cov_mat():
    from tqdm import tqdm
    dist_to_Id = lambda cov: gauss_overlap(cov,np.zeros(2*N),np.eye(2*N),np.zeros(2*N))*2**N #Dist should be "overlap"
    best_cov = tw.random.random_covariance(N,pure=True)
    best_dist = dist_to_Id(best_cov)
    print("trying to find a good covariance matrix")
    for n in tqdm(range(40000)):
        new_cov = tw.random.random_covariance(N,pure=True)
        dist = dist_to_Id(new_cov)
        if dist > best_dist:
            best_cov = new_cov
            best_dist = dist
    return best_cov, best_dist

def gen_S(cov):
    from scipy.linalg import block_diag
    from strawberryfields.decompositions import williamson
    Ss = []
    Svs = []
    N = cov.shape[0]//2
    for i in range(N):
        sub_cov = cov[[i,i+N],:][:,[i,i+N]]
        db, S = williamson(sub_cov)
        Ss.append(S)
        Svs.append(db[0,0])
    return block_diag(*Ss)[[i for i in range(0,2*N,2)]+[i for i in range(1,2*N,2)],:][:,[i for i in range(0,2*N,2)]+[i for i in range(1,2*N,2)]]#, Svs

def get_MPO_from_trainer(trainer):
    trainer._initialize_MPOs()
    MPO = trainer.MPOs[0] * trainer.prefactors[0]
    for i in range(1,len(trainer.MPOs)):
        MPO += trainer.MPOs[i] * trainer.prefactors[i]
    MPO = copy.copy(MPO)
    MPO.apply_to_arrays(lambda x: x.detach().cpu().numpy())
    return MPO

def get_GBSMPO_from_trainer(trainer):
    trainer._initialize_MPOs()
    MPO = trainer.MPOs[0] * trainer.prefactors[0]
    MPO = copy.copy(MPO)
    MPO.apply_to_arrays(lambda x: x.detach().cpu().numpy())
    return MPO

def get_energy_stats(MPS,MPO):
    exp = (MPS.H.reindex({"k"+str(i):"b"+str(i) for i in range(N)})|MPO|MPS).contract(all, optimize='auto-hq')
    expsq = (MPS.H.reindex({"k"+str(i):"bi"+str(i) for i in range(N)})|MPO.reindex({"b"+str(i):"bi"+str(i) for i in range(N)}).reindex({"k"+str(i):"b"+str(i) for i in range(N)})|MPO|MPS).contract(all, optimize='auto-hq')
    return exp.real, np.sqrt(expsq.real - exp.real**2)

def save_SVs(MPS,filename):
    out = np.zeros((N,D))
    for i in range(N):
        MPS.canonize(i)
        S = np.linalg.svd(MPS[i].data.reshape(-1,D),compute_uv=False)
        out[i] = np.append(S, np.array([0 for _ in range(D-len(S))]))
    np.save(os.path.join(filename),out)

def plot_losses(losses,N,D,seed,kappa):
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_yscale("log")

    # Create inset axes
    axins = ax.inset_axes([0.45, 0.45, 0.5, 0.5])  # [left, bottom, width, height]

    # Plot the zoomed-in data
    axins.plot(losses)
    axins.hlines(losses[-1], 0, len(losses), alpha=0.7, color='r', ls="--",zorder=-1, label="ADAM")
    axins.set_yscale("log")

    # Set the limits of the inset axes
    ulim = np.quantile(losses,0.4)
    axins.set_ylim(min(losses)*(1-1e-3),ulim)
    axins.set_xlim(np.argmin(np.abs(np.array(losses)-ulim)),len(losses)*1.01)
    ax.indicate_inset_zoom(axins)
    ax.set_title(f"Minimum loss: {min(losses)}")
    fig.savefig(f"GeneratedData/LossCurve_N{N}_D{D}_k{kappa}_seed{seed}.png")

Es = []
Evars = []
Ekappas = []
Ekappavars = []
EGBSs = []
EGBSvars = []

SWs = []
SWkappas = []
SWGBSs = []

runtimes = []
kappa_runtimes = []
GBS_runtimes = []

cov_dists = []
cov_traces = []

current_seeds = []

nonLBO_energies = dict()
pLBO_energies = dict()
GBS_energies = dict()

pdict = {
    "D": D,
    "N": N,
    "dtype": torch.cfloat,
    "basis": "local",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "max bond dim": D,
}
pdict["kappa"] = kappa

for seed in seeds:
    gALS._print("Starting experiment with seed",seed)
    pdict["seed"] = seed
    current_seeds.append(seed)

    np.random.seed(seed)
    cov, dist = gen_cov_mat()
    cov_dists.append(dist[0,0])
    cov_traces.append(np.trace(cov))
    np.save(f"GeneratedData/cov_N{N}_seed{seed}.npy",cov)
    
    pdict["covariance matrix"] = cov
    trainer = gALS.TrainingModule(pdict)

    MPO_nonLBO = get_MPO_from_trainer(trainer)
    
    dmrg = qtn.DMRG1(MPO_nonLBO, bond_dims=bond_dims)
    dmrg.opts["local_eig_backend"] = "LOBPCG"
    start = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore')
        dmrg.solve(verbosity=True,max_sweeps=7, tol = 0)
    kappa_runtime = time.time() - start

    tmpstate = copy.deepcopy(dmrg.state)
    tmpstate.apply_to_arrays(lambda x: torch.tensor(x, dtype=pdict["dtype"]))
    gALS._print("Temporary state created for trainer")
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore')
        dmrg.solve(verbosity=True,max_sweeps=15)
    runtimes.append(time.time()-start)

    #################### saving non LBO stuff ####################
    gALS._print("Saving non LBO stuff")
    try:
        nonLBO_energies[f"seed{seed}"] = [float(E.real) for E in dmrg.energies]
        with open(f"GeneratedData/nonLBO_N{N}_D{D}_chi{bond_dims[-1]}_k{kappa}_seeds_{'_'.join(str(seed) for seed in seeds)}.json",'w') as fp:
            json.dump(nonLBO_energies, fp) #change to get the dmrg_energies as real instead of complex64
    except Exception as error:
        print("Could not save dmrg energies due to error:",error)
        print(nonLBO_energies)

    MPS_nonLBO = dmrg.state
    try:
        E, Evar = get_energy_stats(MPS_nonLBO,MPO_nonLBO)
        Es.append(E)
        Evars.append(Evar)
    except Exception as error:
        print("Could not get energy stats for MPS due to error:",error)
        Es.append(dmrg.energy)
        Evars.append(0)
    save_SVs(MPS_nonLBO,f"GeneratedData/SVs_nonLBO_N{N}_D{D}_chi{bond_dims[-1]}_k{kappa}_seed{seed}.npy")
    SWs.append((MPS_nonLBO.H@MPS_nonLBO).real)
    ###############################################################
    gALS._print("optimizing basis parameters")
    trainer.lr = 1
    trainer.losses.append(np.inf)
    trainer.MPS = tmpstate
    kappa_runtime -= time.time()
    trainer.learn_run(verbose=True)
    kappa_runtime += time.time()

    #################### saving basis parameter history ####################
    gALS._print("Saving basis parameters and creating figure of loss curves")
    plot_losses(trainer.losses,N,D,seed,kappa)
    try:
        tmp = {key: tensor for key, tensor in trainer.bp_history.items()}
        tmp['alpha_real'] = np.real(tmp['alpha'])
        tmp['alpha_imag'] = np.imag(tmp['alpha'])
        del tmp['alpha']
        tmp = {key: array.tolist() for key, array in tmp.items()}
        with open(os.path.join('GeneratedData',f'basisparams_N{N}_D{D}_chi{D}_k{kappa}_seed{seed}.json'),'w') as fp:
            json.dump(tmp, fp)
    except Exception as error:
        print("Could not save basis parameters due to error:",error)
        print(tmp)
    ########################################################################
    gALS._print("Running DMRG in the optimized basis")

    MPO_pLBO = get_MPO_from_trainer(trainer)
    p0 = trainer.MPS
    p0.apply_to_arrays(lambda x: x.detach().cpu().numpy())

    dmrgpLBO = qtn.DMRG1(MPO_pLBO, bond_dims=bond_dims[6:],p0=p0)
    dmrgpLBO.opts["local_eig_backend"] = "LOBPCG"
    kappa_runtime -= time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore')
        dmrgpLBO.solve(verbosity=True,max_sweeps=7+15)
    kappa_runtime += time.time()
    kappa_runtimes.append(kappa_runtime)

    #################### savingp pLBO stuff ########################
    gALS._print("Saving pLBO stuff")
    try:
        pLBO_energies[f"seed{seed}"] = [float(E.real) for E in dmrgpLBO.energies]
        with open(f"GeneratedData/pLBO_N{N}_D{D}_chi{bond_dims[-1]}_k{kappa}_seeds_{'_'.join(str(seed) for seed in seeds)}.json",'w') as fp:
            json.dump(pLBO_energies, fp) #change to get the dmrg_energies as real instead of complex64
    except Exception as error:
        print("Could not save dmrg energies due to error:",error)
        print(pLBO_energies)

    MPS_pLBO = dmrgpLBO.state
    try:
        E, Evar = get_energy_stats(MPS_pLBO,MPO_pLBO)
        Ekappas.append(E)
        Ekappavars.append(Evar)
    except Exception as error:
        print("Could not get energy stats for MPS due to error:",error)
        Ekappas.append(dmrgpLBO.energy)
        Ekappavars.append(0)
    save_SVs(MPS_pLBO,f"GeneratedData/SVs_pLBO_N{N}_D{D}_chi{bond_dims[-1]}_k{kappa}_seed{seed}.npy")
    SWkappas.append((MPS_pLBO.H@MPS_pLBO).real)
    ###############################################################

    #################### Solving GBS #########################
    print("Running and saving GBS stuff")
    S = gen_S(cov)
    pdict["Hamiltonian"] = torch.tensor(np.linalg.inv(np.linalg.inv(S)@cov@np.linalg.inv(S.T))/2,dtype=torch.cfloat,device=pdict["device"])
    tmp_trainer = gALS.TrainingModule(pdict)
    MPO_GBS = get_GBSMPO_from_trainer(tmp_trainer)
    dmrg_GBS = qtn.DMRG1(MPO_GBS, bond_dims=bond_dims)
    dmrg_GBS.opts["local_eig_backend"] = "LOBPCG"
    start = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore')
        dmrg_GBS.solve(verbosity=True,max_sweeps=15)
    GBS_runtimes.append(time.time()-start)
    #################### saving GBS stuff ####################
    try:
        GBS_energies[f"seed{seed}"] = [float(E.real) for E in dmrg_GBS.energies]
        with open(f"GeneratedData/GBS_N{N}_D{D}_chi{bond_dims[-1]}_seeds_{'_'.join(str(seed) for seed in seeds)}.json",'w') as fp:
            json.dump(GBS_energies, fp) #change to get the dmrg_energies as real instead of complex64
    except Exception as error:
        print("Could not save dmrg energies due to error:",error)
        print(GBS_energies)

    MPS_GBS = dmrg_GBS.state
    try:
        E, Evar = get_energy_stats(MPS_GBS,MPO_GBS)
        EGBSs.append(E)
        EGBSvars.append(Evar)
    except Exception as error:
        print("Could not get energy stats for MPS due to error:",error)
        EGBSs.append(dmrg_GBS.energy)
        EGBSvars.append(0)
    save_SVs(MPS_GBS,f"GeneratedData/SVs_GBS_N{N}_D{D}_chi{bond_dims[-1]}_seed{seed}.npy")
    SWGBSs.append((MPS_GBS.H@MPS_GBS).real)
    ###############################################################
    gALS._print("Saving data...")

    data = {
            "Es":Es,
            "Evars":Evars,
            "Ekappas":Ekappas,
            "Ekappavars":Ekappavars,
            "EGBSs":EGBSs,
            "EGBSvars":EGBSvars,
            "SWs":SWs,
            "SWkappas":SWkappas,
            "SWGBSs":SWGBSs,
            "runtimes":runtimes,
            "kappa_runtimes":kappa_runtimes,
            "GBS_runtimes":GBS_runtimes,
            "current_seed":current_seeds,
            "cov_dists":cov_dists,
            "cov_traces":cov_traces,}
    pd.DataFrame(data).to_csv(f"GeneratedData/data_N{N}_D{D}_chi{bond_dims[-1]}_k{kappa}_seeds_{'_'.join(str(seed) for seed in seeds)}.csv")

gALS._print("Done!")