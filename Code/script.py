import torch
import quimb.tensor as qtn
import numpy as np
import tqdm
import warnings
from scipy import stats
import datetime


def _print(statement,verbose=True):
    '''
    Simple helper function that prints the current time and a statement, with the option to turn off the print statement.
    '''
    if verbose:
        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} - {statement}")

class SingleSiteMinimizer(torch.nn.Module):
    '''
    Class for minimizing the energy of a single site in the MPS.

    - MPS: The Matrix Product State core to be optimized.
    - MPOs: The Matrix Product Operator cores that are contracted with the MPS.
    - basis_params: The parameters for the basis transformation, also optimizeable.
    - envs: The environments for constructing the effective hamiltonians.
    - MPO_cores: The functions for constructing the MPO cores based on the X and P matrices.
    - prefactors: The prefactors for each constituent in the Hamiltonian.
    - learnable_params: A dictionary with booleans for the parameters that should be optimized.
    - _get_XandP: Inherited function for constructing the X and P matrices based on the basis parameters.

    The forward function calculates the variational energy for the instantaneous values of the basis parameters. This is then minimized using an optimizer in the _local_update method below.
    '''
    def __init__(self, MPS, MPOs, basis_params, envs, MPO_cores, prefactors, learnable_params, _get_XandP):
        super().__init__()
        self.envs = envs
        self.MPO_cores = MPO_cores
        self.prefactors = prefactors
        self._get_XandP = _get_XandP
        
        self.mps_params = torch.nn.Parameter(MPS.tensors[0].data) # The MPS core is made tuneable.
        
        MPS.apply_to_arrays(lambda x: self.mps_params)
        self.MPS = MPS
        self.MPOs = MPOs

        self.basis_params = torch.nn.ParameterDict({key: torch.nn.Parameter(basis_params[key],requires_grad=learnable_params[key]) for key in basis_params.keys()}) # The relevant basis parameters are made tuneable.
        self._create_reindex() # This is a hacky way to reindex the hermitian conjugate of the MPS in order to contract the Hermitian form.

    def forward(self):
        MPS = self.norm_fn(self.MPS)
        out = 0
        X, P = self._get_XandP(self.basis_params)
        for MPO, MPO_core, prefactor, env, reindex in zip(self.MPOs, self.MPO_cores, self.prefactors, self.envs, self.reindeces):
            MPO.apply_to_arrays(lambda x: MPO_core(X,P))
            out += prefactor * (MPS.H.reindex(reindex)|MPO|MPS|env).contract(..., optimize='auto-hq')
        return out

    def norm_fn(self,psi):
        '''
        Method for normalizing the MPS.
        '''
        nfact = (psi.H @ psi)**0.5
        return psi.multiply(1 / nfact, spread_over='all')

    def _create_reindex(self):
        '''
        Method for creating the reindexing dictionary for the hermitian conjugate of the MPS.
        '''
        ###### ALL of this functions is very hacky ~~~ should consider re-implementing. ######
        mps = self.MPS
        reindeces = []
        for MPO, env in zip(self.MPOs, self.envs):
            tmp = MPO|mps|env
            
            site_ID = mps.site_tags_present[0]
            if len(tmp.outer_inds()) == 2:
                if site_ID == "I0":
                    new_ind = (set(tmp.select("I1").outer_inds()).intersection(set(tmp.outer_inds()))).pop()
                else:
                    new_ind = (set(tmp.select("I0").outer_inds()).intersection(set(tmp.outer_inds()))).pop()
                old_ind = list(mps.outer_inds())
                old_ind.remove("k"+site_ID[1:])
                reindex = {old_ind[0]:new_ind,"k"+site_ID[1:]:"b"+site_ID[1:]}
            else:
                new_left_ind = (set(tmp.select("I0").outer_inds()).intersection(set(tmp.outer_inds()))).pop()
                new_right_ind = (set(tmp.outer_inds()).difference({new_left_ind,"b"+site_ID[1:]})).pop()
                old_inds = mps.outer_inds()
                reindex = {old_inds[1]:new_left_ind,old_inds[2]:new_right_ind,"k"+site_ID[1:]:"b"+site_ID[1:]} #This is so cursedly hacked... yikes.
            reindeces.append(reindex) #Seemingly only 1 reindex is necessary since they all share the same indexing.
        self.reindeces = reindeces

class TrainingModule():
    '''
    Creates a training module object based on a parameterdict.
    The parameterdict should contain the following keys:
    - device ['cpu','cuda']: Whether the calculations are performed on the CPU or GPU.
    - D [int]: The physical dimension / cutoff of the modes.
    - covariance matrix [np.array]: The covariance matrix of the Gaussian state.
    - kappa [float]: The strength parameter of the global CZ gate (specific to this example).
    - max bond dim / bond dimensions [int/list]: The maximum bond dimension of the MPS or a list of bond dimensions for each site.

    Optional keys:
    - dtype [torch.dtype]: The datatype of the calculations. Default is torch.cfloat which is the fastest for autograd.
    - lr [float]: The learning rate of the optimizer.
    - tol [float]: The tolerance for the optimizer to terminate.
    - max iter [int]: The maximum number of iterations for the optimizer.
    - basis [str]: The type of basis transformation. Either 'local' or 'standard', i.e. whether to optimize the basis or not.
    - basis_params [dict]: The initial values of the basis parameters.
    - learnable_params [dict]: A dictionary with booleans for the parameters that should be optimized.
    - jit [bool]: Whether to use JIT compilation for the optimizer. Makes the subsequent computations faster at the cost of an initial overhead. It is usually recommended to have this on.
    - svd cutoff [float]: The cutoff for the SVD truncation of the MPS cores. If None, no truncation (based on the singular values) is performed.
    - seed [int]: A seed for the random initialization of the MPS.
    '''
    def __init__(self,parameterdict):
        self.parameterdict = parameterdict
        self.dtype = parameterdict["dtype"] if "dtype" in parameterdict.keys() else torch.cfloat
        self.device = parameterdict["device"]
        self.N = parameterdict["covariance matrix"].shape[0] // 2
        assert self.N%2 == 1, "Number of modes must be odd, otherwise simply implement the middle most core of the MPO for the even case. (..or contact the author and ask him to do it.)"
        self.D = parameterdict["D"]
        self.lr = parameterdict["lr"] if "lr" in parameterdict.keys() else 0.001
        self.tol = parameterdict["tol"] if "tol" in parameterdict.keys() else 1e-5
        self.max_iter = parameterdict["max iter"] if "max iter" in parameterdict.keys() else 5000
        self.basis = parameterdict["basis"] if "basis" in parameterdict.keys() else "local"
        self.jit = parameterdict["jit"] if "jit" in parameterdict.keys() else True
        self.svd_cutoff = parameterdict["svd cutoff"] if "svd cutoff" in parameterdict.keys() else None
        self.kappa = parameterdict["kappa"]

        self.losses = [np.inf]

        try:
            self.basis_params = parameterdict["basis_params"]
            _print("Initialized basis parameters with given values")
        except:
            self.basis_params = {'alpha': torch.zeros(self.N,dtype=torch.cfloat,device=self.device),
                                 'r': torch.zeros(self.N,dtype=torch.float,device=self.device),
                                 'phi': torch.zeros(self.N,dtype=torch.float,device=self.device),
                                 'theta': torch.zeros(self.N,dtype=torch.float,device=self.device),
                                 's': torch.zeros(self.N,dtype=torch.float,device=self.device),
                                 'gamma': torch.zeros(self.N,dtype=torch.float,device=self.device),
                                 'kappa': torch.zeros(self.N,dtype=torch.float,device=self.device)}
        try:
            self.learnable_params = parameterdict["learnable_params"]
        except:
            if self.basis == "local":
                self.learnable_params = {key: True for key in self.basis_params.keys()}
            elif self.basis == "standard":
                self.learnable_params = {key: False for key in self.basis_params.keys()}
        _print(f"Learnable parameters are: {list(key for key in self.learnable_params.keys() if self.learnable_params[key])}")

        #SAVE BASIS_PARAMS
        self.bp_history = {key: item.cpu().numpy() for key,item in self.basis_params.items()}

        self._initialize_coefficients()
        # self._initialize_MPS()
        self._PrepareMatrixTensors()
        # self._initialize_MPOs()

        try:
            self.max_bond_dim = max(self.parameterdict["bond dimensions"])
        except:
            self.max_bond_dim = self.parameterdict["max bond dim"]

    def _PrepareMatrixTensors(self):
        '''
        Stores the necessary matrices that are later used in '_get_XandP'.
        '''
        self._extra = 4 #4 is enough for this example. It depends on the power of the X and P matrices in the Hamiltonian and the parameterization of the local basis.
        self._a = torch.diag(torch.tensor([np.sqrt(i) for i in range(1,self.D + self._extra)],dtype=self.dtype,device=self.device),1)
        self._Id = torch.eye(self.D + self._extra,dtype=self.dtype,device=self.device)
        self._n = torch.tensor([i for i in range(0,self.D + self._extra)],dtype=self.dtype,device=self.device)
        self._nsqrd = self._n**2
    
    def _get_XandP(self, basis_params):
        """Returns the X and P matrices for a given set of parameters, with increased dimension such that later truncation is possible in order to mitigate loss of commutation relations."""

        theta_coshr = torch.exp(1j*basis_params["theta"]) * torch.cosh(basis_params["r"])
        phitheta_sinhr = torch.exp(1j*basis_params["phi"] - 1j*basis_params["theta"]) * torch.sinh(basis_params["r"])
        tau = 0.5j*basis_params["s"] * (theta_coshr + phitheta_sinhr)

        nkerrn = torch.diag(2*self._n + 1)
        akerr = self._a @ torch.diag(torch.exp(1j*basis_params["kappa"]*(2*self._n - 1)))
        aakerr = self._a @ self._a @ torch.diag(torch.exp(4j*basis_params["kappa"]*(self._n - 1)))        
        
        a_out = akerr * (theta_coshr + tau) + akerr.conj().T * ( - phitheta_sinhr + tau) + 0.5j * basis_params["gamma"] * (theta_coshr + phitheta_sinhr) * (nkerrn + aakerr + aakerr.conj().T) + basis_params["alpha"] * self._Id
        a_out_dagger = a_out.conj().T
        X_out = (a_out + a_out_dagger)/np.sqrt(2)
        P_out = 1j * (a_out_dagger - a_out)/np.sqrt(2)
        return X_out, P_out

    def _initialize_MPS(self):
        '''
        Random initialization of the MPS. That it is random seems to be critical for the optimization, but it may also be intialized from e.g. dmrg, simply be overwriting the self.MPS attribute.
        Two possible ways to initialize the MPS:
         - Specify a "max bond dim" (integer) such that the MPS will have bond dimensions D, D^2, D^3, ..., chi_max, chi_max, ..., D^2, D.
         - Specfiy a list of integers "bond dimensions" of length N - 1 which will be the bond-dimensions of the MPS.
        '''
        np.random.seed(self.parameterdict["seed"]) if "seed" in self.parameterdict.keys() else None
        scale = 1/(1*self.D) #This scaling parameter is necessary for the brute force normalization to work properly.
                             #If the random values are either too small or too large, the inner product yields nan.
        def randC(*shape):
            return np.random.uniform(-scale, scale, shape) + 1.j * np.random.uniform(-scale, scale, shape)

        def MPS_from_bond_dimensions(dims): #Takes a list and yields an MPS with the given bond dimensions.
            assert len(dims) == self.N-1
            cores = [randC(dims[0],self.D)]
            for i in range(1,self.N-1):
                cores.append(randC(dims[i-1],dims[i],self.D))
            cores.append(randC(dims[-1],self.D))
            return qtn.MatrixProductState(*[cores])
        
        try:
            tn = MPS_from_bond_dimensions(self.parameterdict["bond dimensions"])
            self.max_bond_dim = max(self.parameterdict["bond dimensions"])
            statement = "Seeded the random MPS with the given bond dimensions"
        except:
            bond_dimensions = [self.D**i if i < np.log2(self.parameterdict["max bond dim"])/np.log2(self.D) else self.parameterdict["max bond dim"] for i in range(1,self.N//2+1)]
            bond_dimensions += bond_dimensions[:self.N//2-(self.N%2 == 0)][::-1]
            tn = MPS_from_bond_dimensions(bond_dimensions)
            self.max_bond_dim = self.parameterdict["max bond dim"]
            statement = "Seeded the random MPS with the given max bond dimension"
        
        tn.apply_to_arrays(lambda x: torch.tensor(x, dtype=self.dtype))
        tn = self.norm_fn(tn)
        tn = self.norm_fn(tn)
        assert not tn[0].data[0,0].isnan().item(), "Normalization failed, check the scaling parameter in _initialize_MPS"
        self.MPS = tn
        _print("Initial MPS prepared: "+statement)
    
    def _initialize_MPOs(self):
        '''
        Constructs the MPOs of the Hamiltonian.
        There are multiple because we, in this example, consider a block diagonal MPO for the total Hamiltonian.
        '''
        basis_params = self.basis_params
        N = self.N
        MPOs = []
        constructors = [self.construct_MPO_cores(n) for n in range(N)]
        for const in range(len(constructors[0])):
            cores = []
            for i in range(N):
                X, P = self._get_XandP({key: self.basis_params[key][i] for key in self.basis_params.keys()})
                cores.append(constructors[i][const](X,P))
            MPO = qtn.MatrixProductOperator(cores)
            MPO = MPO.reindex({"k"+str(i):"bt"+str(i) for i in range(N)})
            MPO = MPO.reindex({"b"+str(i):"k"+str(i) for i in range(N)})
            MPO = MPO.reindex({"bt"+str(i):"b"+str(i) for i in range(N)})
            MPOs.append(MPO)
        self.MPOs = MPOs

    def _initialize_coefficients(self):
        '''
        A simple method that initializes the coefficients for the Hamiltonian.
        '''
        cov = self.parameterdict["covariance matrix"]
        cov /= np.power(np.linalg.det(cov),1/2*self.N) #Account for varying value of hbar (might as well set it to 2, because then the determinant equals 1.)
        H = torch.tensor(np.linalg.inv(cov)/2,dtype=self.dtype,device=self.device)
        self.XXcoeffs = H[:self.N,:self.N].real
        self.PPcoeffs = H[self.N:,self.N:].real
        self.XPcoeffs = 2 * H[:self.N,self.N:].real

        self.prefactors = [1, self.kappa, self.kappa**2, self.kappa, self.kappa**2]
        

    def learn_run(self,verbose=True,callable=None):
        '''
        Helper method for a typical round of optimization, consisting of multiple sweeps at different learning rates and tolerances.
        'callable' is a function that is called after each sweep, and could e.g. be something that saves a figure based on the attribute 'losses'.
        '''
        self._local_update = self._local_update_ADAM
        self.tol = 1e-6
        self.max_iter = 5000
        for _ in range(10):
            print("\nAt step: ",_) if verbose else print("At step: ",_)
            minloss = self.losses[-1]
            self.sweep(verbose=verbose)
            if minloss > self.losses[-1] > 0.9999*minloss or self.losses[-1] < 7e-6:
                break
            if self.losses[-1] <2e-4:
                self.lr = 1e-3
                self.tol = 1e-5
            if callable:
                callable()


    def sweep(self,verbose=False):
        """
        Sweeps left to right in a DMRG like fashion, optimizing the MPS and the basis parameters simultaneously. 
        There is probably plenty of room for optimization here, but at least it works.
        """
        MPS = self.MPS #Assumed to be normalized
        _print("Right orthogonalizing MPS",verbose=verbose)
        MPS.canonize(0) #Speedup can be made here by implementing right orthogonalization with cuda compatibility. However, this is not the bottleneck.
        
        #Construct the MPOs
        if self.basis == "local":
            self._initialize_MPOs()
        MPOs = self.MPOs

        #Put them on CPU in order to save space on GPU for gradients
        for MPO in MPOs:
            MPO.apply_to_arrays(lambda x: x.cpu())

        #Gather constituents for MPS
        cores = [MPS.select("I"+str(self.N-1))]
        for n in range(self.N-2,0,-1):
            cores.append(MPS.select("I"+str(n)))
        cores.append(MPS.select("I0"))
        cores.reverse()

        #Now for Rayleigh quotients
        PRODS = [] #Contains the right environments for each site for all MPOs. (i.e. a list of lists.)
        RAYLEIGH = [] #Contains the full Rayleigh quotients for each MPO.
        LEFT = [] #Contains the left environments for the current site for all MPOs.
        MPSH = MPS.H.reindex({"k"+str(i):"b"+str(i) for i in range(self.N)})
        for MPO in MPOs:
            Rayleigh = MPSH|MPO|MPS
            tmp = Rayleigh.select("I"+str(self.N-1))^...
            prods = [tmp]
            for n in range(self.N-2,0,-1):
                prods.append(Rayleigh.select("I"+str(n))@prods[-1])
            prods.reverse()
            PRODS.append(prods)
            LEFT.append(Rayleigh.select("I0"))
            RAYLEIGH.append(Rayleigh)

        #Now that environments have been created, put it back on the device and begin optimization from site 0.
        for MPO, prods in zip(MPOs,PRODS):
            MPO.apply_to_arrays(lambda x: x.to(self.device))
            prods[0].apply_to_arrays(lambda x: x.to(self.device))
        cores[0].apply_to_arrays(lambda x: x.to(self.device))

        #Collect basis parameters and construct the SingleSiteMinimizer object.
        BasisParams_site = {key: self.basis_params[key][0] for key in self.basis_params.keys()}
        SS = SingleSiteMinimizer(cores[0],[MPO.select("I0") for MPO in MPOs],BasisParams_site,[prods[0] for prods in PRODS],self.construct_MPO_cores(0),self.prefactors,self.learnable_params,self._get_XandP)
        _print("Updating site 0",verbose=verbose)
        self._local_update(SS,verbose=verbose) #Optimization of the first site.
        cores[0].apply_to_arrays(lambda x: SS.MPS[0].data.detach()) #Once it terminates, the MPS core is updated.
        for key in self.basis_params.keys():
            self.basis_params[key][0] = SS.basis_params.get_parameter(key).item() #The basis parameters are updated.
        cores[1].apply_to_arrays(lambda x: x.to(self.device)) #Move the next core to the device.
        self._move_orthog_center_right(cores[0],cores[1],self.svd_cutoff,self.max_bond_dim) #Move the orthogonality center to the right.

        #Construct left-most environments on device.
        site_constructors = self.construct_MPO_cores(0)
        for i in range(len(RAYLEIGH)):
            left = RAYLEIGH[i].select("I0")
            left.tensors[0].apply_to_arrays(lambda x: cores[0].H[0].data)
            left.tensors[1].apply_to_arrays(lambda x: site_constructors[i](*self._get_XandP(BasisParams_site))) #Notice that 'BasisParams_site' is used, however, that is because it conatins the optimized basis parameters, since these were changed inplace in the SS object.
            left.tensors[2].apply_to_arrays(lambda x: cores[0][0].data)
            LEFT[i] = left^...
        #Move unecessary stuff to CPU
        cores[0].apply_to_arrays(lambda x: x.cpu())
        for prods in PRODS:
            prods[0].apply_to_arrays(lambda x: x.cpu()) #(could probably delete these if needed?)

        #Now do the same for each site.
        for n in range(1,self.N-1):
            for prods in PRODS:
                prods[n].apply_to_arrays(lambda x: x.to(self.device))
            BasisParams_site = {key: self.basis_params[key][n] for key in self.basis_params.keys()}
            SS = SingleSiteMinimizer(cores[n],[MPO.select("I"+str(n)) for MPO in MPOs],BasisParams_site,[left|prods[n] for left, prods in zip(LEFT,PRODS)],self.construct_MPO_cores(n),self.prefactors,self.learnable_params,self._get_XandP)
            _print("Updating site "+str(n),verbose=verbose)
            if n == self.N//2:
                ## Often the middlemost MPO lacks a sparsity that is present in the rest of the MPO cores.
                ## Therefore, the JIT compilation can take an usually long time,
                ## and it is thus more time efficient to simply do the calculations with the non-jitted version.
                self._local_update(SS,verbose=verbose,jit=False)
            else:
                self._local_update(SS,verbose=verbose)
            cores[n].apply_to_arrays(lambda x: SS.MPS[n].data.detach())
            for key in self.basis_params.keys():
                self.basis_params[key][n] = SS.basis_params.get_parameter(key).item()
            cores[n+1].apply_to_arrays(lambda x: x.to(self.device))
            self._move_orthog_center_right(cores[n],cores[n+1],self.svd_cutoff,self.max_bond_dim)
            site_constructors = self.construct_MPO_cores(n)
            for i in range(len(LEFT)):
                left_tmp = RAYLEIGH[i].select("I"+str(n))
                left_tmp.tensors[0].apply_to_arrays(lambda x: cores[n].H[n].data)
                left_tmp.tensors[1].apply_to_arrays(lambda x: site_constructors[i](*self._get_XandP(BasisParams_site)))
                left_tmp.tensors[2].apply_to_arrays(lambda x: cores[n][n].data)
                LEFT[i] = left_tmp@LEFT[i]
            cores[n].apply_to_arrays(lambda x: x.cpu())
            for prods in PRODS:
                prods[n].apply_to_arrays(lambda x: x.cpu())
        
        #Finally, optimizing the last site.
        BasisParams_site = {key: self.basis_params[key][-1] for key in self.basis_params.keys()}
        SS = SingleSiteMinimizer(cores[-1],[MPO.select("I"+str(self.N-1)) for MPO in MPOs],BasisParams_site,[left for left in LEFT],self.construct_MPO_cores(self.N-1),self.prefactors,self.learnable_params,self._get_XandP)
        _print("Updating site "+str(self.N-1),verbose=verbose)
        self._local_update(SS,verbose=verbose)
        cores[-1].apply_to_arrays(lambda x: SS.MPS[self.N-1].data.detach())
        for key in self.basis_params.keys():
            self.basis_params[key][-1] = SS.basis_params.get_parameter(key).item()
        cores[-1].apply_to_arrays(lambda x: x.cpu())
        self.MPS = self.norm_fn(MPS)#.reindex({"b"+str(i):"k"+str(i) for i in range(self.N)})

        #SAVE BASIS_PARAMS
        self.bp_history = {key: np.vstack((self.bp_history[key],self.basis_params[key].cpu().numpy())) for key in self.basis_params.keys()}

    def _local_update_ADAM(self,SSnjit,verbose=False,jit=True):
        '''
        Optimization with a SGD optimizer. Is used to optimize the MPS core and the local basis parameters.
        '''
        if self.jit & jit:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action='ignore',
                    message='.*trace might not generalize.*',
                )
                SS = torch.jit.trace_module(SSnjit, {"forward": []}) #JIT compilation for speedup
        else:
            SS = SSnjit
        if verbose:
            pbar = tqdm.tqdm(range(self.max_iter))
        else:
            pbar = range(self.max_iter)
        optimizer = torch.optim.Adam(SS.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.get_lambda_lr(sum(n.numel() for n in SS.parameters())))
        for _ in pbar:
            optimizer.zero_grad()
            loss = SS().real
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss = loss.item()
            if loss < -1e-5:
                break #Loss should not be negative
            if _ == 0:
                break_even_thresh = loss #This threshold is to ensure that the optimizer can only terminate once the loss is lower than what it started out with.
            self.losses.append(loss)
            if verbose:
                pbar.set_description(f"{loss}")
            if _ > 50 and loss < break_even_thresh:
                ## This is a simple convergence criterion, which can be tuned manually.
                ## It is based on the slope of the loss function over the last 5 iterations.
                ## Once it is flat enough (compared to some tol) and negative, the optimizer terminates.
                res = stats.linregress(np.arange(len(self.losses[-6:-1])),self.losses[-6:-1])
                if np.abs(res.slope/res.intercept) < self.tol and np.sign(res.slope) == -1:
                    break
        SS.mps_params.data /= (SSnjit.MPS.H@SSnjit.MPS).real.item()**0.5 #once the optimizer terminates, the MPS is normalized.
    #####################################################################################################
    def _local_update_BFGS(self,SSnjit,verbose=False,jit=True):
        '''
        Optimization with a BFGS optimizer. See "_local_update_ADAM" for more details.
        This optimizer contains a line search and is good for initial optimization.
        However, it seems to terminate at a worse minimum than SGD.
        Therefore, the optimal strategy is either to use BFGS for initial optimization and then switch to SGD,
        or to use DMRG to get a good initial guess and then use SGD.
        '''
        if self.jit & jit:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action='ignore',
                    message='.*trace might not generalize.*',
                )
                SS = torch.jit.trace_module(SSnjit, {"forward": []})
        else:
            SS = SSnjit
        if verbose:
            pbar = tqdm.tqdm(range(self.max_iter))
        else:
            pbar = range(self.max_iter)
        optimizer = torch.optim.LBFGS(SS.parameters(),
                                      lr = self.lr,
                                      # history_size=10, 
                                      max_iter=self.LBFGS_max_iter,
                                      tolerance_change = self.tolerance_change,
                                      tolerance_grad = self.tolerance_grad,
                                      line_search_fn="strong_wolfe")
        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            loss = SS().real
            if loss.requires_grad:
                loss.backward()
            # self.losses.append(loss.item())
            return loss
        for _ in pbar:
            loss = optimizer.step(closure)
            if _ == 0:
                break_even_thresh = loss
            self.losses.append(loss.item())
            if verbose:
                pbar.set_description(f"{loss}")
            if _ > 5 and loss <= break_even_thresh:
                res = stats.linregress(np.arange(len(self.losses[-6:-1])),self.losses[-6:-1])
                if np.abs(res.slope/res.intercept) < self.tol and (np.sign(res.slope) == -1 or np.sign(res.slope) == 0):
                    break
        SS.mps_params.data /= (SSnjit.MPS.H@SSnjit.MPS).real.item()**0.5
    
    def get_lambda_lr(self,d_model,warmup_steps=6):
        '''
        Returns a scheduler for the optimizer.
        '''
        d = d_model**(-1)
        w = warmup_steps**(-3/2)
        def scheduler(epoch):
            epoch += 1
            return d * min(epoch**(-0.1),epoch*w)
        return scheduler
    
    def norm_fn(self,psi):
        '''
        Method for normalizing the MPS.
        '''
        nfact = (psi.H @ psi)**0.5
        return psi.multiply(1 / nfact, spread_over='all')

    def _move_orthog_center_right(self,prev,next,cutoff,max_bond):
        '''
        This moves the center of orthogonality from one site to the next in an implicitly left-to-right fashion.
        The use of this method could perhaps be replaced by native functions in quimb.. But this is helpful for when the cores are stored in a list.
        '''
        prev = prev.tensors[0]
        next = next.tensors[0]
        prevshape = prev.shape #Both are as [left,right,physical] unless site=0 [right,physical] or site=N [left,physical]
        nextshape = next.shape

        if len(prevshape) == 2: #Ie. on the left most site
            prevdata = prev.data.movedim(0,1)
        else:
            prevdata = prev.data.movedim(1,2).reshape(prevshape[0]*prevshape[2],prevshape[1])
            
        U,S,Vadj = torch.linalg.svd(prevdata,
                                full_matrices=False)
        if cutoff:
            N_cut = min((S/S[0] > cutoff).sum(),max_bond)
        else:
            N_cut = min(nextshape[0],max_bond)
        prevdata = U[:,:N_cut]
        if len(prevshape) == 2:
            # prevdata = prevdata.view(N_cut,prevshape[0]).movedim(0,1) # At some point fix: "RuntimeError: shape '[6, 6]' is invalid for input of size 48"
            prevdata = prevdata.movedim(0,1)
        else: 
            prevdata = prevdata.view(prevshape[0],prevshape[2],N_cut).movedim(1,2)
        if len(nextshape) == 2:
            nextdata = (S[:N_cut].cfloat().unsqueeze(1)*Vadj[:N_cut])@next.data
        else:
            nextdata = ((S[:N_cut].cfloat().unsqueeze(1)*Vadj[:N_cut])@next.data.view(nextshape[0],nextshape[1]*nextshape[2])).view(N_cut,nextshape[1],nextshape[2])
        prev.apply_to_arrays(lambda x: prevdata)
        next.apply_to_arrays(lambda x: nextdata)

    def construct_MPO_cores(self, n):
        '''
        One of the key methods that returns the function that builds the MPO cores at site n for given X and P matrices.
        '''
        D = self.D
        N = self.N
        Id = self._Id[:D,:D]

        a = 2 * self.XXcoeffs
        b = 2 * self.PPcoeffs
        _b = self.PPcoeffs
        c = self.XPcoeffs #equal to gamma (in the arxiv version regarding MPOs) even though the factor two is not introduced here. (It is further up..)

        if n == 0:

            def MPO_core_gauss(X,P):
                SingleSite = (a[n,n]/2 * X@X + b[n,n]/2 * P@P + c[n,n]/2 * ( P@X + X@P ) - 0.5*self._Id)[:D,:D]
                S = torch.zeros((4,D,D),dtype=self.dtype,device=self.device)
                for i, mat in enumerate([X[:D,:D],P[:D,:D],SingleSite,Id]):
                    S[i] = mat
                return S
            
            def MPO_core_PX(X,P):
                S = torch.zeros((2,D,D),dtype=self.dtype,device=self.device)
                S[0] = b[n,n] * P[:D,:D]
                S[1] = X[:D,:D]
                return S
            
            def MPO_core_X2(X,P):
                S = torch.zeros((2,D,D),dtype=self.dtype,device=self.device)
                S[0] = _b[n,n] * Id
                S[1] = (X@X)[:D,:D]
                return S
            
            def MPO_core_Xes(X,P):
                S = torch.zeros((4,D,D),dtype=self.dtype,device=self.device)
                XPPX = 2 * (X@P)[:D,:D] -1j * Id
                for i, mat in enumerate([Id,X[:D,:D],(X@X)[:D,:D],XPPX]):
                    S[i] = mat
                return S
            
            def MPO_core_X2es(X,P):
                S = torch.zeros((2,D,D),dtype=self.dtype,device=self.device)
                S[0] = (X@X)[:D,:D]
                S[1] = X[:D,:D]
                return S

        elif n == N-1:

            def MPO_core_gauss(X,P):
                SingleSite = (a[n,n]/2 * X@X + b[n,n]/2 * P@P + c[n,n]/2 * ( P@X + X@P ) - 0.5*self._Id)[:D,:D]
                S = torch.zeros((4,D,D),dtype=self.dtype,device=self.device)
                for i, mat in enumerate([X[:D,:D],P[:D,:D],Id,SingleSite]):
                    S[i] = mat
                return S
            
            def MPO_core_PX(X,P):
                S = torch.zeros((2,D,D),dtype=self.dtype,device=self.device)
                S[0] = X[:D,:D]
                S[1] = b[n,n] * P[:D,:D]
                return S

            def MPO_core_X2(X,P):
                S = torch.zeros((2,D,D),dtype=self.dtype,device=self.device)
                S[0] = (X@X)[:D,:D]
                S[1] = _b[n,n] * Id
                return S
            
            def MPO_core_Xes(X,P):
                S = torch.zeros((4,D,D),dtype=self.dtype,device=self.device)
                XPPX = 2 * (X@P)[:D,:D] -1j * Id
                for i, mat in enumerate([Id,X[:D,:D],(X@X)[:D,:D],XPPX]):
                    S[i] = mat
                return S
            
            def MPO_core_X2es(X,P):
                S = torch.zeros((2,D,D),dtype=self.dtype,device=self.device)
                S[0] = (X@X)[:D,:D]
                S[1] = X[:D,:D]
                return S

        elif n < N//2:
            
            def MPO_core_gauss(X,P):
                SingleSite = (a[n,n]/2 * X@X + b[n,n]/2 * P@P + c[n,n]/2 * ( P@X + X@P ) - 0.5*self._Id)[:D,:D]
                X = X[:D,:D]
                P = P[:D,:D]

                S = torch.zeros((2*(n+1),2*(n+1)+2,D,D),dtype=self.dtype,device=self.device)
                diag_inds = tuple(i for i in range(2*n)) + (-2,-1)
                S[diag_inds,diag_inds] = Id
                S[-1,2*n] = X
                S[-1,2*n+1] = P
                S[-1,2*n+2] = SingleSite
                for n_left in range(n):
                    S[2*n_left,2*n+2] = a[n_left,n]*X + c[n_left,n]*P
                    S[2*n_left+1,2*n+2] = c[n,n_left]*X + b[n_left,n]*P
                return S
            
            def MPO_core_PX(X,P):
                S = torch.zeros((2,2,D,D),dtype=self.dtype,device=self.device)
                S[[0,1],[0,1]] = X[:D,:D]
                S[1,0] = b[n,n] * P[:D,:D]
                return S

            def MPO_core_X2(X,P):
                S = torch.zeros((2,2,D,D),dtype=self.dtype,device=self.device)
                S[[0,1],[0,1]] = (X@X)[:D,:D]
                S[1,0] = _b[n,n] * Id
                return S
            
            shape_correction = int(n == 1)
            def MPO_core_Xes(X,P):
                S = torch.zeros((3*n +2 - shape_correction,3*n+5,D,D),dtype=self.dtype,device=self.device)
                X2 = (X@X)[:D,:D]
                XPPX = 2 * (X@P)[:D,:D] -1j * Id
                X = X[:D,:D]
                ind = n
                column_inds = tuple(i for i in range(ind)) + tuple(i for i in range(ind+1,S.shape[0]))
                row_inds = tuple(i for i in range(ind)) + tuple(i for i in range(ind+4,S.shape[1]-shape_correction))
                S[column_inds, row_inds] = X

                for i in range(ind):
                    S[i,-1] = c[n,i] * X2 + _b[n,i] * XPPX

                    back_ind = -2 -2*i + shape_correction
                    S[back_ind,-1] = _b[i,n] * Id
                    S[back_ind-1,-1] = c[i,n] * Id

                S[ind,ind] = Id
                S[ind,ind+1] = X
                S[ind,ind+2] = X2
                S[ind,ind+3] = XPPX
                return S
            
            def MPO_core_X2es(X,P):
                S = torch.zeros((2+n-shape_correction,3+n,D,D),dtype=self.dtype,device=self.device)
                X2 = (X@X)[:D,:D]
                X = X[:D,:D]
                row_inds = [0] + list(i for i in range(2,S.shape[1]-shape_correction))
                column_inds = list(i for i in range(S.shape[0]))
                S[column_inds,row_inds] = X2
                S[0,1] = X
                if shape_correction:
                    S[-1,-1] = b[n,n-1] * X
                else:
                    S[1:-1,-1] = b[n,:n,None,None].flip(0) * X
                return S
            
        elif n == N//2:
            if N%2 == 0:
                raise ValueError("N is even, implement the middle core of the MPO")
                Sshape0 = N+2
                Sshape1 = N
            else:
                Sshape0 = N+1
                Sshape1 = N+1
            
            def MPO_core_gauss(X,P):
                SingleSite = (a[n,n]/2 * X@X + b[n,n]/2 * P@P + c[n,n]/2 * ( P@X + X@P ) - 0.5*self._Id)[:D,:D]
                X = X[:D,:D]
                P = P[:D,:D]
                S = torch.zeros((Sshape0,Sshape1,D,D),dtype=self.dtype,device=self.device)
                for i in range(Sshape0//2 - 1):
                    for j in range(Sshape1//2 - 1):
                        j_tilde = N-1-j
                        S[2*i,2*j] = a[i,j_tilde]*Id
                        S[2*i,2*j+1] = c[i,j_tilde]*Id
                        S[2*i+1,2*j] = c[j_tilde,i]*Id
                        S[2*i+1,2*j+1] = b[i,j_tilde]*Id
                    S[2*i,-2] = a[i,N//2]*X + c[i,N//2]*P
                    S[2*i+1,-2] = c[N//2,i]*X + b[i,N//2]*P
                for j in range(Sshape1//2 - 1):
                    j_tilde = N-1-j
                    S[-1,2*j] = a[N//2,j_tilde]*X + c[j_tilde,N//2]*P
                    S[-1,2*j+1] = c[N//2,j_tilde]*X + b[N//2,j_tilde]*P
                S[(-2,-1),(-2,-1)] = Id
                S[-1,-2] = SingleSite
                return S
            
            def MPO_core_PX(X,P):
                S = torch.zeros((2,2,D,D),dtype=self.dtype,device=self.device)
                S[[0,1],[0,1]] = X[:D,:D]
                S[1,0] = b[n,n] * P[:D,:D]
                return S

            def MPO_core_X2(X,P):
                S = torch.zeros((2,2,D,D),dtype=self.dtype,device=self.device)
                S[[0,1],[0,1]] = (X@X)[:D,:D]
                S[1,0] = _b[n,n] * Id
                return S
            
            trace_gamma = torch.trace(c)
            def MPO_core_Xes(X,P):
                S = torch.zeros((3*(N-1)//2+2,3*(N-1)//2+2,D,D),dtype=self.dtype,device=self.device)
                X2 = (X@X)[:D,:D]
                XPPX = 2 * (X@P)[:D,:D] -1j * Id
                X = X[:D,:D]

                S[n+1:-2:2,0:n] = c[:n,n+1:,None,None].flip(0,1) * X
                S[n+2:-1:2,0:n] = _b[:n,n+1:,None,None].flip(0,1) * X

                S[0:n,n+1:-2:2] = c[n+1:,0:n].T[:,:,None,None] * X
                S[0:n,n+2:-1:2] = _b[n+1:,0:n].T[:,:,None,None] * X

                S[n,0:n] = c[n,n+1:,None,None].flip(0) * X2 + _b[n,n+1:,None,None].flip(0) * XPPX
                S[0:n,n] = c[n,0:n,None,None] * X2 + _b[n,0:n,None,None] * XPPX

                S[[n,-1],[-1,n]] = X
                S[n,n+1:-2:2] = c[n+1:,n,None,None] * Id
                S[n,n+2:-1:2] = _b[n+1:,n,None,None] * Id
                S[n+1:-2:2,n] = c[:n,n,None,None].flip(0) * Id
                S[n+2:-1:2,n] = _b[:n,n,None,None].flip(0) * Id

                S[n,n] = trace_gamma * X
                return S
            
            def MPO_core_X2es(X,P):
                S = torch.zeros((n+2,n+2,D,D),dtype=self.dtype,device=self.device)
                X2 = (X@X)[:D,:D]
                X = X[:D,:D]
                S[[0,-1],[-1,0]] = X2
                S[0,1:-1] = b[n+1:,n,None,None] * X
                S[1:-1,0] = b[n,:n,None,None].flip(0) * X
                S[1:-1,1:-1] = b[n+1:,:n].flip(1).T[:,:,None,None] * X2
                return S
        
        elif n > N//2:
            
            def MPO_core_gauss(X,P):
                SingleSite = (a[n,n]/2 * X@X + b[n,n]/2 * P@P + c[n,n]/2 * ( P@X + X@P ) - 0.5*self._Id)[:D,:D]
                X = X[:D,:D]
                P = P[:D,:D]
                S = torch.zeros((2*(N+1-n),2*(N-n),D,D),dtype=self.dtype,device=self.device)
                diag_inds = tuple(i for i in range(2*(N-n-1))) + (-2,-1)
                S[diag_inds,diag_inds] = Id
                S[-4,-2] = X
                S[-3,-2] = P
                S[-1,-2] = SingleSite
                for n_right in range(N-n-1):
                    n_right_tilde = N - 1 - n_right
                    S[-1,2*n_right] = a[n,n_right_tilde]*X + c[n_right_tilde,n]*P
                    S[-1,2*n_right+1] = c[n,n_right_tilde]*X + b[n,n_right_tilde]*P
                return S

            def MPO_core_PX(X,P):
                S = torch.zeros((2,2,D,D),dtype=self.dtype,device=self.device)
                S[[0,1],[0,1]] = X[:D,:D]
                S[1,0] = b[n,n] * P[:D,:D]
                return S

            def MPO_core_X2(X,P):
                S = torch.zeros((2,2,D,D),dtype=self.dtype,device=self.device)
                S[[0,1],[0,1]] = (X@X)[:D,:D]
                S[1,0] = _b[n,n] * Id #* P[:D,:D]
                return S
            
            shape_correction = int(n == N-2)
            def MPO_core_Xes(X,P):
                S = torch.zeros((3*(N-n)+2,3*(N-n)-1 - shape_correction,D,D),dtype=self.dtype,device=self.device)
                X2 = (X@X)[:D,:D]
                XPPX = 2 * (X@P)[:D,:D] -1j * Id #X@P + P@X
                X = X[:D,:D]
                ind = N - n-1
                row_inds = tuple(i for i in range(ind)) + tuple(i for i in range(ind+4,3*(N-n)+2- shape_correction))
                column_inds = tuple(i for i in range(ind)) + tuple(i for i in range(ind+1,3*(N-n)-1- shape_correction))
                S[row_inds, column_inds] = X

                for i in range(ind):
                    S[-1,i] = c[n,-1 -i]*X2 + _b[n,-1 -i]*XPPX

                    back_ind = -2 -2*i + shape_correction
                    S[-1,back_ind] = _b[-1 -i,n] * Id
                    S[-1,back_ind-1] = c[-1-i,n] * Id

                S[ind,ind] = Id
                S[ind+1,ind] = X
                S[ind+2,ind] = X2
                S[ind+3,ind] = XPPX
                return S
            
            def MPO_core_X2es(X,P):
                S = torch.zeros((2+N-n,1+N-n-shape_correction,D,D),dtype=self.dtype,device=self.device)
                X2 = (X@X)[:D,:D]
                X = X[:D,:D]
                row_inds = [0] + list(i for i in range(2,S.shape[0]-shape_correction))
                column_inds = list(i for i in range(S.shape[1]))
                S[row_inds,column_inds] = X2
                S[1,0] = X
                if shape_correction:
                    S[-1,-1] = b[n+1,n] * X
                else:
                    S[-1,1:-1] = b[n+1:,n,None,None] * X
                return S

        return [MPO_core_gauss, MPO_core_PX, MPO_core_X2, MPO_core_Xes, MPO_core_X2es]