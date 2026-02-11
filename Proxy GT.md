## INVERSE PROBLEM: ML Model Trajectory Matching with HR Simulation

### **Given :**
  1. An ML model trained to predict coarse-resolution turbulence dynamics
  2. A high-resolution differentiable (HR) physics simulator
  
### **Goal:**
  For each autoregressive step, find an HR initial condition that, when evolved through K steps of HR physics simulator, matches the hypothetical HR equivalent of the LR model prediction at that autoregressive step.
  
### **Methodology:**
1. Generate a coarse trajectory using the ML model (forward pass)
2. Optimize the HR initial condition using gradient descent:
   - Simulate the HR state forward
   - Coarsen the result
   - Compare to ML model's prediction
   - Backpropagate through the entire simulation to update the initial condition

### **Hyperparameters:**
1. The number K of HR simulator steps to match the ML emulator state can be set to 2 for Kolmogorov Flow. It may be relevant to look at the effect of the number of simulation steps on the [Validation](#validation) metrics.
2. The number of training iterations was set to 1000. It would be relevant to explore how much increasing or reducing it decreases the accuracy of the HR equivalent. In particular, later steps 
3. Suppose we want to estimate the HR equivalent of the ML prediction $\hat{x}_t$ at timestep $t$. The initialization for the optimization of the initial condition is set to the true HR state $x_t$.

### **Validation:**
1. The pure accuracy between the coarsened proxy HR and the model's prediction is a good metric for training, but not so much for evaluation. Indeed, we are more interested to check if the obtained proxy state matches the "true" hypothetical high-resolution equivalent of the LR model prediction. 
2. As the time $t$ increases, it may become harder and harder to match the model's prediction $\hat{x}_t$ since there may not be an HR equivalent of the model's prediction if it drifts outside of physical distribution. Since the usage of the proxy emulator (see ) depends on how well it matches the true state, this may be of concern.

### **Usage of the proxy emulator:**
1. Given the model prediction $\hat{x}_t$, the HR proxy is written as $\tilde{x}_t$. We wish to look at the evolution of $ \lVert \mathcal{M}(\hat{x}_t) - GT(\tilde{x}_t) \rVert $. An increase in this distance indicates the presence of exposure-bias.
2. Look at the evolution of $\lVert \mathcal{M}(\hat{x}_t) - x_{t+1} \rVert - \lVert \mathcal{M}(\hat{x}_t) - GT(\tilde{x}_t) \rVert $. This quantity is the trajectory drift, which is the error that has accumulated until step $t$ and which cannot be reduced even when using the true simulator.
3. One could also think about "divergence spectrum" (or some sort of bifurcation map), where we call the ground-truth simulation multiple times on a given HR proxy, and look at divergence we get from the true trajectory (at a given point). This could tell us about the "irreducible error" behaviour. 
4. For different models (with let's say fixed one-step error), can we obtain a response of the trajectory drift w.r.t the shape of the error (for instance, the frequencies)

### **Resolutions:** 
1. HR Simulation (256x256)
    ↓ (coarsen by 4x)
2. Coarse Resolution (64x64) ← ML Model operates here
    ↑ (what we optimize to match)
3. Proxy HR Simulation (256x256) 

### **Potential related litterature:**
1. Dynamical systems