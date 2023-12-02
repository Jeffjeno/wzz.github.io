---
title: 'Diffusion Models in Time Series Applications'
date: 2023-08-11
permalink: /posts/2023/08/Diff_in_Time/
tags:
  - Diffusion Model
  - Time Series Forecasting
  - Time Series Generating
---

A survey for Diffusion Models in Time Series Applications.

Diffusion Models in Time Series Applications
======

## Time Series Forecasting

### Problem Formulation

Given historical multivariate time series $X_c^0=\{x_1^0, x_2^0, ..., x_{t_0-1}^0\}$, the goal is to predict future time series $X_p^0=\{x_{t_0}^0, x_{t_0+1}^0, ..., x_T^0\}$. 

Learn the conditional probability distribution: 
$$
q(X_p^0|X_c^0) = \prod_{t=t_0}^T q(x_t^0|X_c^0)
$$


### Models

#### TimeGrad: Based on DDPM, uses RNN to encode historical information.

TimeGrad is based on **denoising diffusion probabilistic models (DDPMs)**. It models the conditional distribution 
$$
p(x_t|h_{t-1})
$$
where $h_{t-1}$ encodes historical information.

<img src="https://cdn.jsdelivr.net/gh/Imbernoulli/mdimages@main/%E6%88%AA%E5%B1%8F2023-08-22%2013.09.43.png" alt="截屏2023-08-22 13.09.43" style="zoom: 40%;" />  <img src="https://cdn.jsdelivr.net/gh/Imbernoulli/mdimages@main/%E6%88%AA%E5%B1%8F2023-08-22%2013.10.28.png" alt="截屏2023-08-22 13.10.28" style="zoom: 25%;" />

TimeGrad is an autoregressive model for multivariate time series forecasting based on denoising diffusion probabilistic models (DDPMs). 

Given a multivariate time series $X_0 = \{x_0^1, x_0^2, ..., x_0^T\}$ where $x_0^t \in \mathbb{R}^D$, the goal is to predict the future time series $X_0^p = \{x_0^{t_0}, x_0^{t_0+1}, ..., x_0^T\}$ based on the historical context window $X_0^c = \{x_0^1, x_0^2, ..., x_0^{t_0-1}\}$.

The model approximates the conditional distribution as:

$$
q(X_0^{t_0:T}|X_0^{1:t_0-1}) = \prod_{t=t_0}^T p_\theta(x_0^t|h_{t-1})
$$

where 
$$
h_t = \text{RNN}_\theta(concat(x_t^0, c_t), h_{t-1})
$$
 encodes historical information.

**3. Training:**

The training objective is a variational lower bound:

$$
\min_\theta \, \mathbb{E}_{q(x^{0:N})}[-\log p_\theta(x^{0:N}) + \log q(x^{1:N}|x^0)]
$$

Minimize the negative log-likelihood, approximate upper bound:

$$
\mathbb{E}_{k,x_t^0,\epsilon}[\delta(k) \parallel \epsilon - \epsilon_\theta(\sqrt{\tilde{\alpha}_k}x_t^0 + \sqrt{1-\tilde{\alpha}_k}\epsilon, h_{t-1}, k) \parallel^2]
$$

**4. Generation:**

Sample $x_t$ step-by-step and update hidden state $h_t$.

Sampling is done via Langevin dynamics:

$$
x^{n-1}_t = \frac{1}{\sqrt{\alpha_n}}(x_t^n - \frac{\beta_n}{\sqrt{1-\bar{\alpha}_n}}\epsilon_\theta(x_t^n, h_{t-1}, n)) + \sqrt{\Sigma_\theta} z
$$

where $z \sim \mathcal{N}(0, I)$. The sampling process starts from Gaussian noise and iteratively denoises using the learned reverse process. The mean prediction is corrected using the noise prediction network $ε_θ$. Adding Gaussian noise z simulates one step of the reverse Markov chain.

#### ScoreGrad: Based on SDE, also uses RNN.

ScoreGrad is based on **continuous energy-based generative models**. It models the conditional distribution 
$$
p(x_t|h_{t-1})
$$
where $h_{t-1}$ encodes historical information.

![截屏2023-08-22 10.02.49](https://cdn.jsdelivr.net/gh/Imbernoulli/mdimages@main/截屏2023-08-22 10.02.49.png)

**Forward Process**

- Continuous diffusion process defined by SDE:

$$
dx_t = f(x_t, k)dk + g(k)dw
$$

**Reverse Process**

- Reverse-time SDE:

$$
dx_t = [f(x_t, k) - \frac{1}{2}g(k)^2\nabla_{x_t}\log q_k(x_t|h_t)]dk + g(k)d\tilde{w}
$$

- Approximates 
$$
\nabla_{x_t}\log q_k(x_t|h_t)
$$
 with neural network $s_\theta(x_t, h_t, k)$

**Historical Information**

- Encoded via RNN or other sequential modeling methods

**Training Objective**

Minimizes continuous form of denoising score matching objective:

$$
L(\theta) = \frac{1}{T}\sum_{t=1}^T L_t(\theta)
$$

$$
L_t(\theta) =\arg\min_{\theta}\mathbb{E}_{t_s}[\lambda(t_s)\mathbb{E}_{x_t^0,x_{t_s}^t}[||s_\theta(x_{t_s}^t, h_t, t_s) - \nabla_{x_{t_s}^t} \log p_{t_s}^0(x_{t_s}^t|x_t^0)||_2^2]]
$$

<img src="https://cdn.jsdelivr.net/gh/Imbernoulli/mdimages@main/%E6%88%AA%E5%B1%8F2023-08-22%2010.05.31.png" alt="截屏2023-08-22 10.05.31" style="zoom:50%;" />  <img src="https://cdn.jsdelivr.net/gh/Imbernoulli/mdimages@main/%E6%88%AA%E5%B1%8F2023-08-22%2010.05.06.png" alt="截屏2023-08-22 10.05.06" style="zoom:50%;" /> 

**Generation**

- Samples from reverse-time SDE

- Flexible due to continuous energy-based formulation

- Achieved state-of-the-art on multivariate time series data

The key is continuous diffusion for flexibility and encoding history to capture temporal dependencies.

#### D^3^VAE: Bidirectional VAE + Denoising Score Matching.

D3VAE is based on a **bidirectional variational autoencoder (BVAE)** and integrates diffusion, denoising, and disentanglement.

![截屏2023-08-22 10.30.49](https://cdn.jsdelivr.net/gh/Imbernoulli/mdimages@main/%E6%88%AA%E5%B1%8F2023-08-22%2010.30.49.png)

**Diffusion**

- Uses a coupled diffusion process to augment input and target time series
- Reduces aleatoric uncertainty
- Improves generalization for short time series

![截屏2023-08-22 10.31.26](https://cdn.jsdelivr.net/gh/Imbernoulli/mdimages@main/%E6%88%AA%E5%B1%8F2023-08-22%2010.31.26.png)

**Forward process**

$$
X^{(t)} = \sqrt{\bar{\alpha}_t} X^{(0)} + (1-\bar{\alpha}_t)\delta_X
$$

$$
Y^{(t)} = \sqrt{\bar{\alpha}'_t}Y^{(0)} + (1-\bar{\alpha}'_t)\delta_Y
$$

**Coupled diffusion process:**

- For CONTEXT window:

$$
x_{1:t_0-1}^k = \sqrt{\tilde{\alpha}_k}x_{1:t_0-1}^0 + \sqrt{1-\tilde{\alpha}_k}\epsilon
$$

- For PREDICTION window: 

$$
x_{t_0:T}^k = \sqrt{\tilde{\alpha}_k'}x_{t_0:T}^0 + \sqrt{1-\tilde{\alpha}_k'}\epsilon
$$

Where $\tilde{\alpha}_k' > \tilde{\alpha}_k$, increase noise level for PREDICTION window.

**Bidirectional VAE:** 

Learn 
$$
p_\phi(Z|x_{1:t_0-1}^0)
$$
 to map CONTEXT to latent space.  

Decode latent representations to predict disturbed PREDICTION window $\hat{x}_{t_0:T}^k$.

**Denoising score matching:**

- Employs denoising score matching (DSM) to clean diffused time series

- Removes noise and improves accuracy. Single-step gradient jump to remove noise:

  $$
  \hat{x}_{t_0:T}^0 \leftarrow \hat{x}_{t_0:T}^k - \sigma_0^2\nabla_{\hat{x}_k}E(\hat{x}_k;e)
  $$

**Training objective:**

Minimize KL divergence, denoising score matching loss, total correlation, and MSE loss.

- Objective:

$$
L = \psi \cdot D_{KL}(q(Y^{(t)}) | p_\theta(\hat{Y}^{(t)})) + \lambda \cdot L(ζ,t) + \gamma \cdot L_{TC} + L_{mse}(\hat{Y}^{(t)}, Y^{(t)})
$$

**Generation:** 

- Sample from the learned $p_\phi$ and $p_\theta$. Samples

$$
Z \sim p_\phi(Z|X)
$$

and generates 

$$
\hat{Y} \sim p_\theta(\hat{Y}|Z)
$$

- Achieves state-of-the-art results on time series forecasting

#### DSPD/CSPD: Model time series as values of a continuous function, with continuous Gaussian noise.

#### ![截屏2023-08-22 10.42.08](https://cdn.jsdelivr.net/gh/Imbernoulli/mdimages@main/%E6%88%AA%E5%B1%8F2023-08-22%2010.42.08.png)

1. Forward process: 
$$
q(X^k|X^0) = \mathcal{N}(\sqrt{\tilde{\alpha}_k}X^0, (1-\tilde{\alpha}_k)\Sigma)
$$

2. Reverse process: 
$$
p_\theta(X^{k-1}|X^k) = \mathcal{N}(\mu_\theta(X^k, k), (1-\alpha_k)\Sigma)
$$
3. Training: Minimize squared error.
4. Generation: Sample reverse process step-by-step.

<img src="https://cdn.jsdelivr.net/gh/Imbernoulli/mdimages@main/%E6%88%AA%E5%B1%8F2023-08-22%2010.44.53.png" alt="截屏2023-08-22 10.44.53" style="zoom: 50%;" />


##### Discrete Stochastic Process Diffusion (DSPD)

DSPD is a generative modeling technique for time series data based on denoising diffusion probabilistic models. It treats time series as discretized measurements from an underlying continuous function.

In DSPD, noise is gradually added to the time series data $X_0$ across $N$ discrete steps to reach a final noisy output $X_N$. This is done by adding noise from a stochastic process rather than independent Gaussian noise at each step. This preserves the continuity and correlations of the time series when adding noise.

Two options for the stochastic process noise are:

- Gaussian process: Sample noise $ɛ(t)$ from a GP prior with covariance matrix $Σ(t)$. This produces smooth, correlated noise functions.

- Ornstein-Uhlenbeck (OU) process: Sample noise $ɛ(t)$ from an OU process, which is a stationary, mean-reverting stochastic process.

The forward diffusion process is defined as:

$$
q(X_n | X_{n-1}) = N(\sqrt{1 - β_n}X_{n-1}, β_nΣ)
$$

$$
q(X_n | X_0) = N(\sqrt{α_n} X_0, (1 - α_n)Σ)
$$

where $α_n = 1 - β_n$ and $Σ(t)$ is the covariance matrix.

The reverse generative model learns to invert this diffusion process by predicting the noise $ɛ$ that was added to the original $X_0$. This is parameterized as:

$$
p(X_{n-1} | X_n) = N(μ_θ(X_n, t, n), β_nΣ)
$$

where $μ_θ$ is a neural network predicting the noise. The loss function is the MSE between predicted and true noise.

After training, new samples are generated by:

1. Sample initial $X_N$ from the stochastic process prior.
2. Iteratively denoise $X_N \rightarrow \ldots \rightarrow X_0$ using the learned model.

##### Continuous Stochastic Process Diffusion (CSPD)

CSPD follows the same overall procedure as DSPD but uses a continuous diffusion process defined by an SDE.

The forward SDE process is:

$$
dX_s = f(X_s, s)ds + g(s)dW_s
$$

where $W_s$ is a Wiener process and $s$ is continuous time. This gradually adds noise to $X_0$ to reach a noisy $X_S$.

The reverse process involves learning the score function $\nabla_{X_s} \log p(X_s)$ to sample from $p(X_0)$ by reversing the SDE. The score function is predicted by a neural network and optimized with MSE loss.

After training, new samples are generated by:

1. Sample noisy $X_S$ from the SDE forward process.
2. Reverse the SDE using the predicted score to denoise back to $X_0$.

In summary, both DSPD and CSPD add stochastic process noise to preserve continuity, then train a model to invert the diffusion process and generate new samples. The key difference is DSPD uses discrete steps while CSPD is continuous.

## Time Series Generation

### Problem Formulation

Given a time series $X^0=\{x_1^0, x_2^0, ..., x_T^0\}$, the goal is to generate synthetic samples $x_{1:T}^0$ that resemble the real ones.

Learn the joint probability distribution: $p(x_{1:T}^0)$. 

### Model

#### TSGM: Encode to latent space, sample with conditional score matching network, then decode.

TSGM is a generative model for time-series data and is composed of three networks: an encoder, a decoder, and a conditional score network. The overall process involves embedding time-series data into a latent space, training a conditional score network, and generating fake time-series samples.

#### ![截屏2023-08-22 11.08.59](https://cdn.jsdelivr.net/gh/Imbernoulli/mdimages@main/%E6%88%AA%E5%B1%8F2023-08-22%2011.08.59.png)

**Encoder and Decoder:** 

$$
h_t = En(h_{t-1}, x_t)
$$

$$
\hat{x}_t = De(h_t)
$$
The **encoder** and **decoder** are responsible for mapping time-series data to and from a latent space. In this context, let's denote the data space as $\mathcal{X}$ and the latent space as $\mathcal{H}$. The encoder function $e$ maps data from $\mathcal{X}$ to $\mathcal{H}$, while the decoder function $d$ performs the reverse mapping. The encoding and decoding process is defined recursively as follows:

Here, 
$$
\mathbf{x}_{1:T}
$$
represents a time-series sample of length $T$, 
$$
\mathbf{x}_t
$$
is an observation at time $t$ in the sequence, and 
$$
\mathbf{h}_{1:T}
$$
are the corresponding embedded vectors. The encoder and decoder are implemented using recurrent neural networks (RNNs).

**Conditional Score Network**

The **conditional score network** is used to generate conditional log likelihoods for time-series observations. Unlike independent samples in other generative tasks, time-series data has dependencies between observations. To account for this, the score network is designed to learn the conditional log likelihood based on previous observations. The authors modify the U-net architecture for this purpose, adapting it from 2D to 1D convolutions to handle time-series data. The score network takes into account both the diffused data and the conditioning information to generate fake time-series samples.

**Conditional score matching:**

Learn $s_\theta(h_k, h_{t-1}, k)$, minimize objective:
$$
L_{Score} = \mathbb{E}_{h_0, k}[\sum_{t=1}^T L(t, k)]
$$
Where: 
$$
L(t, k) = \mathbb{E}_{h_k}[||\delta(k) s_\theta(h_k, h_{t-1}, k) - \nabla_{h_t}\log q_k^0(h_t|h_0)||^2]
$$

**Training Objective Functions**

The training process involves two main objective functions:

1. **Encoder and Decoder Training ($L_{ed}$)**: The encoder and decoder are trained using the Mean Squared Error (MSE) loss between the input time-series data and its reconstructed counterpart by the encoder-decoder process. The loss is defined as:
$$
L_{ed} = E_{x_{1:T}}[||\hat{x}_{1:T} - x_{1:T}||_2^2]
$$

2. **Conditional Score Network Training ($L_{score}$)**: This is a central contribution of the method. At each time step $t$ within the sequence, the method uses the conditional score network to learn the gradient of the conditional log likelihood. This is achieved by diffusing the data through a stochastic process and comparing the generated samples with the gradients of the log likelihood. The loss function is defined as:
$$
L_{score} = E_{s}[E_{x_{1:T}}[∑_{t=1}^{T} λ(s) l_3(t, s)]]
$$

**Generation:** 

Solve reverse-time SDE, sample latent states, then decode to time series.

1. **ncoder and Decoder Initialization**: Start with initialized encoder and decoder networks, trained to map real time-series data to a latent space and back.

2. **Conditional Score Network Setup**: Employ the trained conditional score network, designed to generate data considering temporal dependencies.

3. **Generation Loop**: Repeat for each time step:

   a. **Diffusion and Conditioning**: Diffuse observed data through a stochastic process to get diffused samples.

   b. **Score Network Inference**: Calculate the gradient of the conditional log likelihood using the score network.

   c. **Sample Generation**: Adjust the diffused sample using the gradient, considering previous observations.

   d. **Temporal Update**: Incorporate the generated sample into observations for the next step.

4. **Completion**: Generate a synthetic time-series sequence with preserved temporal patterns and dependencies.

The combination of these components ensures the generated data captures the characteristics of real-world time-series data, making it valuable for various applications.

