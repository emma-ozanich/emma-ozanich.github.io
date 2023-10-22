---
title: "Deep clustering of reef soundscape"
categories:
  - ocean acoustics
  - bioacoustics
  - convolutional autoencoder
header:
  teaser: assets/images/Figure14.png
classes: wide
---

A deep embedded clustering (DEC) algorithm was developed to separate fish calls and whale song in coral reef ambient noise. We applied image processing to spectrograms (time-frequency) of the sounds. 


<p align="center"><img src="/assets/images/reefnoise_long.png" alt="drawing" width="100%"/></p>


The DEC uses a convolutional auto-encoder (CAE) to obtain low-dimensional kernels from the image. Then, a similarity loss is applied to the kernels (clustering layer), the CAE is retrained with a joint loss, and the class labels are generated for each sample.

<p align="center"><img src="/assets/images/Figure4-1.png" alt="drawing" width="65%"/> </p>

More details can be found in our paper: 

E. Ozanich, A. Thode, P. Gerstoft, L. A. Freeman, and S. Freeman, “Deep embedded clustering of coral reef bioacoustics,”
_J. Acoust. Soc. Am. 149_ (2021): 2587–2601.


 {% include jsmath.js %}
 
## Data Generation and Processing
Fish communicate with series of pulses, or trains, that are generated with their swim bladders. We simulate pulse trains using a Gaussian-modulated sinusoid with _N_ $$\in\lfloor 13 \times beta(3.5, 8)\rfloor$$ random pulses and a center frequency $$f_c$$ based on observations of real fish pulses. The pulses are randomly offset and spaced at a random _dt_. The random seed and reference phase are fixed, which guarantees we are always generating the same dataset!


```python
rng(seed)
iphase = rd.rand(1)
for ii in np.arange(N):
	⋮
	x = x+np.sin(-2*np.pi*(fc*(t-dt*ii-offset)+iphase))*np.exp(-a*(t-dt*ii-offset)**2)
```

The discrete fourier transform (DFT) is applied in overlapping segments to generate a spectrogram. As an image, we keep only the magnitude (absolute value). The DFT "smears" the fish pulses, which are shorter than the DFT length! We use the same DFT length for fish calls and whale song, aiming for a happy medium between good time resolution and frequency resolution of both.

| ---------------------------------------------------------------------------------------------|------------------------------|-----------------------------------------------------------------------------------------------|
|<img align="center" src="/assets/images/Fish_Call_Timeseries.png" alt="drawing" width="100%"/>|$$\xrightarrow[]{\text{DFT}}$$|<img align="center" src="/assets/images/Fish_Call_Spectrogram.png" alt="drawing" width="97%"/>|


Humpback whales were heard calling offshore during our experiment. Their songs are non-impulsive frequency-modulated sinusoids, or quadratic sweeps. The duration of the sweep _T_, start frequency $$f_0$$, and bandwidth _df_ are based on observed data and previous studies. The call can have an upsweep (_df_ > 0) or a downsweep (_df_ < 0).


```python
rng(seed)
delta = (t>offset)*(t<(offset+T))
x = delta*np.sin(-2*np.pi*(df/(T**2)/3*(t-offset)**3 + f0*(t-offset) + rd.rand(1)))
```

| ---------------------------------------------------------------------------------------------|------------------------------|-----------------------------------------------------------------------------------------------|
|<img align="center" src="/assets/images/Whale_Call_Timeseries.png" alt="drawing" width="100%"/>|$$\xrightarrow[]{\text{DFT}}$$|<img align="center" src="/assets/images/Whale_Call_Spectrogram.png" alt="drawing" width="97%"/>|



Both signals are normalized to their maximum, and then Gaussian noise is added with a random signal-to-noise ratio (SNR) using $$\sigma$$.

```python
⋮
x = x/np.max(np.abs(x[:]))
xn = x+sigma*rd.randn(x.shape[-1])
```

The reconstructed images serve as simplified versions of the observed data, allowing us to test the effect of relative class ratios on the DEC model, and to use transfer learning from the simulated data to real data.

<p align="center"><img src="/assets/images/simulated_calls.png" alt="drawing" width="60%"/></p>

## Deep embedded clustering
A convolutional autoencoder is used discover informative 5x4 kernels of the original image (left panel) and reconstruct it from the reduced-dimension kernels (panel 2). At the smallest dimension, the kernels are flattened into 10x1 vectors representing the image. The mean squared error (MSE) between the reconstructed image and the original image is used to tune the CAE weights and kernels for the best reproduction.


<p align="center"><img src="/assets/images/DEC_recon_example.png" alt="drawing" width="60%"/></p>

Then, a cluster loss is included at the kernel layer, and the CAE weights are tuned with both MSE and the cluster loss to make the kernels more separable (panel 3)

<p align = "center"> $$ L = 0.1\cdot KL + 0.9\cdot MSE $$ </p>.

_KL_ is the Kullback-Leibler divergence, 

<p align = "center"> $$KL(P||Q) = \sum_n \sum_k p_{nk} \log \left(\frac{p_{nk}}{q_{nk}} \right) $$ </p>

$$q_{nk}$$ is the empirical Student's t-distribution of the kernel variables around their cluster means, which are initialized with K-means.
$$p_{nk}$$ is the weighted squared distribution of $$q_{nk}$$, and it puts more penalty on kernels that are further from their cluster center, causing them to have larger adjustments toward the center during training.

## Results

Analysis of 100 simulations of different datasets shows the performance of the DEC on class separation using the spectrogram images. The known labels from the simulations were used to test the model performance.


Three cases were considered: equal-sized groups (classes) of whale song and fish calls __(a-c)__, inequal-sized classes of whale song and fish call __(d-f)__, and equal-sized clsases of whale song, fish call, and whale song overlaying fish call __(g-i)__.
Since we suspect many more fish calls than whale song after observing the experimental data, we test the effect of inequal class size on the model.

<p align="center"><img src="/assets/images/DEC_simulated.png" alt="drawing" width="60%"/></p>

Observing that DEC does not perform well for inequal class sizes, we added the Gaussian mixture means (GMM) method for comparison. GMM was applied directly to the kernels from CAE without additional training of the DEC.
The DEC algorithm uses the K-means assumption of equi-distant clusters (fixed-variance Gaussians), whereas GMM learns the variance and the cluster means, allowing it to optimize cluster sizes.

<p align="center"><img src="/assets/images/DEC_experiment_updated.png" alt="drawing" width="60%"/></p>

Using some hand-labeled events detected in the experiment data, we show that the GMM does perform better than DEC due to the class imbalance in the real data.
Unlike curated data, these raw data contain multiple, often overlapping, signals with varying SNR, including unknown signals not accounted for. Additional array processing and human labeling is needed to raise the accuracy from 77% to >90%.

<p align="center"><img src="/assets/images/messy_classifications.png" alt="drawing" width="60%"/></p>
