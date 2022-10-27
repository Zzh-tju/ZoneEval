<img src="flyleaf.png"/>

# Why does spatial equilibrium matter?

### The detector cannot perform uniformly across the zones.
<img src="detection-quality.png"/>

### The detection performance is correlated with the object distribution.

When the object distribution satisfies the centralized photographer’s bias, the detector will favor more to the central zone, while losing the performance in most areas outside.

### This is not good for robust detection application.

If you have a fire dataset like this, the detector will be good at detecting fire in the central zone of the image. But for the zone near to the image border, uh huh, hope you are safe.

<img src="fire-data.png" width="600"/>
<img src="fire.png" width="600"/>

## Zone Evaluation

Let’s start by the definition of evaluation zones. We define a rectangle region $R_i=\text{Rectangle}(p,q)=\text{Rectangle}((r_iW,r_iH),((1-r_i)W,(1-r_i)H))$ like this, 
<div align="center"><img src="rectangle.png" width="300"/></div>

where $i\in$ { $0,1,\cdots,n$ }, $n$ is the number of zones.

Then, the evaluation zones are disigned to be a series of annular zone $z_i^j=R_i\setminus R_j$, $i\textless j$.
We denote the range of the annular zone $z_i^j$ as $(r_i,r_j)$ for brevity.

<div align="center"><img src="zone-range.gif" width="300"/></div>

We measure the detection performance for a specific zone $z_i^j$ by only considering the ground-truth objects and the detections whose centers lie in the zone $z_i^j$.
Then, for an arbitrary evaluation metric $m$, for instance Average Precision (AP), the evaluation process stays the same to the conventional ways, yielding Zone Precision (ZP), denoted by ZP@ $z_i^j$. Consider the default setting $n=5$, the evaluation zones look like this,

<div align="center"><img src="eval-zone.png" width="300"/></div>

Now that we have 5 ZPs, and they indeed provide more information about the detector's performance. We further present a **S**patial equilibrium **P**recision (SP), and we use this single value to characterize the detection performance for convenient usage.

<div align="center"> $\mathrm{SP}=\sum\limits_{i=0}^{n-1}\mathrm{Area}(z_i^{i+1})\mathrm{ZP}\text{@}z_i^{i+1}$ </div>

where $\mathrm{Area}(z)$ calculates the area of the zone $z_i^j$ in the normalized image space (square image with unit area 1). In general, SP is a weighted sum of the
 5 ZPs, that is,
 
<div align="center"> $\mathrm{SP}=0.36\mathrm{ZP}\text{@}z_0^1+0.28\mathrm{ZP}\text{@}z_1^2+0.20\mathrm{ZP}\text{@}z_2^3+0.12\mathrm{ZP}\text{@}z_3^4+0.04\mathrm{ZP}\text{@}z_4^5$ </div>

One can see that when $n=1$, our SP is identical to traditional AP as the term $\mathrm{Area}(z_i^j)=1$, which means that the detectors are assumed to perform uniformly in the whole image zone.
Our SP is based on the assumption to the traditional AP, i.e., the detector performs uniformly in the zone.
The difference is, our SP applies this assumption to a series of smaller zones, rather than the full map for traditional AP.
As $n$ increases, the requirements for spatial equilibrium become stricter and stricter. And a large $n$ is also acceptable if a more rigorous spatial equilibrium is required.
