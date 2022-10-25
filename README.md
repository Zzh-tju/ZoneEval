<img src="flyleaf.png"/>

# Why does spatial equilibrium matter?

### The detector cannot perform uniformly across the zones.
<img src="detection-quality.png"/>

### The detection performance is correlated with the object distribution.

When the object distribution satisfies the centralized photographer’s bias, the detector will favor more to the central zone, while losing the performance in most areas outside.

### This is not good for robust detection application.

If you have a fire dataset like this, the detector will be good at detecting fire in the central zone of the image. But for the zone near to the image border, uh huh, hope you are safe.
<img src="fire-data.png" width="600"/>
<img src="fire.png"/>

## Zone Evaluation

Let’s start by the definition of evaluation zones. We define a rectangle region $R_i=\text{Rectangle}(p,q)=\text{Rectangle}((r_iW,r_iH),((1-r_i)W,(1-r_i)H))$ like this, 
<img src="rectangle.png"/>
where $i\in{0,1,\cdots,n}$, $n$ is the number of zones.
