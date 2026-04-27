For a performance requirement statement, we can quantify it. Specifically, we expect to derive a function expression that takes a performance metric as input and outputs the user satisfaction level corresponding to that metric. We may simply assume that the function expression is a piecewise linear function. For example, given the requirement statement:

> In the scenario of real-time electrocardiogram (ECG) monitoring, the software shall receive and process ECG signal data at a sampling frequency no lower than 1000 Hz.

The corresponding piecewise linear quantification function is:
$$
f(x) = 
\begin{cases}
0 & \text{if } x \leq 900 \\
1/100(x-900) & \text{if } 900 < x < 1000 \\
1 & \text{if } x \geq 1000
\end{cases}
$$
A piecewise linear function has multiple "inflection points". The set of inflection points for the above piecewise linear function is:
$$
\{(900,0),(1000,1)\}
$$

Please extract the formal answer from the content below, and output **only** the form of a point list. For example, the only answer you can output is: "[[10.0, 1.0], [11.0, 0.0]]". The content you need to process is: