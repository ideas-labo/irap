Accurately converting performance requirements described in natural language into computable quantitative indicators is a key challenge in the field of software engineering, shifting from qualitative evaluation to quantitative analysis. Our goal is to establish a stakeholder satisfaction function $f(x)$ for any performance indicator $x$, where the function outputs the stakeholder satisfaction (ranging from $[0, 1]$) corresponding to the value of $x$.

We adopt piecewise linear functions to characterize such satisfaction curves, as they offer simplicity, ease of interpretation, and sufficient expressive power. This type of piecewise linear function can be uniquely determined by its set of inflection points $\{(x_1, y_1), \ldots, (x_n, y_n)\}$.

For example, consider the requirement statement:

> In real-time ECG monitoring scenarios, the software must receive and process ECG signal data at a sampling frequency of no less than 1000Hz.

With a preset tolerance range of $10\%$, the set of inflection points for this requirement is:
$$
\{(900, 0), (1000, 1)\}
$$
This set can directly restore the corresponding quantitative function $f(x)$:
$$
f(x) = 
\begin{cases}
0 & \text{if } x \leq 900 \\
1/100(x-900) & \text{if } 900 < x < 1000 \\
1 & \text{if } x \geq 1000
\end{cases}
$$
Thus, our task is formally defined as a sequence-to-sequence conversion problem:
$$
\text{Performance Requirement Statement} \rightarrow [(x_1, y_1), \ldots, (x_n, y_n)]
$$

However, the quantitative form directly derived from performance requirement statements is not precise and does not fully align with the quantitative form expected by the requirement setters. The following examples illustrate the differences between the quantitatively derived form from the literal meaning and the expected quantitative form by the requirement setters:
