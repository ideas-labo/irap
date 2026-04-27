Accurately converting performance requirements described in natural language into computable quantitative indicators is a key challenge in the field of software engineering, transitioning from qualitative evaluation to quantitative analysis. Our goal is to establish a stakeholder satisfaction function \( f(x) \) for any performance indicator \( x \), where the function outputs the stakeholder satisfaction (within the range \([0, 1]\)) corresponding to the value of \( x \).

We adopt a piecewise linear function to characterize this satisfaction curve, which offers simplicity, interpretability, and sufficient expressive power. Such a piecewise linear function can be uniquely determined by its set of inflection points \(\{(x_1, y_1), \ldots, (x_n, y_n)\}\).

For example, consider the requirement statement:

> In real-time ECG monitoring scenarios, the software must receive and process ECG signal data at a sampling frequency of no less than 1000Hz.

Given a preset tolerance range of 10%, the set of inflection points for this requirement is:
$$
\{(900, 0), (1000, 1)\}
$$
This set can directly restore the corresponding quantitative function \( f(x) \):
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
\text{Performance requirement statement} \rightarrow [(x_1, y_1), \ldots, (x_n, y_n)]
$$

Please convert the following performance requirement into a quantitative form based on the above theory (output the result as a list of 2D points, such as [[10.0, 1.0], [11.0, 0.0]]. Please strictly follow the specified format for output, and do not include any additional content.):
> 