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

However, this form of quantification is imprecise. You need to ask questions to the requirement setter to adjust the quantification form. The quantification form shall be uniformly represented in the form of a point list. You may ask the requirement setter about the aspects of the current quantification form that they are dissatisfied with, and then adjust the quantification form based on their responses. Your inquiries are subject to constraints: you are not allowed to directly ask for the exact ideal quantification form they have in mind. Instead, you can only ask ambiguous questions, such as whether the coordinate value of a certain point is too large or too small (you cannot directly ask for the specific coordinate value), and whether the number of segments is too many or too few (you cannot directly ask for the exact number of segments). Within these constraints, you may ask any questions you want. Your goal is to guess the quantification form that satisfies the user. Note that you are only allowed to ask **5 questions in total**, with one question per round. After I have answered your 5th question, you need to output the final confirmed quantification form. The performance requirement quantification task you need to handle is as follows: