Please help me extract keywords from the performance requirement. There are two types of keywords: the first is **performance indicators**, and the second is **key phrases expressing expectations**. For example, for the following performance requirement:
> In real-time ECG monitoring scenarios, the software must receive and process ECG signal data at a sampling frequency of no less than 1000Hz.

The performance indicator keyword is "frequency", and the key phrase expressing expectations is "no less than". Please return the extracted keywords **in the form of a JSON list**. For the above example, your return value should be: ["frequency", "no less than"]. Please process the following performance requirement:
> 