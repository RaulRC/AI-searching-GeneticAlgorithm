# AI-searching-GeneticAlgorithm
Genetic Algorithm Python implementation based on Russell &amp; Norvig


## Example

Given the lineal equation

f(x1, x2, ..., xn) = w1·x1 + w2·x2 + ... wn·xn

or the polynomial equation

f(x1, x2, ..., xn) = w1 + w2·x2^2 + ... wn·xn^n

Search for a set of independent variables, X

```{python}
X = {x1, x2, ..., xn}
```

which maximize f(x1, x2, ... xn)


## Solution

Having for example ```W``` defined as:

```{python}
W = [2.5, 3.4, -1.9, 4.9, 0, -8, 1]
```

run

```{bash}
$ python3 GeneticAlgorithm_Example.py
```

output

```{bash}
Best individual: [ 8.39555748e-01  3.22001033e+03 -2.80964618e+03  3.95066504e+03
 -1.29639545e+02 -3.05995054e+03  8.53383140e-01]
Best value: 60127.17814725384

```

![alt text](https://github.com/RaulRC/AI-searching-GeneticAlgorithm/blob/master/results.png "GA Example results")
