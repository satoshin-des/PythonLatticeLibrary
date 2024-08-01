# Python Lattice Library(abbr. PLL)
This is a python library to use lattices on python.<br>
This library has algorithms to lattice reduce, solve SVP, solve CVP, and other many operation for lattices.

## How to Use?
### Initialization
If you want to use ```A = numpy.array([[123, 0, 0], [234, 1, 0], [345, 0, 1]])``` as lattice basis matrix, you can do like below:

```Python
import PythonLatticeLibrary as PLL
import numpy as np

A = np.array([[123, 0, 0], [234, 1, 0], [345, 0, 1]])
b = PLL.lattice(A)
b.print()
```
Output is below.

```
Basis =
[[123   0   0]
 [234   1   0]
 [345   0   1]]
Rank = 3
Volume = 122.99999999999994
```

Not only numpy.array but list is available for argument of ``PLL.lattice()``:

```Python
import PythonLatticeLibrary as PLL
import numpy as np

A = [[123, 0, 0], [234, 1, 0], [345, 0, 1]]
b = PLL.lattice(A)
b.print()
```
Output is below.

```
Basis =
[[123   0   0]
 [234   1   0]
 [345   0   1]]
Rank = 3
Volume = 122.99999999999994
```

You can generates a random lattice like below:

```Python
import PythonLatticeLibrary as PLL

b = PLL.random_lattice(5) # Generates 5-dimensional lattice basis.
b.print()
```

Output is below:

```
Basis =
[[875   0   0   0   0]
 [932   1   0   0   0]
 [951   0   1   0   0]
 [801   0   0   1   0]
 [754   0   0   0   1]]
Rank = 5
Volume = 875.0000000044823
```

You can reduce lattice basis(e.g. LLL-reduction, Deep-LLL-reduction, etc.) like below:

```Python
import PythonLatticeLibrary as PLL

b = PLL.random_lattice(5)
b.print()
b.basis = b.LLL() # LLL-reduction
b.print()
```

Output is below.

```
Basis =
[[711   0   0   0   0]
 [940   1   0   0   0]
 [500   0   1   0   0]
 [592   0   0   1   0]
 [555   0   0   0   1]]
Rank = 5
Volume = 710.9999999597205
Basis =
[[ 0  1  2  1 -2]
 [-1  2  2 -1  1]
 [ 1 -3  2  0  2]
 [ 0  2 -1  3  2]
 [-4 -1  1  1  1]]
Rank = 5
Volume = 710.9999999999999
```

You can select reduction parameter:

```Python
import PythonLatticeLibrary as PLL

b = PLL.random_lattice(5)
b.print()
b.basis = b.LLL(delta=0.5) # LLL-reduction
b.print()
```

Output is below:

```
Basis =
[[814   0   0   0   0]
 [659   1   0   0   0]
 [723   0   1   0   0]
 [912   0   0   1   0]
 [781   0   0   0   1]]
Rank = 5
Volume = 813.9999999534335
Basis =
[[ 1  1 -1  1  1]
 [-2 -1  1  0  2]
 [ 2 -4  0  2  0]
 [-2 -2 -3  2 -1]
 [-1  2  2  4 -3]]
Rank = 5
Volume = 813.9999999999997
```

You can use other functions for lattice reduction like LLL. Below is the examples.

```Python
import PythonLatticeLibrary as PLL

b = PLL.random_lattice(5)
b.print()
b.basis = b.DeepLLL() # Deep-LLL-reduction
b.print()
```

Output is below:

```
Basis =
[[726   0   0   0   0]
 [952   1   0   0   0]
 [676   0   1   0   0]
 [655   0   0   1   0]
 [900   0   0   0   1]]
Rank = 5
Volume = 726.0000000397084
Basis =
[[-2 -1 -1  0  1]
 [ 1 -1  1  1  2]
 [ 0 -2  2  0 -1]
 [-3  3  4 -1  1]
 [ 0 -1 -2  6 -1]]
Rank = 5
Volume = 725.9999999999999
```

You can solve SVP using function ``ENUM_SVP()``:

```Python
import PythonLatticeLibrary as PLL

b = PLL.random_lattice(7)
b.print()
b.basis = b.LLL() # LLL-reduction
v = b.ENUM_SVP()
print(v)
```

Output is below:

```
Basis =
[[924   0   0   0   0   0   0]
 [719   1   0   0   0   0   0]
 [552   0   1   0   0   0   0]
 [723   0   0   1   0   0   0]
 [608   0   0   0   1   0   0]
 [834   0   0   0   0   1   0]
 [995   0   0   0   0   0   1]]
Rank = 7
Volume = 924.000000001688
[ 0 -1  0 -1  1  1  0]
```

As same as solving an SVP, you can solve CVP to target t using ``ENUM_CVP(t)``:

```Python
import PythonLatticeLibrary as PLL
import numpy as np

b = PLL.random_lattice(7)
t = np.random.randint(1, 20, size=7)
b.print()
print(t)
b.basis = b.LLL() # LLL-reduction
v = b.ENUM_CVP(t)
print(v)
```

Output is below:

```
Basis =
[[885   0   0   0   0   0   0]
 [799   1   0   0   0   0   0]
 [566   0   1   0   0   0   0]
 [766   0   0   1   0   0   0]
 [650   0   0   0   1   0   0]
 [654   0   0   0   0   1   0]
 [541   0   0   0   0   0   1]]
Rank = 7
Volume = 884.9999999999998
[18  1  4 17 18  9 15]
[17  2  5 17 18  8 15]
```

## What Functions are Available?
Available functions in this library are below(This is not a list of all functions in this library):
- ```vol()```: Computes volume of the lattice.
- ```GSO()```: Computes Gram-Schmidt information of the lattice basis.
- ```potential()```: Computes potential of the lattice basis.
- ```size()```: Size-reduces the lattice basis.
- ```Gauss()```: Gauss-reduces the 2-dimensional lattice basis.
- ```LLL(delta)```: LLL-reduces the lattice basis.
- ```DeepLLL(delta)```: Deep-LLL-reduces the lattice basis.
- ```ENUM_SVP()```: Enumerates the shortest vector on the lattice.
- ````Babai(t)````: Computes an approximate solution of CVP to target t on the lattice with Babai's nearest plane algorithm.
- ```ENUM_CVP(t)```: Enumerates the closest vector to target t on the lattice.