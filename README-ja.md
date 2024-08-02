# Python Lattice Library(abbr. PLL)
このPythonライブラリの中には主に格子基底簡約，最短ベクトル問題(SVP)や最近ベクトル問題(CVP)の求解，そして格子に関する演算などが備わっています．

## 使い方
### 初期化
もし```A = numpy.array([[123, 0, 0], [234, 1, 0], [345, 0, 1]])``` を格子基底行列として使いたい場合には以下の様にします：

```Python
import PythonLatticeLibrary as PLL
import numpy as np

A = np.array([[123, 0, 0], [234, 1, 0], [345, 0, 1]])
b = PLL.lattice(A)
b.print()
```
出力は以下：

```
Basis =
[[123   0   0]
 [234   1   0]
 [345   0   1]]
Rank = 3
Volume = 122.99999999999994
```

``numpy.array``だけではなく，``list``でもできます:

```Python
import PythonLatticeLibrary as PLL
import numpy as np

A = [[123, 0, 0], [234, 1, 0], [345, 0, 1]]
b = PLL.lattice(A)
b.print()
```
出力は以下：

```
Basis =
[[123   0   0]
 [234   1   0]
 [345   0   1]]
Rank = 3
Volume = 122.99999999999994
```

乱数的に格子基底を生成することもできます：

```Python
import PythonLatticeLibrary as PLL

b = PLL.random_lattice(5) # Generates 5-dimensional lattice basis.
b.print()
```

出力は以下：

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

格子基底簡約(e.g. LLL簡約, DeepLLL簡約, etc.)をしたいときは，以下の様にしてできます：

```Python
import PythonLatticeLibrary as PLL

b = PLL.random_lattice(5)
b.print()
b.basis = b.LLL() # LLL-reduction
b.print()
```

出力は以下：

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

簡約パラメタを指定するときは以下の様にできます．

```Python
import PythonLatticeLibrary as PLL

b = PLL.random_lattice(5)
b.print()
b.basis = b.LLL(delta=0.5) # 0.5-LLL簡約
b.print()
```

出力は以下：

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

LLL簡約以外の簡約も，LLLと同じような感じでできます．以下は一例：

```Python
import PythonLatticeLibrary as PLL

b = PLL.random_lattice(5)
b.print()
b.basis = b.DeepLLL() # DeepLLL簡約
b.print()
```

出力は以下：

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

最短ベクトルを求めるときは``ENUM_SVP()``でできます：

```Python
import PythonLatticeLibrary as PLL

b = PLL.random_lattice(7)
b.print()
b.basis = b.LLL() # LLL-reduction
v = b.ENUM_SVP()
print(v)
```

出力は以下：

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

tを目標ベクトルとする最近ベクトルを求めるときも，最短ベクトルと同様に``ENUM_CVP(t)``でできます：

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

出力は以下：

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

## 使える関数
使える関数は以下（ただし以下の表は使える関数の一覧ではなくいくつか主要な函数を抜萃したに過ぎないことには注意されたい）：
- ```vol()```: 格子の体積を求める．
- ```GSO()```: GS0情報を求める．
- ```potential()```: 格子基底のpotentialを求める．
- ```size()```: サイズ基底簡約．
- ```Gauss()```: Gauss簡約．
- ```LLL(delta)```: delta-LLL簡約．
- ```DeepLLL(delta)```: delta-DeepLLL簡約．
- ```ENUM_SVP()```: 最短ベクトルを数え上げ法で求める．
- ````Babai(t)````: tを目標ベクトルとする近似最近ベクトルをBabaiの最近平面法で求める．
- ```ENUM_CVP(t)```: tを目標ベクトルとする最近ベクトルを数え上げ法で求める．