# LWC-84、节点自由表面与三侧 CFS-CPML

## 物理域与离散约定

求解方程为

\[
p_{tt}=L(p)+q,\qquad L(p)=c^2(x,z)\left(D_{xx}^{(8)}p+D_{zz}^{(8)}p\right).
\]

物理网格是节点中心的 `401×401`，`x,z=0,5,…,2000 m`，数组顺序为 `[z,x]`。左、右、下各向外扩展 40 个 CPML 节点，顶部没有 CPML；不含差分 halo 的计算数组是 `441×481`。九点二阶导数系数（偏移 `-4…4`）为

```text
[-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560] / h²
```

空间变速时严格使用复合算子 `L(L(p)+q)`，不会把它化为错误的 `c⁴∇⁴p`。

## LWC-84 时间推进与震源修正

物理域核心推进为

\[
p^{n+1}=2p^n-p^{n-1}+\Delta t^2 a^n+
\frac{\Delta t^4}{12}\left[L(a^n)+q_{tt}^n\right],\qquad
a^n=L(p^n)+q^n.
\]

Ricker 子波设 `a=π²f0²`、`τ=t-t0`、`t0=1.5/f0`：

\[
w=(1-2a\tau^2)e^{-a\tau^2},
\]

\[
w_t=(-6a\tau+4a^2\tau^3)e^{-a\tau^2},
\]

\[
w_{tt}=(-6a+24a^2\tau^2-8a^3\tau^4)e^{-a\tau^2}.
\]

四阶一致起步为

\[
p^1=p^0+\Delta t\,p_t^0+\frac{\Delta t^2}{2}(Lp^0+q^0)
+\frac{\Delta t^3}{6}(Lp_t^0+q_t^0)
+\frac{\Delta t^4}{24}\{L(Lp^0+q^0)+q_{tt}^0\}.
\]

实现位于 `lwc84.py`、`ricker.py`、`stencils.py`。时间循环中的 Ricker 值及导数在 GPU 上按 batch 向量化计算，没有每步 `.item()`、`.cpu()` 或有限性主机同步。

## 顶部压力自由表面

`z=0` 是真实物理节点，并在每次空间算子调用前强制 `p[:,0,:]=0`。半径 4 的顶部 ghost 按节点奇延拓：

\[
p(-j\Delta z)=-p(j\Delta z),\quad j=1,2,3,4.
\]

这与八阶模板一致，并产生压力反射极性翻转。数值验收中顶部最大绝对压力为 0，反射波与等传播距离直达波的相关系数为 `-0.99729`，幅值比为 `1.00742`。

## 无分裂 CFS-CPML/ADE

每个方向使用复频移坐标伸缩

\[
s_i=\kappa_i+\frac{\sigma_i}{\alpha_i+i\omega},\qquad i\in\{x,z\}.
\]

剖面从物理域/CPML 界面到外边界按归一化深度 `d∈[0,1]` 变化：

\[
\sigma(d)=\sigma_{max}d^m,\qquad
\kappa(d)=1+(\kappa_{max}-1)d^m,\qquad
\alpha(d)=\alpha_{max}(1-d),
\]

其中 `m=3`、`κmax=3`、`αmax=π fmin`、`fmin=8 Hz`，

\[
\sigma_{max}=-\frac{(m+1)c_{ref}\ln R}{2L_{pml}},\qquad R=10^{-8}.
\]

离散 ADE 系数为

\[
b_i=\exp[-(\sigma_i/\kappa_i+\alpha_i)\Delta t],\qquad
a_i=\frac{\sigma_i(b_i-1)}{\kappa_i(\sigma_i+\kappa_i\alpha_i)}.
\]

每个方向维护两组记忆变量 `ψ_i`、`φ_i`。对当前压力执行：

1. 用八阶一阶导数计算 `D_i p`；
2. `ψ_i←b_i ψ_i+a_i D_i p`；
3. `g_i=κ_i⁻¹D_i p+ψ_i`；
4. 计算 `D_i g_i`；
5. `φ_i←b_i φ_i+a_i D_i g_i`；
6. `D̃_ii p=κ_i⁻¹D_i g_i+φ_i`；
7. 仅在 CPML active 区采用 `D̃_ii`，物理区采用直接九点 `D_ii^(8)`。

顶部的 `σ_z=0`，所以没有顶部 PML。外侧计算边界清零只发生在 CPML 之外，不能替代 CPML。实现没有乘法阻尼 mask、Cerjan 海绵或一阶吸收边界。

四阶修正项对瞬时加速度再次应用物理 LWC 算子；ADE 记忆变量按上述递推更新一次。因此准确表述是：**物理区域 LWC-84 核心与三侧无分裂 CFS-CPML 边界耦合**。当前没有把整个扩展域宣称为已严格证明的全局四阶时间格式。

## 数值验收

`scripts/validate_lwc84_numerics.py` 用较深参考域相减，隔离底部 CPML 差异反射。CUDA 实测如下：

| f0 (Hz) | 垂直反射比 / dB | 斜入射反射比 / dB |
|---:|---:|---:|
| 10 | 1.0841e-4 / -79.30 | 1.2195e-4 / -78.28 |
| 15 | 7.6494e-5 / -82.33 | 8.9386e-5 / -80.97 |
| 25 | 7.2637e-5 / -82.78 | 6.7633e-5 / -83.40 |

验收产物位于 `artifacts/lwc84_numerical_validation_401to201_801/`。这些值属于当前离散、网格与窗口的实测结果，不等同于理论参数 `R=1e-8`。

## 参考文献

- Roden, J. A. & Gedney, S. D. (2000), “Convolution PML: An efficient FDTD implementation of the CFS-PML for arbitrary media,” *Microwave and Optical Technology Letters*, 27(5), 334–339, DOI `10.1002/1098-2760(20001205)27:5<334::AID-MOP14>3.0.CO;2-A`.
- Komatitsch, D. & Martin, R. (2007), “An unsplit convolutional perfectly matched layer improved at grazing incidence for the seismic wave equation,” *Geophysics*, 72(5), SM155–SM167, DOI `10.1190/1.2757586`.
- Fornberg, B. (1988), “Generation of finite difference formulas on arbitrarily spaced grids,” *Mathematics of Computation*, 51(184), 699–706, DOI `10.1090/S0025-5718-1988-0935077-0`.

