# **AbstrakTS Documentation**

**Rizal Purnawan**\
ORCID: [0000-0001-8858-4036](https://orcid.org/0000-0001-8858-4036)

## **1. Introduction**

This markdown document is a technical documentation for `AbstrakTS` in `abstrakTS` module. The mathematical frameworks of algorithms contained in `AbstrakTS` will be explained in this technical documentation. Formal mathematical expressions will regularly be employed in the frameworks. Therefore, the readers are expected to have sufficient knowledge in some mathematical subjects in order to smoothly be able to understand the frameworks. Otherwise, we encourage the readers to explore several notions in set theory (Stoll, 1979), first order logic (Stoll, 1979), measure theory (Salamon, 2016), probability theory (Bremaud, 2020) and general topology (Andre, 2024).

## **2. Important Probability Theoretic and Statistics Concepts**

Probability theory and statistics will play a huge role in the mathematical framework of this algorithm. Therefore, it is essential to understand the basic concepts in probability theory for understanding the framework.

For a fluid discussion, let us define a general probability space $(\Omega, \mathcal{F}, P)$. Note that $\Omega$ is the set of all samples, $\mathcal{F}$ is a family of subsets of $\Omega$ known as a $\sigma$-algebra on $\Omega$ and it represents events related to $\Omega$ and $P: \mathcal{F} \to [0, 1]$ is a probability measure. A structure inherently attached to probability space is random variable, which is a $P$-measurable function $X: \Omega \to \mathbb{R}$, meaning that it is a measurable function with respect to the probability space $(\Omega, \mathcal{F}, P)$. In particular, we designate $X: \Omega \to \mathbb{R}$ to be an $L^2(P)$ function, which is employed in most practices.

### **2.1. Expectation**

The first probability theoretic concept to be presented is expectation, which is given as a map $\mathrm{E}: L^1(P) \to \mathbb{R}$ and is defined by a Lebesgue integral
\begin{equation}
    \forall X \in L^1(P) :\;
    \mathrm{E}[X] := \int_{\Omega} X \,\mathrm{d}P \,.
\end{equation}
Note that for any finite measure $\mu$, $L^p(\mu) \subset L^1(\mu)$ is true. And probability spaces are equipped with probability measures which are finite: $P(\Omega) = 1 < \infty$. Therefore, $L^2(P) \subset L^1(P)$, and expectation can also be applied to $L^2(P)$ random variables.

### **2.2. Covariance**

The next concept is covariance, which can informally be understood as the expectation of the product of deviations of two random variables. Formally, it is a map $\mathrm{cov}: L^2(P) \times L^2(P) \to \mathbb{R}$ defined by
\begin{equation}
    \forall X, Y \in L^2(P) :\;
    \mathrm{cov}(X, Y) := \mathrm{E}[(X - \mathrm{E}[X])(Y - \mathrm{E}[Y])] \,.
\end{equation}
Follows from the linearity property of Lebesgue integral, we can simplify the expression of covariance into
\begin{equation}
    \forall X, Y \in L^2(P) :\;
    \mathrm{cov}(X, Y) = \mathrm{E}[X Y] - \mathrm{E}[X] \mathrm{E}[Y] \,.
\end{equation}

### **2.3. Variance**

A derivative concept of covariance is variance, which is the covariance of a random variable with itself. It is formally given as a map $\mathrm{var}: L^2(P) \to \mathbb{R}$ defined by
\begin{equation}
    \forall X \in L^2(P) :\;
    \mathrm{var}(X) := \mathrm{cov}(X, X) = \mathrm{E}[X^2] - \mathrm{E}[X]^2 \,.
\end{equation}

### **2.4. Standard Deviation**

Then, a derivative concept of variance is standard deviation, which is defined as the square root of variance. IT is formally given as a map $\mathrm{std}: L^2(P) \to \mathbb{R}$ defined by
\begin{equation}
    \forall X \in L^2(P) :\;
    \mathrm{std}(X) := \sqrt{ \mathrm{var}(X) } = \left( \mathrm{E}[X^2] - \mathrm{E}[X]^2 \right)^{\frac{1}{2}} \,.
\end{equation}

### **2.5. Correlation**

The next one is correlation. There are several formulations regarding correlation. The standard correlation is the Pearson correlation. Pearson correlation is well-defined on a family of $L^2(P)$ random variables which are almost surely non-constants. Suppose
\begin{equation}
    \mathcal{V} := \{ X \in L^2(P) \mid \mathrm{var}(X) > 0 \} \,,
\end{equation}
i. e., $\mathcal{V}$ is the family of such random variables. Pearson correlation is given as a map $\mathrm{corr}: \mathcal{V}\times \mathcal{V} \to [-1, 1]$ defined by
\begin{equation}
    \forall X, Y \in \mathcal{V} :\;
    \mathrm{corr}(X, Y)
    := \frac{ \mathrm{cov}(X, Y) }{ \mathrm{std}(X) \cdot \mathrm{std}(Y) }
    = \frac{ \mathrm{E}[X Y] - \mathrm{E}[X] \mathrm{E}[Y] }{
        \left( \mathrm{E}[X^2] - \mathrm{E}[X]^2 \right)^{\frac{1}{2}}
        \left( \mathrm{E}[Y^2] - \mathrm{E}[Y]^2 \right)^{\frac{1}{2}}
    }
    \,.
\end{equation}
Another formulation of correlation is the Spearman correlation, which technically is the Pearson correlation of rank variables of a pair of random variables. A rank variable of a random variable $X \in L^2(P)$ is a map $R_X: \Omega \to \mathbb{N}$ such that $R_X(\omega)$ is the rank of $X(\omega)$ on $X$, for every $\omega \in \Omega$. Note that $R_X \in L^2(P)$. Spearman correlation can be given as a map $\mathrm{corr}_r: \mathcal{V} \times \mathcal{V} \to [-1, 1]$ defined by
\begin{equation}
    \forall X, Y \in \mathcal{V} :\;
    \mathrm{corr}_r(X, Y) := \mathrm{corr}( R_X, R_Y ) \,.
\end{equation}
Spearman correlation describes monotonic relation between two random variables.

### **2.6. Conditional Expectation**

Conditional expectation is expectation of a restricted random variable with respect to the restricted probability space. It can also be defined via the original probability space without accounting for the restrictions. In terms of the original probability space, it is a map $\mathrm{E}: L^1(P) \times \mathcal{F} \to \mathbb{R}$ defined by
\begin{equation}
    \forall X \in L^1(P) \forall A \in \mathcal{F} :\;
    \mathrm{E}[X \mid A]
    := \frac{1}{P(A)} \int_{A} X \,\mathrm{d}P \,.
\end{equation}
In this sense, we can also view the ordinary expectation as a conditional expectation $\mathrm{E}[\cdot \mid \Omega]$ as it is shown by
\begin{equation}
    \forall X \in L^1(P) :\;
    \mathrm{E}[X \mid \Omega] = \frac{1}{P(\Omega)} \int_\Omega X \,\mathrm{d}P = \frac{1}{1} \int_\Omega X \,\mathrm{d}P = \int_\Omega X \,\mathrm{d}P = \mathrm{E}[X] \,.
\end{equation}
As mentioned, conditional expectation can be viewed as a restriction. Let us fix an event $A \in \mathcal{F}$ and a random variable $X: \Omega \to \mathbb{R}$. The restriction probability space on $A$ is $(A, \mathcal{F}_A, P_A)$ where $\mathcal{F}_A$ is the $\sigma$-algebra on $A$ and it is in fact a sub-$\sigma$-algebra of $\mathcal{F}$ and the probability measure $P_A: \mathcal{F}_A \to [0, 1]$ is defined by as the conditional probability
\begin{equation}
    \forall B \in \mathcal{F}_A :\;
    P_A(B) := P(B \mid A) = \frac{ P(B \cap A) }{P(A)} = \frac{P(B)}{P(A)} \,.
\end{equation}
Then conditional expectation of $X$ given $A$ can be defined as the expectation of the restriction $X\big|_A: A \to \mathbb{R}$ with respect to $(A, \mathcal{F}_A, P_A)$ and hence
\begin{equation}
    \mathrm{E}[X \mid A]
    := \mathrm{E}\left[ X\big|_A \right]
    = \int_A X\big|_A \,\mathrm{d} P_A
    = \frac{1}{P(A)} \int_A X \chi_A \,\mathrm{d}P
    = \frac{1}{P(A)} \int_{A \cap A} X \,\mathrm{d}P
    = \frac{1}{P(A)} \int_A X \,\mathrm{d}P \,.
\end{equation}

### **2.7. Conditional Variance**

The notion of conditional expectation allows us to formulate the notion of conditional variance. Conditional variance is defined as variance with the expectation expressions within it replaced with conditional expectation. Formally, it is a map $\mathrm{var}(\cdot \mid \cdot): L^2(P) \times \mathcal{F} \to \mathbb{R}$ defined by
\begin{equation}
    \forall X \in L^2(P) \forall A \in \mathcal{F} :\;
    \mathrm{var}(X \mid A) := \mathrm{E}[X^2 \mid A] - \mathrm{E}[X \mid A]^2 \,.
\end{equation}

### **2.8. Conditional Standard Deviation**

Since conditional variance is well-defined, then conditional standard deviation naturally follows from it. Conditional standard deviation is a map $\mathrm{std}( \cdot \mid \cdot): L^2(P) \times \mathcal{F} \to \mathbb{R}$ defined as the square root of conditional variance, i. e.,
\begin{equation}
    \forall X \in L^2(P) \forall A \in \mathcal{F} :\;
    \mathrm{std}( X \mid A ) := \sqrt{ \mathrm{var}( X \mid A ) } \,.
\end{equation}

### **2.9. Moving Average**

Moving average is well-defined on random variables of particular probability space. The underlying sample space shall be a totally ordered set and a metric space. For a simplicity, let us designate that $\Omega \subset \mathbb{R}$ for this particular discussion. The sample $\Omega$ will be equipped with a metric $d: \Omega \times \Omega \to \mathbb{R}$ which is a restriction of a standard Euclidean metric on $\mathbb{R}$. Hence
\begin{equation}
    \forall x, y \in \Omega :\; d(x, y) := |x - y| \,.
\end{equation}
Then moving average of a random variable $X: \Omega \to \mathbb{R}$ is given as a map $\mathrm{MA}_X: \Omega \to \mathbb{R}$ defined by the conditional expectation
\begin{equation}
    \forall \omega \in \Omega:\;
    \mathrm{MA}_X(\omega) := \mathrm{E}[X \mid B_w(\omega)]
\end{equation}
where $B_w: \Omega \to \mathcal{P}(\mathbb{R})$ is an open ball operator of radius $w$, for some $w > 0$. The open ball operator is defined by
\begin{equation}
    \forall \omega \in \Omega :\;
    B_w(\omega) := \{ x \in \Omega \mid d(x, \omega) < w \} \,.
\end{equation}
It is always assumed that $B_w(\omega) \in \mathcal{F}$, for every $\omega \in \Omega$.

### **2.10. Moving Variance**

Moving variance is a similar concept to moving average. Let us also designate $\Omega \to \mathbb{R}$ for this discussion with a well-defined metric $d: \Omega \times \Omega \to \mathbb{R}$. Moving variance of a random variable $X \in L^2(P)$ is a map $\mathrm{MV}_X: \Omega \to \mathbb{R}$ defined by the conditional variance
\begin{equation}
    \forall \omega \in \Omega :\;
    \mathrm{MV}_X(\omega) := \mathrm{var}(X \mid B_w(\omega)) \,.
\end{equation}
Likewise, $B_w: \Omega \to \mathbb{R}$ is an open ball operator of radius $w$, for some $w > 0$. It is also assumed that $B_w(\omega) \in \mathcal{F}$, for every $\omega \in \Omega$.

### **2.11. Moving Standard Deviation**

Since moving variance is well-defined, moving standard deviation naturally follows from it. Let us designate $\Omega \to \mathbb{R}$ for this discussion with a well-defined metric $d: \Omega \times \Omega \to \mathbb{R}$. Moving standard deviation of a random variable $X \in L^2(P)$ is a map $\mathrm{Mstd}_{X}: \Omega \to \mathbb{R}$ defined by the conditional standard deviation
\begin{equation}
    \forall \omega \in \Omega :\;
    \mathrm{Mstd}_X(\omega) := \mathrm{std}(X \mid B_w(\omega)) \,.
\end{equation}
Likewise, $B_w: \Omega \to \mathbb{R}$ is an open ball operator of radius $w$, for some $w > 0$. It is also assumed that $B_w(\omega) \in \mathcal{F}$, for every $\omega \in \Omega$.


## **3. Model of Time Series**

A time series can be expressed as a map $Y: T \to \mathbb{R}$ where $T$ is the set of time indices, or any totally ordered indices. Naturally, $T$ is discrete. However, continuous treatment on $T$ is also possible. In this framework, we will consider $T$ as a discrete set. In performing forecast, or reconstructing $Y$, it will only be possibly conducted on some subset $T_0 \subset T$ instead of on $T$ itself, except $T$ is infinite. It is essentially true since we will use lag features during the decomposition. Therefore, we will decompose the time series $Y$ on $T_0$ into trend, secondary component, seasonal component and residual which are given by maps $\tau, \sigma, \varsigma, \varepsilon: T_0 \to \mathbb{R}$ respectively such that
\begin{equation}
    \forall t \in T_0:\;
    Y(t) = \tau(t) + \sigma(t) + \varsigma(t) + \varepsilon(t)
\end{equation}
and $\tau, \sigma, \varsigma, \varepsilon \in L^2(T, \mathcal{F}_0, P_0)$. We also designate that given any finite subinterval $S \subset T$, these functions have a property $\tau\big|_{S}, \sigma\big|_{S}, \varsigma\big|_{S}, \varepsilon\big|_{S} \in L^2(S, \mathcal{F}_S, P_S)$ with respect to the probability space $(S, \mathcal{F}_{S}, P_S)$. Note that the probability space $(T, \mathcal{F}, P)$ is the underlying probability of the original problem, and it is also designated that $Y \in L^2(T, \mathcal{F}, P)$. All these designations imply that $Y\big|_{T_0} \in L^2(T_0, \tilde{F}_0, P_0)$ and $Y\big|_S \in L^2(S, \mathcal{F}_S, P_S)$. The model of the time series is given by a map $\hat{Y}: T_0 \to \mathbb{R}$ and is defined by
\begin{equation}
    \forall t \in T_0 :\;
    \hat{Y}(t) := \tau(t) + \sigma(t) + \varsigma(t) \,,
\end{equation}
Importantly, $\varepsilon$ is assumed to be independent and identically distributed (iid) with zero expectation, constant moving variance and having zero autocorrelations. This designated property for $\varepsilon$ is known as a white noise.

It can be inferred that the decomposition of the time series is actually not on $T$, but rather on $T_0$. And it is more of the decomposition of the restriction $Y\big|_{T_0}: T_0 \to \mathbb{R}$ rather than the decomposition of the original $Y: T \to \mathbb{R}$.

### **3.1. Trend Component**

The trend component $\tau: T_0 \to \mathbb{R}$ of the time series $Y\big|_{T_0}: T_0 \to \mathbb{R}$ will be determined in accordance with the moving average $\mathrm{MA}_Y$ of the time series on $T_0$. The trend function $\tau: T_0 \to \mathbb{R}$ will be constructed to model $\mathrm{MA}_{Y\big|_{T_0}}: \tilde{T} \to \mathbb{R}$. The construction can be either using a deterministic model or a stochastic model. If using deterministic a model, one can choose an appropriate polynomial function that best fits $\mathrm{MA}_Y: T \to \mathbb{R}$. However, the choice of polynomial should be taken with a great care since a seemingly fitting polynomial could lead to an extremely misleading value on the far future forecast. Since we emphasize on stochastic model, we will only discuss the stochastic model in detail in this framework.

Note that the index set $T$ is totally ordered and discrete, and so is $T_0$. Then we can also express $T = \{t_k\}_{k = 1}^{|T|}$. One way to construct a stochastic model for $\tau: T_0 \to \mathbb{R}$ is using lag features. Let us introduce a lag operator. Suppose a set $J \subset \mathbb{N}$ is the set of all possible lags such that $\max{J} < |T|$. In fact, $T_0$ is defined by
\begin{equation}
    T_0 := \big\{ t_k \big\}_{k = \max{J}}^{|T|} \,.
\end{equation}
We can also define
\begin{equation}
    \forall j \in J :\; T_j := \big\{ t_k \big\}_{k = \max{J} - j}^{|T| - j} \,.
\end{equation}
The lag operator is given by a map
\begin{equation}
    \forall j \in J :\;
    \kappa_j: L^2(T_0, \mathcal{F}_0, P_0) \to L^2(T_j, \mathcal{F}_j, P_j)
\end{equation}
and is defined by
\begin{equation}
    \forall j \in J \forall f \in L^2(T, \mathcal{F}, P) \forall k \in \{ \max{J}, \max{J} + 1, \dotsc, |T|\} :\;
    (\kappa_j(f))(t_k) := f(t_{k - j})
\end{equation}
It is worth noting that
\begin{equation}
    \forall j \in J :\; \forall f \in L^2(T, \mathcal{F}, P) \forall \tilde{f} \in L^2(T_j, \mathcal{F}_j, P_j) :\;
    \tilde{f} = \chi_{T_j} f \,.
\end{equation}
where
\begin{equation}
    \forall j \in J :\; \chi_{T_j}: T \to \{0, 1\}
    \;\land\;
    \forall j \in J \forall t \in T :\;
    \chi_j(t) :=
    \begin{cases}
        1   &: t \in T_j \\
        0   &: t \notin T_j
    \end{cases} \,.
\end{equation}
It means that we can always express a square integrable function on $T_j$ as a restriction of a square integrable function on $T$, for every $j \in J$.

Then there exists some map $\phi: \mathbb{R}^{|J|} \to \mathbb{R}$ where $\tau: T_0 \to \mathbb{R}$ is defined by
\begin{equation}
    \tau := \phi \circ \big( \kappa_j(\mathrm{MA}_Y) \big)_{j \in J}
\end{equation}
such that for some considerably small $\rho > 0$, $\tau$ is $\rho$-reliable on $\mathrm{MA}_Y$, i. e.,
\begin{equation}
    \Big\| \mathrm{MA}_Y - \tau \Big\|_{2: \mathrm{MA}_Y - \mathrm{E}[ \mathrm{MA}_Y ]}^2 < \rho \,.
\end{equation}
In our approach, we will use a machine learning model to represent the map $\phi: \mathbb{R}^{|J|} \to \mathbb{R}$. The best possible ML frameworks that we are about to use is either `LinearRegression()` from scikit-learn or `LSTM` from `keras.layers`.

### **3.2. Secondary Component**

The secondary component $\sigma: T_0 \to \mathbb{R}$ of the restricted time series $Y\big|_{T_0}: T \to \mathbb{R}$ can be a linear combination of subcomponents such as second order trend and so on. Supposing there are $K$ subcomponents, then the subcomponents can be represented by maps $\sigma_1, \dotsc, \sigma_K: T_0 \to \mathbb{R}$ such that
\begin{equation}
    \forall t \in T_0 :\; \sigma(t) = \sum_{k = 1}^K \alpha_k \sigma_k(t)
\end{equation}
where $\alpha_1, \dotsc, \alpha_K \in \mathbb{R}$. We will model $\sigma_1, \dotsc, \sigma_K$ as stochastic models. It is designated that for every $k \in \{1, \dotsc, K\}$ there exist some set $J_k \subset \mathbb{N}$ and a map $\phi_k: \mathbb{R}^{|J_k|} \to \mathbb{R}$ such that
\begin{equation}
    \sigma_k = \phi_k \circ \left( \kappa_j ( \mathrm{MA}_{\tilde{Y}_k} ) \right)_{j \in J_k}
\end{equation}
where
\begin{equation}
    \tilde{Y}_k := Y\big|_{T_0} - \tau - \sum_{i = 0}^{k - 1} \alpha_i \sigma_i
\end{equation}
with
\begin{equation}
    \forall t \in T_0 :\; \sigma_0(t) := 0 \,.
\end{equation}
Each set $J_k$ represents the set of possible lags for each $\sigma_k$, for every $k \in \{1, \dotsc, K\}$. The maps $\phi_1, \dotsc, \phi_K: \mathbb{R}^{|J_k|} \to \mathbb{R}$ will be represented by machine learning models. Likewise, we will use either `LinearRegression()` or `keras.layers.LSTM`. The nature of the maps $\phi_1, \dotsc, \phi_K$ is the same with that of $\phi$ for trends.

There are also situations where the model of $\sigma_k: T_0 \to \mathbb{R}$ is not best described by lag features. For instance, there is a periodic pattern when we smoothen $\tilde{Y}_k$ into $\mathrm{MA}_{\tilde{Y}_k}$. In this case, the model may be best described by a wave function composed of a linear combination of sinusoidal functions. Let $\psi: \mathbb{R}^{M + 1} \to \mathbb{R}$, $\theta_0, \theta_1, \dotsc, \theta_{M}: T_0 \to \mathbb{R}$ be maps, for some $M \in \mathbb{N}$. We define $\sigma_k$ by
\begin{equation}
    \sigma_k := \psi \circ \Theta
    \quad\land\quad
    \Theta: T_0 \to \mathbb{R}^M
    \quad\land\quad
    \Theta : = ( \sin \circ\, \theta_0, \cos \circ\, \theta_0, \dotsc, \sin \circ\, \theta_{M}, \cos \circ\, \theta_{M} )
    \,.
\end{equation}
The maps $\theta_0, \dotsc, \theta_{M}$ are defined by
\begin{equation}
    \forall n \in \{0, 1, \dotsc, M\} \forall t \in T_0 :\;
    \theta_n(t) := \frac{2 n \pi t}{L}
\end{equation}
where $L \in (0, \infty)$. The map $\psi$ will be represented by a machine learning model. However, the framework of $\psi$ is given by
\begin{equation}
    \forall x_0, y_0, x_1, y_1, \dotsc, x_{M}, y_{M} \in \mathbb{R} :\;
    \psi( x_0, y_0, x_1, y_1, \dotsc, x_{M}, y_{M} )
    := \sum_{n = 0}^{M} a_n x_n + b_n y_n \,,
\end{equation}
where $a_0, b_0, a_1, b_1, \dotsc, a_{M}, b_{M} \in \mathbb{R}$. We will use scikit-learn's `LinearRegerssion` to model $\psi$, and the constants $a_0, b_0, a_1, b_1, \dotsc, a_{M}, b_{M} \in \mathbb{R}$ will be obtained as the regression coefficients of the algorithm. The value of $M$ will be determined in the algorithm of our model, which is given by
\begin{equation}
    M := \underset{m \in \mathcal{M}}{\arg \min} \; \mathrm{E}\big[ \, |\mathrm{MA}_{\tilde{Y}_k} - \sigma_k | \, \big]
\end{equation}
where $\mathcal{M} \subset \mathbb{N}$. The task of the users will be determining $\max{\mathcal{M}}$.

There is a chance that seasonality may not occur in a time series. In this case, it is required for $\sigma: T_0 \to \mathbb{R}$ that $Y_1 - \sigma := (Y\big|_{T_0} - \tau) - \sigma = \varepsilon: T\big|_{T_0} \to \mathbb{R}$ is a white noise.

### **3.3. Seasonal Component**

Seasonality can be defined as a dependence of certain component in time series to a periodic pattern in the time domain. In our approach, we expect an existence of seasonality on $Y_2 := (Y - \tau - \sigma): T_0 \to \mathbb{R}$. Practically, one needs to conduct statistical analysis on $Y_2: T_0 \to \mathbb{R}$ to observe the existence of seasonality such as finding periodic linear relations on $Y_2$. If the result of the statistical analysis suggests the existence of seasonality, then it is necessary to construct additional features. We refer the type seasonality in this formalism as modulo seasonality.

Let $\mathcal{M} \subset \mathbb{N}$. The additional features, referred to as modulo features, are given as maps
\begin{equation}
    \forall p \in \mathcal{M} :\; \mu_p: T_0 \to \mathbb{N}
\end{equation}
defined by
\begin{equation}
    \forall k \in \{ \max{J}, \max{J} + 1, \dotsc, |T|\} \forall p \in \mathcal{M} :\;
    \mu_p(t_k) := k - \left\lfloor \frac{k}{p} \right\rfloor p \,.
\end{equation}
The set $\mathcal{M}$ is the set of all possible periods. It shall satisfy
\begin{equation}
    \exists \delta > 0 \forall p \in \mathcal{M} :\;
    \big| \mathrm{corr}( Y_2, \mu_p) \big| > \delta
\end{equation}
The value of $\delta$ shall be determined by the users. Then there exists some map $\eta: \mathbb{R}^{|\mathcal{M}|} \to \mathbb{R}$ such that
\begin{equation}
    \varsigma = \eta \circ \big( \mu_p \big)_{p \in \mathcal{M}} \,.
\end{equation}
The map $\eta: \mathbb{R}^M \to \mathbb{R}$ will be represented by a machine learning model using either `LinearRegression()` or `keras.layers.LSTM`.

It is required that the residual $\varepsilon = Y\big|_{T_0} - \tau - \sigma - \varsigma: T_0 \to \mathbb{R}$ is a white noise.

### **3.4. Weak White Noise**

We have mentioned that a white noise is iid with zero expectation, constant moving variance and zero autocorrelation. All these three characterizations will be used to identify white noise. In practices, it might be very difficult to find that $\varepsilon$ is a perfect white noise. Therefore, for an efficient computation, we introduce the notion of weak white noise such that $\varepsilon$ is required to be a weak white noise instead of a perfect white noise.

We define a map $\nu: T_0 \to \mathbb{R}$ to be a weak white noise is and only if there exists some considerably small $\delta_1, \delta_2, \delta_3 > 0$ such that the following properties hold:
1. $\big| \mathrm{E}[\nu] \big| < \delta_1$
2. $\exists c > 0 \forall t \in T_0 :\; \mathrm{MV}_{\nu}(t) \in B_{\delta_2}(c)$
3. $\forall j \in \{1, \dotsc, |T|\}:\; \big| \mathrm{corr}(\nu\big|_{T_j}, \kappa_j(\nu)) \big| < \delta_3$

Then the triple $(\delta_1, \delta_2, \delta_3)$ will be referred to as the weak level of $\nu$.


## **References**
- Stoll, Robert R. (1979). *Set Theory and Logic*. Dover Publications.
- Salamon, Dietmar A. (2016). *Measure and Integration*. European Mathematical Society.
- Bremaud, Pierre. (2020). *Probability Theory and Stochastic Processes*. Springer.
- Andre, Robert. (2024). *Point-set Topology with Topics: Basic General Topology For Graduate Studies*. World Scientific Publishing Co Pte Ltd.
