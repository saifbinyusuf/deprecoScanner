# Phase 5 Ground-Truth Benchmark Review ($N = 150$)

This document contains all 150 call-site items in the frozen evaluation benchmark.
Ground truth `is_deprecated_call` is established under human authority.

## Summary Statistics
- **Total Benchmark Items**: 150
- **True Deprecations (Expected)**: 108 (72.00%)
- **True Benign (Expected)**: 42 (28.00%)
- **Candidate Positives**: 100
- **Pipeline Misses**: 8 (4 `errprint`, 4 `rvs_ratio_uniforms`)
- **Label Anomalies**: 2 (`scipy_1560`, `scipy_1500`)
- **Fresh Hard Negatives**: 40 (16 replacements, 8 submodules, 8 stdlib, 8 wrappers)

---

## All Benchmark Items

### `bench_001` — `scipy.linalg.pinv2` (scipy)
- **Sample ID**: `scipy_1235` | **Line**: 11:25 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `predicted_views.append(predicted_target @ pinv2(self.weights_list[i]))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1235)</summary>

```python
      1 |     def predict_view(self, *args):
      2 |         # Regress original given views onto target
      3 |         transformed_views = self.transform_view(*args)
      4 | 
      5 |         # Get the regression from the training data with available views
      6 |         predicted_target = np.mean([transformed_views[i] for i in range(len(args)) if args[i] is not None], axis=0)
      7 | 
      8 |         predicted_views = []
      9 |         for i, view in enumerate(args):
     10 |             if view is None:
-->  11 |                 predicted_views.append(predicted_target @ pinv2(self.weights_list[i]))
     12 |             else:
     13 |                 predicted_views.append(view)
     14 |         for i, predicted_view in enumerate(predicted_views):
     15 |             predicted_views[i] += self.dataset_means[i]
     16 |         return predicted_views
```

</details>

---

### `bench_002` — `scipy.misc.face` (scipy)
- **Sample ID**: `scipy_2150` | **Line**: 1:7 | **Stratum**: `hard_negative` (`misc_non_benchmark_symbol`)
- **Call Site**: `gray = scipy.misc.ascent()`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: non-benchmark function ascent.*

<details>
<summary>View Enclosing Code (scipy_2150)</summary>

```python
-->   1 | gray = scipy.misc.ascent()
      2 | color = scipy.misc.face()
```

</details>

---

### `bench_003` — `scipy.integrate.cumtrapz` (scipy)
- **Sample ID**: `scipy_1500` | **Line**: 40:19 | **Stratum**: `label_anomaly` (`alias_declaration_anomaly`)
- **Call Site**: `ret.data = _sp_cumtrapz(arr, crd_arr.reshape(-1), axis=fld_axis, initial=initial)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Documented anomaly: internal alias declaration inside method body.*

<details>
<summary>View Enclosing Code (scipy_1500)</summary>

```python
      1 |     def cumtrapz(self, axis=-1, fudge_factor=None, initial=0):
      2 |         """Cumulatively integrate field over a single axis
      3 | 
      4 |         Args:
      5 |             axis (str, int): axis name or index
      6 |             fudge_factor (callable): function that is called with
      7 |                 func(data, crd_arr), where crd_arr is shaped. This is
      8 |                 useful for including parts of the jacobian, like
      9 |                 sin(theta) dtheta.
     10 |             initial (float): Initial value
     11 | 
     12 |         Returns:
     13 |             Field
     14 |         """
     15 |         ret = None
     16 | 
     17 |         try:
     18 |             from scipy.integrate import cumtrapz as _sp_cumtrapz
     19 | 
     20 |             if axis in self.crds.axes:
     21 |                 axis = self.crds.axes.index(axis)
     22 |             assert isinstance(axis, (int, np.integer))
     23 |             crd_arr = self.get_crd(axis, shaped=True)
     24 | 
     25 |             try:
     26 |                 crd_arr = np.expand_dims(crd_arr, axis=self.nr_comp)
     27 |                 if self.nr_comp > axis:
     28 |                     fld_axis = axis
     29 |                 else:
     30 |                     fld_axis = axis + 1
     31 |             except TypeError:
     32 |                 fld_axis = axis
     33 | 
     34 |             if fudge_factor is None:
     35 |                 arr = self.data
     36 |             else:
     37 |                 arr = self.data * fudge_factor(self.data, crd_arr)
     38 | 
     39 |             ret = viscid.empty_like(self)
-->  40 |             ret.data = _sp_cumtrapz(arr, crd_arr.reshape(-1), axis=fld_axis,
     41 |                                     initial=initial)
     42 |         except ImportError:
     43 |             viscid.logger.error("Scipy is required to perform cumtrapz")
     44 |             raise
     45 | 
     46 |         return ret
```

</details>

---

### `bench_004` — `numpy.product` (numpy)
- **Sample ID**: `numpy_1576` | **Line**: 26:18 | **Stratum**: `hard_negative` (`stdlib_name_collision_product`)
- **Call Site**: `values = list(itertools.product(*list(kwargs.values())))`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: itertools.product standard library collision.*

<details>
<summary>View Enclosing Code (numpy_1576)</summary>

```python
      1 |     def product(self, **kwargs:dict)->list:
      2 |         """Cartesian product of input variables.
      3 |         This method is inspired by `itertools.product <https://docs.python.org/3/library/itertools.html#itertools.product>`_.
      4 | 
      5 |         Must pass a list for each ``sampling_var`` that should be considered. Not all ``sampling_vars`` must be referenced. 
      6 |         Sampling vars that are excluded, will generate a value according to their assigned ``fun_var_pdf`` (see :py:meth:`set_sampling_var`).
      7 | 
      8 |         Args:
      9 |             kwargs: Keyword arguments of the form ``var_name=var_values``.
     10 | 
     11 |         Returns:
     12 |             Returns the newly created sampling plan.
     13 |         """
     14 |         # Check if all key word values are lists:
     15 |         check = np.alltrue([isinstance(v, list) for v in kwargs.values()])
     16 |         if not check:
     17 |             raise ValueError('keyword values must be lists')
     18 | 
     19 |         # Check if all key words are existing sampling variables:
     20 |         keys = kwargs.keys()
     21 |         check = np.alltrue([v in self.sampling_var_names for v in keys])
     22 |         if not check:
     23 |             raise ValueError('keyword names must be existing sampling variables')
     24 | 
     25 |         # Create cartesian product of all values passen in kwargs:
-->  26 |         values = list(itertools.product(*list(kwargs.values())))
     27 | 
     28 |         # Create new sampling cases:
     29 |         for value in values:
     30 |             # Zip together the value(s) of the current case with the respective keys:
     31 |             case = dict(zip(keys, value))
     32 |             # Add sampling case
     33 |             self.add_sampling_case(**case)
     34 | 
     35 |         return self.sampling_plan
```

</details>

---

### `bench_005` — `scipy.signal.hanning` (scipy)
- **Sample ID**: `scipy_306` | **Line**: 9:21 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `window = scipy.signal.hanning(len(series))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_306)</summary>

```python
      1 | def WindowDataSeries(series, window=None):
      2 | 
      3 |   """
      4 |     Apply window function to data set, defaults to Hanning window.
      5 |   """
      6 | 
      7 |   # generate default window
      8 |   if window == None:
-->   9 |     window = scipy.signal.hanning(len(series))
     10 | 
     11 |   # check dimensions
     12 |   assert len(series)==len(window), 'Window and data must be same shape'
     13 | 
     14 |   # get sum of squares
     15 |   sumofsquares = (window**2).sum()
     16 |   assert sumofsquares > 0, 'Sum of squares of window non-positive.'
     17 | 
     18 |   # generate norm
     19 |   norm = (len(window)/sumofsquares)**(1/2)
     20 | 
     21 |   # apply window
     22 |   return series * window * norm
```

</details>

---

### `bench_006` — `scipy.interpolate.interp2d` (scipy)
- **Sample ID**: `scipy_1144` | **Line**: 18:6 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `vi = intp.interp2d(G['lon_v'][0, :], G['lat_v'][:, 0], v)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1144)</summary>

```python
      1 | def add_velocity_vectors(ax, aa, ds, fn, v_scl=3, nngrid=80, zlev='top'):
      2 |     # v_scl: scale velocity vector (smaller to get longer arrows)
      3 |     # GET DATA
      4 |     G = zrfun.get_basic_info(fn, only_G=True)
      5 |     if zlev == 'top':
      6 |         u = ds['u'][0, -1, :, :].values
      7 |         v = ds['v'][0, -1, :, :].values
      8 |     elif zlev == 'bot':
      9 |         u = ds['u'][0, 0, :, :].values
     10 |         v = ds['v'][0, 0, :, :].values
     11 |     # ADD VELOCITY VECTORS
     12 |     # set masked values to 0
     13 |     u[np.isnan(u)]=0
     14 |     v[np.isnan(v)]=0
     15 |     # create interpolant
     16 |     import scipy.interpolate as intp
     17 |     ui = intp.interp2d(G['lon_u'][0, :], G['lat_u'][:, 0], u)
-->  18 |     vi = intp.interp2d(G['lon_v'][0, :], G['lat_v'][:, 0], v)
     19 |     # create regular grid
     20 |     daax = aa[1] - aa[0]
     21 |     daay = aa[3] - aa[2]
     22 |     axrat = np.cos(np.deg2rad(aa[2])) * daax / daay
     23 |     x = np.linspace(aa[0], aa[1], int(round(nngrid * axrat)))
     24 |     y = np.linspace(aa[2], aa[3], int(nngrid))
     25 |     xx, yy = np.meshgrid(x, y)
     26 |     # interpolate to regular grid
     27 |     uu = ui(x, y)
     28 |     vv = vi(x, y)
     29 |     mask = uu != 0
     30 |     # plot velocity vectors
     31 |     ax.quiver(xx[mask], yy[mask], uu[mask], vv[mask],
     32 |         units='y', scale=v_scl, scale_units='y', color='b')
```

</details>

---

### `bench_007` — `scipy.linalg.pinv2` (scipy)
- **Sample ID**: `scipy_1283` | **Line**: 14:20 | **Stratum**: `hard_negative` (`submodule_non_deprecated_pinv`)
- **Call Site**: `sig_t_inv = scipy.linalg.pinv(sig_t, rtol=rtol, atol=0)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: non-deprecated standard pinv, not pinv2.*

<details>
<summary>View Enclosing Code (scipy_1283)</summary>

```python
      1 |     def _invert_sig_t(sig_t: np.ndarray, lamb: float, rtol: float) -> np.ndarray:
      2 |         """Invert the correlation matrix. If the provided regularization values are not enough to stabilize the inversion process for the given matrix, the function calls itself recursively, increasing lamb and rtol by 10%.
      3 | 
      4 |         Args:
      5 |             sig_t (np.ndarray): the correlation matrix
      6 |             lamb (float): regularization term added to the diagonal of the sig_t matrix
      7 |             rtol (float): threshold to filter eigenvector with a eigenvalue under rtol make inversion biased but much more numerically robust
      8 | 
      9 |         Returns:
     10 |             np.ndarray: the inverse of the correlation matrix
     11 |         """
     12 |         try:
     13 |             np.fill_diagonal(sig_t, (1 + lamb))
-->  14 |             sig_t_inv = scipy.linalg.pinv(sig_t, rtol=rtol, atol=0)
     15 |             return sig_t_inv
     16 |         except np.linalg.LinAlgError:
     17 |             return SummaryStatisticsImputation._invert_sig_t(
     18 |                 sig_t, lamb * 1.1, rtol * 1.1
     19 |             )
```

</details>

---

### `bench_008` — `scipy.misc.logsumexp` (scipy)
- **Sample ID**: `scipy_588` | **Line**: 11:0 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `res = scipy.misc.logsumexp(ps + np.log(self.a), axis=1) if log else np.dot(ps, self.a)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_588)</summary>

```python
      1 |     def eval(self, x, ii=None, log=True):
      2 |         """
      3 |         Evaluates the mog pdf.
      4 |         :param x: rows are inputs to evaluate at
      5 |         :param ii: a list of indices specifying which marginal to evaluate. if None, the joint pdf is evaluated
      6 |         :param log: if True, the log pdf is evaluated
      7 |         :return: pdf or log pdf
      8 |         """
      9 | 
     10 |         ps = np.array([self.xs[ix].eval(x, ii, log) for ix in range(len(self.a))]).T
-->  11 |         res = scipy.misc.logsumexp(ps + np.log(self.a), axis=1) if log else np.dot(ps, self.a)
     12 | 
     13 |         return res
```

</details>

---

### `bench_009` — `scipy.misc.logsumexp` (scipy)
- **Sample ID**: `scipy_555` | **Line**: 23:26 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `log_denominator = logsumexp(values, dim=dim, keepdim=True)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_555)</summary>

```python
      1 | def lognormexp(values, dim=0):
      2 |     """Log of exponentiates and normalizes a Tensor/Variable/np.ndarray.
      3 | 
      4 |     input:
      5 |         values: Tensor/Variable/np.ndarray [dim_1, ..., dim_N]
      6 |         dim: n
      7 |     output:
      8 |         result: Tensor/Variable/np.ndarray [dim_1, ..., dim_N]
      9 |             where result[i_1, ..., i_N] =
     10 | 
     11 |                             exp(values[i_1, ..., i_N])
     12 |             ------------------------------------------------------------
     13 |              sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
     14 |     """
     15 | 
     16 |     if isinstance(values, np.ndarray):
     17 |         log_denominator = scipy.misc.logsumexp(
     18 |             values, axis=dim, keepdims=True
     19 |         )
     20 |         # log_numerator = values
     21 |         return values - log_denominator
     22 |     else:
-->  23 |         log_denominator = logsumexp(values, dim=dim, keepdim=True)
     24 |         # log_numerator = values
     25 |         return values - log_denominator.expand_as(values)
```

</details>

---

### `bench_010` — `pandas.Series.pad` (pandas)
- **Sample ID**: `pandas_66` | **Line**: 9:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `pser.pad(inplace=True)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_66)</summary>

```python
      1 |     def test_pad(self):
      2 |         pser = pd.Series([np.nan, 2, 3, 4, np.nan, 6], name="x")
      3 |         kser = ks.from_pandas(pser)
      4 | 
      5 |         if LooseVersion(pd.__version__) >= LooseVersion("1.1"):
      6 |             self.assert_eq(pser.pad(), kser.pad())
      7 | 
      8 |             # Test `inplace=True`
-->   9 |             pser.pad(inplace=True)
     10 |             kser.pad(inplace=True)
     11 |             self.assert_eq(pser, kser)
     12 |         else:
     13 |             expected = ks.Series([np.nan, 2, 3, 4, 4, 6], name="x")
     14 |             self.assert_eq(expected, kser.pad())
     15 | 
     16 |             # Test `inplace=True`
     17 |             kser.pad(inplace=True)
     18 |             self.assert_eq(expected, kser)
```

</details>

---

### `bench_011` — `scipy.misc.factorial` (scipy)
- **Sample ID**: `scipy_1564` | **Line**: 2:0 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `a = scipy.misc.factorial(N)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1564)</summary>

```python
      1 | def getBC(N,k):
-->   2 |     a = scipy.misc.factorial(N)
      3 |     b = scipy.misc.factorial(N-k)
      4 |     c = scipy.misc.factorial(k)
      5 |     return a/(b*c)
```

</details>

---

### `bench_012` — `scipy.stats.chisqprob` (scipy)
- **Sample ID**: `scipy_1674` | **Line**: 16:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `return 0.5 * scipy.stats.chisqprob(s, 1) + 0.5 * int(s == 0)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1674)</summary>

```python
      1 | def install(download_dir):
      2 |     local_filename = "%s/MEGSA_beta.zip" % download_dir
      3 |     if not os.path.exists(local_filename):
      4 |         filename, response = urllib.urlretrieve(MEGSA_URL, local_filename)
      5 |     else:
      6 |         filename = local_filename
      7 |     nbsupport.util.check_digest(filename, "655e1ec48a67530672303d8ac7fbc925")
      8 | 
      9 |     with zipfile.ZipFile(local_filename) as archive:
     10 |         with io.TextIOWrapper(archive.open("version beta/MEGSA.R")) as stream:
     11 |             robjects.reval(stream.read())
     12 | 
     13 |     global megsa
     14 |     def megsa(events):
     15 |         s = robjects.r.funEstimate(events.T).rx2("S")[0]
-->  16 |         return 0.5 * scipy.stats.chisqprob(s, 1) + 0.5 * int(s == 0)
```

</details>

---

### `bench_013` — `numpy.alltrue` (numpy)
- **Sample ID**: `numpy_1500` | **Line**: 11:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `homogenous_orientations = numpy.alltrue(orientations == -1) or numpy.alltrue(orientations == 1)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (numpy_1500)</summary>

```python
      1 | def _should_allow_reverse(contours, allow_reflection):
      2 |     # If we are to allow for reflections, we ought to allow for reversing
      3 |     # orientations too, because even if the contours start out oriented in the
      4 |     # same direction, reflections can change that.
      5 |     # Then check if all of the contours are oriented in the same direction.
      6 |     # If they're not, we need to allow for reversing their orientation in the
      7 |     # alignment process.
      8 |     if allow_reflection:
      9 |         return True
     10 |     orientations = numpy.array([numpy.sign(contour.signed_area()) for contour in contours])
-->  11 |     homogenous_orientations = numpy.alltrue(orientations == -1) or numpy.alltrue(orientations == 1)
     12 |     return not homogenous_orientations
```

</details>

---

### `bench_014` — `numpy.product` (numpy)
- **Sample ID**: `numpy_3439` | **Line**: 49:68 | **Stratum**: `hard_negative` (`stdlib_name_collision_product`)
- **Call Site**: `np.repeat(np.asarray(com.values_from_object(x)), b[i]), np.product(a[i])`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: itertools.product standard library collision.*

<details>
<summary>View Enclosing Code (numpy_3439)</summary>

```python
      1 | def cartesian_product(X):
      2 |     """
      3 |     Numpy version of itertools.product.
      4 |     Sometimes faster (for large inputs)...
      5 | 
      6 |     Parameters
      7 |     ----------
      8 |     X : list-like of list-likes
      9 | 
     10 |     Returns
     11 |     -------
     12 |     product : list of ndarrays
     13 | 
     14 |     Examples
     15 |     --------
     16 |     >>> cartesian_product([list('ABC'), [1, 2]])
     17 |     [array(['A', 'A', 'B', 'B', 'C', 'C'], dtype='|S1'),
     18 |     array([1, 2, 1, 2, 1, 2])]
     19 | 
     20 |     See Also
     21 |     --------
     22 |     itertools.product : Cartesian product of input iterables.  Equivalent to
     23 |         nested for-loops.
     24 |     """
     25 |     msg = "Input must be a list-like of list-likes"
     26 |     if not is_list_like(X):
     27 |         raise TypeError(msg)
     28 |     for x in X:
     29 |         if not is_list_like(x):
     30 |             raise TypeError(msg)
     31 | 
     32 |     if len(X) == 0:
     33 |         return []
     34 | 
     35 |     lenX = np.fromiter((len(x) for x in X), dtype=np.intp)
     36 |     cumprodX = np.cumproduct(lenX)
     37 | 
     38 |     a = np.roll(cumprodX, 1)
     39 |     a[0] = 1
     40 | 
     41 |     if cumprodX[-1] != 0:
     42 |         b = cumprodX[-1] / cumprodX
     43 |     else:
     44 |         # if any factor is empty, the cartesian product is empty
     45 |         b = np.zeros_like(cumprodX)
     46 | 
     47 |     return [
     48 |         np.tile(
-->  49 |             np.repeat(np.asarray(com.values_from_object(x)), b[i]), np.product(a[i])
     50 |         )
     51 |         for i, x in enumerate(X)
     52 |     ]
```

</details>

---

### `bench_015` — `scipy.misc.factorial2` (scipy)
- **Sample ID**: `scipy_1927` | **Line**: 15:23 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `c = 1.0 * factorial2(n-3)**2 / np.sqrt(2*np.pi*np.math.factorial(n))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1927)</summary>

```python
      1 |     def get_initializations(self, num_pol = 5, copy_fun = 'relu'):
      2 |         k = []
      3 |         if copy_fun == 'relu':
      4 |             for n in range(num_pol):
      5 |                 if n == 0:
      6 |                     k.append(1.0/np.sqrt(2*np.pi))
      7 |                     #k.append(0.0)
      8 |                 elif n == 1:
      9 |                     k.append(1.0/2)
     10 |                     #k.append(0.0)
     11 |                 elif n == 2:
     12 |                     k.append(1.0/np.sqrt(4*np.pi))
     13 |                     #k.append(0.0)
     14 |                 elif n > 2 and n % 2 == 0:
-->  15 |                     c = 1.0 * factorial2(n-3)**2 / np.sqrt(2*np.pi*np.math.factorial(n))
     16 |                     k.append(c)
     17 |                     #k.append(0.0)
     18 |                 elif n >= 2 and n % 2 != 0:
     19 |                     k.append(0.0)
     20 |         return k
```

</details>

---

### `bench_016` — `pandas.DataFrame.first` (pandas)
- **Sample ID**: `pandas_89` | **Line**: 6:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `self.assert_eq(pdf.first(DateOffset(days=1)), kdf.first(DateOffset(days=1)))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_89)</summary>

```python
      1 |     def test_first(self):
      2 |         index = pd.date_range("2018-04-09", periods=4, freq="2D")
      3 |         pdf = pd.DataFrame([1, 2, 3, 4], index=index)
      4 |         kdf = ks.from_pandas(pdf)
      5 |         self.assert_eq(pdf.first("1D"), kdf.first("1D"))
-->   6 |         self.assert_eq(pdf.first(DateOffset(days=1)), kdf.first(DateOffset(days=1)))
      7 |         with self.assertRaisesRegex(TypeError, "'first' only supports a DatetimeIndex"):
      8 |             ks.DataFrame([1, 2, 3, 4]).first("1D")
```

</details>

---

### `bench_017` — `scipy.misc.face` (scipy)
- **Sample ID**: `scipy_2166` | **Line**: 7:5 | **Stratum**: `hard_negative` (`misc_non_benchmark_symbol`)
- **Call Site**: `im = scipy.misc.imresize(scipy.misc.face(), args_.image_size)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: non-benchmark function imresize.*

<details>
<summary>View Enclosing Code (scipy_2166)</summary>

```python
      1 | net,flatten_loc = pl.load_pytorch_model(args_.pytorch_model, paths=paths)
      2 | params = net.state_dict()
      3 | 
      4 | # forward pass to compute pytorch feature sizes
      5 | args_.image_size = tuple(make_tuple(args_.image_size))
      6 | args_.full_image_size = tuple(make_tuple(args_.full_image_size))
-->   7 | im = scipy.misc.imresize(scipy.misc.face(), args_.image_size)
```

</details>

---

### `bench_018` — `pandas.DataFrame.applymap` (pandas)
- **Sample ID**: `pandas_118` | **Line**: 6:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `assert_eq(ddf.applymap(lambda x: (x, x)), df.applymap(lambda x: (x, x)))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_118)</summary>

```python
      1 | def test_applymap():
      2 |     df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [10, 20, 30, 40]})
      3 |     ddf = dd.from_pandas(df, npartitions=2)
      4 |     assert_eq(ddf.applymap(lambda x: x + 1), df.applymap(lambda x: x + 1))
      5 | 
-->   6 |     assert_eq(ddf.applymap(lambda x: (x, x)), df.applymap(lambda x: (x, x)))
```

</details>

---

### `bench_019` — `scipy.stats.betai` (scipy)
- **Sample ID**: `scipy_470` | **Line**: 15:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `prob = [scipy.stats.betai(0.5*df,0.5,df/(df+tsq)) if tsq is not ma.masked and df/(df+tsq) <= 1.0 else ma.masked  for tsq in t*t]`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_470)</summary>

```python
      1 | def attest_ind(a, b, dim=None):
      2 |     """ Return the t-test statistics on arrays a and b over the dim axis.
      3 |     Returns both the t statistic as well as the p-value
      4 |     """
      5 | #    dim = a.ndim - 1 if dim is None else dim
      6 |     x1, x2 = ma.mean(a, dim), ma.mean(b, dim)
      7 |     v1, v2 = ma.var(a, dim), ma.var(b, dim)
      8 |     n1, n2 = (a.shape[dim], b.shape[dim]) if dim is not None else (a.size, b.size)
      9 |     df = float(n1+n2-2)
     10 |     svar = ((n1-1)*v1+(n2-1)*v2) / df
     11 |     t = (x1-x2)/ma.sqrt(svar*(1.0/n1 + 1.0/n2))
     12 |     if t.ndim == 0:
     13 |         return (t, scipy.stats.betai(0.5*df,0.5,df/(df+t**2)) if t is not ma.masked and df/(df+t**2) <= 1.0 else ma.masked)
     14 |     else:
-->  15 |         prob = [scipy.stats.betai(0.5*df,0.5,df/(df+tsq)) if tsq is not ma.masked and df/(df+tsq) <= 1.0 else ma.masked  for tsq in t*t]
     16 |         return t, prob
```

</details>

---

### `bench_020` — `pandas.DataFrame.first` (pandas)
- **Sample ID**: `pandas_83` | **Line**: 10:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `df_equals(modin_df.first("20D"), pandas_df.first("20D"))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_83)</summary>

```python
      1 | def test_first():
      2 |     i = pd.date_range("2010-04-09", periods=400, freq="2D")
      3 |     modin_df = pd.DataFrame({"A": list(range(400)), "B": list(range(400))}, index=i)
      4 |     pandas_df = pandas.DataFrame(
      5 |         {"A": list(range(400)), "B": list(range(400))}, index=i
      6 |     )
      7 |     with pytest.warns(FutureWarning, match="first is deprecated and will be removed"):
      8 |         modin_result = modin_df.first("3D")
      9 |     df_equals(modin_result, pandas_df.first("3D"))
-->  10 |     df_equals(modin_df.first("20D"), pandas_df.first("20D"))
```

</details>

---

### `bench_021` — `scipy.integrate.cumtrapz` (scipy)
- **Sample ID**: `scipy_1517` | **Line**: 6:32 | **Stratum**: `hard_negative` (`replacement_overload_integrate`)
- **Call Site**: `series = np.pi / 2 / 9.81 * cumulative_trapezoid(`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: modern replacement cumulative_trapezoid.*

<details>
<summary>View Enclosing Code (scipy_1517)</summary>

```python
      1 |     def get_ia(self):
      2 |         """return Arias intensity Ia.
      3 |         """
      4 |         # time history of Arias Intensity
      5 |         acc = self.acc * self.unit_factors[f'{self.acc_unit}-m']
-->   6 |         series = np.pi / 2 / 9.81 * cumulative_trapezoid(
      7 |             acc ** 2, self.time, initial=0)
      8 |         # Total Arias Intensity at the end of the ground motion
      9 |         arias = series[-1]
     10 |         # time history of the normalized Arias Intensity
     11 |         arias_percent = series / arias
     12 |         # ----------------------------------------------------
     13 |         self.Arias = arias
     14 |         self.AriasSeries = series
     15 |         self.AriasPercent = arias_percent
     16 |         return arias
```

</details>

---

### `bench_022` — `scipy.misc.face` (scipy)
- **Sample ID**: `scipy_1145` | **Line**: 22:15 | **Stratum**: `hard_negative` (`misc_non_benchmark_symbol`)
- **Call Site**: `return scipy.misc.imresize(D, dims)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: non-benchmark function imresize.*

<details>
<summary>View Enclosing Code (scipy_1145)</summary>

```python
      1 | def imresize(D, dims, kind='cubic', use_scipy=False):
      2 |     """
      3 |     Resize a floating point image
      4 |     Parameters
      5 |     ----------
      6 |     D : ndarray(M1, N1)
      7 |         Original image
      8 |     dims : tuple(M2, N2)
      9 |         The dimensions to which to resize
     10 |     kind : string
     11 |         The kind of interpolation to use
     12 |     use_scipy : boolean
     13 |         Fall back to scipy.misc.imresize.  This is a bad idea
     14 |         because it casts everything to uint8, but it's what I
     15 |         was doing accidentally for a while
     16 |     Returns
     17 |     -------
     18 |     D2 : ndarray(M2, N2)
     19 |         A resized array
     20 |     """
     21 |     if use_scipy:
-->  22 |         return scipy.misc.imresize(D, dims)
     23 |     else:
     24 |         M, N = dims
     25 |         x1 = np.array(0.5 + np.arange(D.shape[1]), dtype=np.float32)/D.shape[1]
     26 |         y1 = np.array(0.5 + np.arange(D.shape[0]), dtype=np.float32)/D.shape[0]
     27 |         x2 = np.array(0.5 + np.arange(N), dtype=np.float32)/N
     28 |         y2 = np.array(0.5 + np.arange(M), dtype=np.float32)/M
     29 |         f = scipy.interpolate.interp2d(x1, y1, D, kind=kind)
     30 |         return f(x2, y2)
```

</details>

---

### `bench_023` — `scipy.signal.hanning` (scipy)
- **Sample ID**: `scipy_295` | **Line**: 18:21 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `ps = ps*scipy.signal.hanning(ps.shape[0])[:,None]*scipy.signal.hanning(ps.shape[1])[None,:]`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_295)</summary>

```python
      1 | def preparePSF(md, PSSize):
      2 |     global PSFFileName, cachedPSF, cachedOTF2, cachedOTFH, autocorr
      3 |     
      4 |     PSFFilename = md['PSFFile']
      5 |                 
      6 |     if (not (PSFFileName == PSFFilename)) or (not (cachedPSF.shape == PSSize)):
      7 |         try:
      8 |             ps, vox = md['taskQueue'].getQueueData(md['dataSourceID'], 'PSF')
      9 |         except:
     10 |             #fid = open(getFullExistingFilename(PSFFilename), 'rb')
     11 |             #ps, vox = pickle.load(fid)
     12 |             #fid.close()
     13 |             ps, vox = load_psf(PSFFilename)
     14 |             
     15 |         ps = ps.max(2)
     16 |         ps = ps - ps.min()
     17 |         #ps = ps*(ps > 0)
-->  18 |         ps = ps*scipy.signal.hanning(ps.shape[0])[:,None]*scipy.signal.hanning(ps.shape[1])[None,:]
     19 |         ps = ps/ps.sum()
     20 |         PSFFileName = PSFFilename
     21 |         pw = (numpy.array(PSSize) - ps.shape)/2.
     22 |         pw1 = numpy.floor(pw)
     23 |         pw2 = numpy.ceil(pw)
     24 |         cachedPSF = pad.with_constant(ps, ((pw2[0], pw1[0]), (pw2[1], pw1[1])), (0,))
     25 |         cachedOTFH = ifftn(cachedPSF)*cachedPSF.size
     26 |         cachedOTF2 = cachedOTFH*fftn(cachedPSF)
```

</details>

---

### `bench_024` — `pandas.DataFrame.applymap` (pandas)
- **Sample ID**: `pandas_120` | **Line**: 28:20 | **Stratum**: `hard_negative` (`replacement_overload_pandas`)
- **Call Site**: `table = table.map(escape_value)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: modern replacement map.*

<details>
<summary>View Enclosing Code (pandas_120)</summary>

```python
      1 |     def render_table(table, columns=None, hide_index=True):
      2 |         """
      3 |         Renders a table-like object as an HTML table.
      4 | 
      5 |         Args:
      6 |             table: Table-like object (e.g. pandas DataFrame, 2D numpy array, list of tuples).
      7 |             columns: Column names to use. If `table` doesn't have column names, this argument
      8 |                 provides names for the columns. Otherwise, only the specified columns will be
      9 |                 included in the output HTML table.
     10 |             hide_index: Hide index column when rendering.
     11 |         """
     12 |         import pandas as pd
     13 |         from pandas.io.formats.style import Styler
     14 | 
     15 |         pandas_version = Version(pd.__version__)
     16 | 
     17 |         if not isinstance(table, Styler):
     18 |             table = pd.DataFrame(table, columns=columns)
     19 |             # Escape specific characters in HTML to prevent
     20 |             # javascript code injection
     21 |             # Note that `pandas_df.style.to_html(escape=True) does not work
     22 |             # So that we have to manually escape values in dataframe cells.
     23 | 
     24 |             def escape_value(x):
     25 |                 return html.escape(str(x))
     26 | 
     27 |             if hasattr(table, "map"):
-->  28 |                 table = table.map(escape_value)
     29 |             else:
     30 |                 if pandas_version >= Version("2.1.0"):
     31 |                     table = table.map(escape_value)
     32 |                 else:
     33 |                     table = table.applymap(escape_value)
     34 |             table = table.style
     35 | 
     36 |         styler = table.set_table_attributes('style="border-collapse:collapse"').set_table_styles(
     37 |             [
     38 |                 {
     39 |                     "selector": "table, th, td",
     40 |                     "props": [
     41 |                         ("border", "1px solid grey"),
     42 |                         ("text-align", "left"),
     43 |                         ("padding", "5px"),
     44 |                     ],
     45 |                 },
     46 |             ]
     47 |         )
     48 |         if hide_index:
     49 |             rendered_table = (
     50 |                 styler.hide(axis="index").to_html()
     51 |                 if pandas_version >= Version("1.4.0")
     52 |                 else styler.hide_index().render()
     53 |             )
     54 |         else:
     55 |             rendered_table = (
     56 |                 styler.to_html() if pandas_version >= Version("1.4.0") else styler.render()
     57 |             )
     58 |         return f'<div style="max-height: 500px; overflow: scroll;">{rendered_table}</div>'
```

</details>

---

### `bench_025` — `scipy.linalg.pinv2` (scipy)
- **Sample ID**: `scipy_1292` | **Line**: 18:25 | **Stratum**: `hard_negative` (`submodule_non_deprecated_pinv`)
- **Call Site**: `b = scipy.dot(scipy.dot(scipy.linalg.pinv(scipy.dot(scipy.transpose(X),X)),scipy.transpose(X)),y)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: non-deprecated standard pinv, not pinv2.*

<details>
<summary>View Enclosing Code (scipy_1292)</summary>

```python
      1 | def mlr(x,y,order):
      2 | 	"""Multiple linear regression fit of the columns of matrix x
      3 | 	(dependent variables) to constituent vector y (independent variables)
      4 | 	order -     order of a smoothing polynomial, which can be included
      5 | 	in the set of independent variables. If order is
      6 | 	not specified, no background will be included.
      7 | 	b -         fit coeffs
      8 | 	f -         fit result (m x 1 column vector)
      9 | 	r -         residual   (m x 1 column vector)
     10 | 	"""
     11 | 	if order > 0:
     12 | 		s=scipy.ones((len(y),1))
     13 | 		for j in range(order):
     14 | 			s=scipy.concatenate((s,(scipy.arange(0,1+(1.0/(len(y)-1))-0.5/(len(y)-1),1.0/(len(y)-1))**j)[:,nA]),1)
     15 | 		X=scipy.concatenate((x, s),1)
     16 | 	else:
     17 | 		X = x
-->  18 | 	b = scipy.dot(scipy.dot(scipy.linalg.pinv(scipy.dot(scipy.transpose(X),X)),scipy.transpose(X)),y)
     19 | 	f = scipy.dot(X,b)
     20 | 	r = y - f
     21 | 	return b,f,r
```

</details>

---

### `bench_026` — `numpy.product` (numpy)
- **Sample ID**: `numpy_2793` | **Line**: 17:19 | **Stratum**: `hard_negative` (`stdlib_name_collision_product`)
- **Call Site**: `for repeats in itertools.product(*tuple(stride_set)):`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: itertools.product standard library collision.*

<details>
<summary>View Enclosing Code (numpy_2793)</summary>

```python
      1 | def _stride_comb_iter(x):
      2 |     """
      3 |     Generate cartesian product of strides for all axes
      4 |     """
      5 | 
      6 |     if not isinstance(x, np.ndarray):
      7 |         yield x, "nop"
      8 |         return
      9 | 
     10 |     stride_set = [(1,)]*x.ndim
     11 |     stride_set[-1] = (1, 3, -4)
     12 |     if x.ndim > 1:
     13 |         stride_set[-2] = (1, 3, -4)
     14 |     if x.ndim > 2:
     15 |         stride_set[-3] = (1, -4)
     16 | 
-->  17 |     for repeats in itertools.product(*tuple(stride_set)):
     18 |         new_shape = [abs(a*b) for a, b in zip(x.shape, repeats)]
     19 |         slices = tuple([slice(None, None, repeat) for repeat in repeats])
     20 | 
     21 |         # new array with different strides, but same data
     22 |         xi = np.empty(new_shape, dtype=x.dtype)
     23 |         xi.view(np.uint32).fill(0xdeadbeef)
     24 |         xi = xi[slices]
     25 |         xi[...] = x
     26 |         xi = xi.view(x.__class__)
     27 |         assert np.all(xi == x)
     28 |         yield xi, "stride_" + "_".join(["%+d" % j for j in repeats])
     29 | 
     30 |         # generate also zero strides if possible
     31 |         if x.ndim >= 1 and x.shape[-1] == 1:
     32 |             s = list(x.strides)
     33 |             s[-1] = 0
     34 |             xi = np.lib.stride_tricks.as_strided(x, strides=s)
     35 |             yield xi, "stride_xxx_0"
     36 |         if x.ndim >= 2 and x.shape[-2] == 1:
     37 |             s = list(x.strides)
     38 |             s[-2] = 0
     39 |             xi = np.lib.stride_tricks.as_strided(x, strides=s)
     40 |             yield xi, "stride_xxx_0_x"
     41 |         if x.ndim >= 2 and x.shape[:-2] == (1, 1):
     42 |             s = list(x.strides)
     43 |             s[-1] = 0
     44 |             s[-2] = 0
     45 |             xi = np.lib.stride_tricks.as_strided(x, strides=s)
     46 |             yield xi, "stride_xxx_0_0"
```

</details>

---

### `bench_027` — `numpy.alltrue` (numpy)
- **Sample ID**: `numpy_1571` | **Line**: 46:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `self.assert_(numpy.alltrue(numpy.isnan(nt3.data[365:400])))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (numpy_1571)</summary>

```python
      1 |     def test_boxcar(self):
      2 |         
      3 |         """ test boxcar performance in each steps of a godin filter"""
      4 |                 
      5 |         data=[1.0]*200+[2.0]*100+[1.0]*100
      6 |         data=numpy.array(data)
      7 |         st=datetime.datetime(year=1990,month=2,day=3,hour=11, minute=15)
      8 |         delta=time_interval(hours=1)
      9 |         test_ts=rts(data,st,delta,{})
     10 |         
     11 |         
     12 |         ## first round of boxcar in 24 (11,12).
     13 |         nt=boxcar(test_ts,11,12)
     14 |         self.assert_(numpy.alltrue(numpy.isnan(nt.data[0:11])))
     15 |         assert_array_almost_equal(nt.data[11:188],[1]*177,12)
     16 |         self.assert_(numpy.alltrue(numpy.greater(nt.data[188:211],1)))
     17 |         self.assertAlmostEqual(nt.data[196],1.375)
     18 |         assert_array_almost_equal(nt.data[211:288],[2]*77,12)
     19 |         self.assert_(numpy.alltrue(numpy.greater(nt.data[288:311],1)))
     20 |         self.assertAlmostEqual(nt.data[301],1.4166666667)
     21 |         assert_array_almost_equal(nt.data[311:388],[1]*77,12)
     22 |         self.assert_(numpy.alltrue(numpy.isnan(nt.data[388:400])))
     23 |     
     24 |         ## second round of boxcar in 24 (12,11)
     25 |         nt2=boxcar(nt,12,11)
     26 |         self.assert_(numpy.alltrue(numpy.isnan(nt2.data[0:23])))
     27 |         assert_array_almost_equal(nt2.data[23:177],[1]*154,12)
     28 |         self.assert_(numpy.alltrue(numpy.greater(nt2.data[177:223],1)))
     29 |         self.assertAlmostEqual(nt2.data[196],1.364583333)
     30 |         assert_array_almost_equal(nt2.data[223:277],[2]*54,12)
     31 |         self.assert_(numpy.alltrue(numpy.greater(nt2.data[277:323],1)))
     32 |         self.assertAlmostEqual(nt2.data[301],1.439236111)
     33 |         assert_array_almost_equal(nt2.data[323:377],[1]*54,12)
     34 |         self.assert_(numpy.alltrue(numpy.isnan(nt2.data[377:400])))
     35 |         
     36 |         ## third round of boxcar.   
     37 |         nt3=boxcar(nt2,12,12)
     38 |         self.assert_(numpy.alltrue(numpy.isnan(nt3.data[0:35])))
     39 |         assert_array_almost_equal(nt3.data[35:165],[1]*130,12)
     40 |         self.assert_(numpy.alltrue(numpy.greater(nt3.data[165:235],1)))
     41 |         self.assertAlmostEqual(nt3.data[196],1.393055556)
     42 |         assert_array_almost_equal(nt3.data[235:265],[2]*30,12)
     43 |         self.assert_(numpy.alltrue(numpy.greater(nt3.data[265:335],1)))
     44 |         self.assertAlmostEqual(nt3.data[301],1.453819444)
     45 |         assert_array_almost_equal(nt3.data[335:365],[1]*30,12)
-->  46 |         self.assert_(numpy.alltrue(numpy.isnan(nt3.data[365:400])))
```

</details>

---

### `bench_028` — `scipy.integrate.simps` (scipy)
- **Sample ID**: `scipy_1745` | **Line**: 19:28 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `lf = simps(X_s, x=X_t)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1745)</summary>

```python
      1 | def lifetime_accum(lifetime, X_s, X_t):
      2 |     """
      3 |     Compute the lifetime by integrating the correlation function
      4 |     and accumulate the result into the given lifetime object.
      5 | 
      6 |     Parameters
      7 |     ----------
      8 |     X_s: array-like
      9 |         Array with the correlation function.
     10 |         Works also with other decaying scores that are defined
     11 |         in the range [0,1]=[min_skill,max_skill].
     12 |     X_t: array-like
     13 |         Array with the forecast lead times in the desired unit,
     14 |         e.g. [min, hour].
     15 |     """
     16 |     if lifetime["rule"] == "trapz":
     17 |         lf = np.trapz(X_s, x=X_t)
     18 |     elif lifetime["rule"] == "simpson":
-->  19 |         lf = simps(X_s, x=X_t)
     20 |     elif lifetime["rule"] == "1/e":
     21 |         euler_number = 1.0 / exp(1.0)
     22 |         X_s_ = np.array(X_s)
     23 | 
     24 |         is_euler_reached = np.sum(X_s_ <= euler_number) > 0
     25 |         if is_euler_reached:
     26 |             idx_b = np.argmax(X_s_ <= euler_number)
     27 |             if idx_b > 0:
     28 |                 idx_a = idx_b - 1
     29 |                 fraction_score = (
     30 |                     (euler_number - X_s[idx_b])
     31 |                     * (X_t[idx_a] - X_t[idx_b])
     32 |                     / (X_s[idx_a] - X_s[idx_b])
     33 |                 )
     34 |                 lf = X_t[idx_b] + fraction_score
     35 |             else:
     36 |                 # if all values are below the 1/e value, return min lead time
     37 |                 lf = np.min(X_t)
     38 |         else:
     39 |             # if all values are above the 1/e value, return max lead time
     40 |             lf = np.max(X_t)
     41 | 
     42 |     lifetime["lifetime_sum"] += lf
     43 |     lifetime["n"] += 1
```

</details>

---

### `bench_029` — `scipy.linalg.pinv2` (scipy)
- **Sample ID**: `scipy_1246` | **Line**: 37:22 | **Stratum**: `hard_negative` (`submodule_non_deprecated_pinv`)
- **Call Site**: `blkAinv = scipy.linalg.pinv(blkA)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: non-deprecated standard pinv, not pinv2.*

<details>
<summary>View Enclosing Code (scipy_1246)</summary>

```python
      1 |     def test_schwarz_gold(self):
      2 |         np.random.seed(0)
      3 | 
      4 |         cases = []
      5 |         cases.append(poisson((4,), format='csr'))
      6 |         cases.append(poisson((4, 4), format='csr'))
      7 |         A = poisson((8, 8), format='csr')
      8 |         A.data[0] = 10.0
      9 |         A.data[1] = -0.5
     10 |         A.data[3] = -0.5
     11 |         cases.append(A)
     12 | 
     13 |         temp = np.random.rand(1, 1)
     14 |         cases.append(csr_matrix(temp.T.dot(temp)))
     15 | 
     16 |         temp = np.random.rand(2, 2)
     17 |         cases.append(csr_matrix(temp.T.dot(temp)))
     18 | 
     19 |         temp = np.random.rand(4, 4)
     20 |         cases.append(csr_matrix(temp.T.dot(temp)))
     21 | 
     22 |         # reference implementation
     23 |         def gold(A, x, b, iterations, sweep='forward'):
     24 |             A = csr_matrix(A)
     25 |             n = A.shape[0]
     26 | 
     27 |             # Default is point-wise iteration with each subdomain a point's
     28 |             # neighborhood in the matrix graph
     29 |             subdomains =\
     30 |                 [A.indices[A.indptr[i]:A.indptr[i+1]] for i in range(n)]
     31 | 
     32 |             # extract each subdomain's block from the matrix
     33 |             subblocks = []
     34 |             for i in range(len(subdomains)):
     35 |                 blkA = (A[subdomains[i], :]).tocsc()
     36 |                 blkA = blkA[:, subdomains[i]].toarray()
-->  37 |                 blkAinv = scipy.linalg.pinv(blkA)
     38 |                 subblocks.append(blkAinv)
     39 | 
     40 |             if sweep == 'forward':
     41 |                 indices = np.arange(len(subdomains))
     42 |             elif sweep == 'backward':
     43 |                 indices = np.arange(len(subdomains)-1, -1, -1)
     44 |             elif sweep == 'symmetric':
     45 |                 indices1 = np.arange(len(subdomains))
     46 |                 indices2 = np.arange(len(subdomains)-1, -1, -1)
     47 |                 indices = np.concatenate((indices1, indices2))
     48 | 
     49 |             # Multiplicative Schwarz iterations
     50 |             for _j in range(iterations):
     51 |                 for i in indices:
     52 |                     si = subdomains[i]
     53 |                     x[si] = np.dot(subblocks[i], (b[si] - A[si, :]*x)) + x[si]
     54 | 
     55 |             return x
     56 | 
     57 |         for A in cases:
     58 | 
     59 |             b = np.random.rand(A.shape[0], 1)
     60 |             x = np.random.rand(A.shape[0], 1)
     61 | 
     62 |             x_copy = x.copy()
     63 |             schwarz(A, x, b, iterations=1, sweep='forward')
     64 |             assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
     65 |                                         sweep='forward'))
     66 | 
     67 |             x_copy = x.copy()
     68 |             schwarz(A, x, b, iterations=1, sweep='backward')
     69 |             assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
     70 |                                         sweep='backward'))
     71 | 
     72 |             x_copy = x.copy()
     73 |             schwarz(A, x, b, iterations=1, sweep='symmetric')
     74 |             assert_almost_equal(x, gold(A, x_copy, b, iterations=1,
     75 |                                         sweep='symmetric'))
```

</details>

---

### `bench_030` — `scipy.misc.factorial2` (scipy)
- **Sample ID**: `scipy_1929` | **Line**: 25:0 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `factor_lm = 1/4*np.sqrt(spm.factorial2(2*l+1)/np.pi/Nlm)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1929)</summary>

```python
      1 | def print_opt_prim_gauss_cart(prim_gauss_sph,fullpath,atomlabel,l):
      2 |  #creating a dictionary of primitive cartesian gaussians from the fitted spherical gaussians
      3 |  import scipy.misc as spm
      4 |  # spherical to cartesian gaussians
      5 |  #l,m)  : [ (Nlm , [( c, lx,ly,lz)])
      6 |  s2c = {
      7 |  (0, 0) :  (1/4 , [( 1, 0, 0, 0)] ) ,
      8 |  (1,-1) :  (1/4 , [( 1, 0, 1, 0)] ) ,
      9 |  (1, 0) :  (1/4 , [( 1, 0, 0, 1)] ) ,
     10 |  (1, 1) :  (1/4 , [( 1, 1, 0, 0)] ) ,
     11 |  (2,-2) :  (1/4 , [( 1, 1, 1, 0)] ) ,
     12 |  (2,-1) :  (1/4 , [( 1, 0, 1, 1)] ) ,
     13 |  (2, 0) :  (  3 , [( 2, 0, 0, 2),
     14 |                    (-1, 2, 0, 0),
     15 |                    (-1, 0, 2, 0)] ) ,
     16 |  (2, 1) :  (1/4 , [( 1, 1, 0, 1)] ) ,
     17 |  (2, 2) :  (  1 , [( 1, 2, 0, 0),
     18 |                    (-1, 0, 2, 0)] ) }
     19 |  
     20 |  PCG = {}
     21 |  for m in range(-l,l+1):
     22 |   Nlm, mytuple = s2c[(l,m)]
     23 |   g_tuples = []
     24 |   for counter,(d,lx,ly,lz) in enumerate(mytuple):
-->  25 |    factor_lm = 1/4*np.sqrt(spm.factorial2(2*l+1)/np.pi/Nlm)
     26 |    for prim_g in prim_gauss_sph: 
     27 |     print(prim_g)
     28 |     g_tuples = g_tuples + [(lx, ly, lz, d*prim_g[1]*factor_lm,prim_g[0])]
     29 |    dict_field= {(l,m) : g_tuples} 
     30 |    PCG.update(dict_field)
     31 | 
     32 |  llabel = { 0 : 's', 1 : 'p', 2 : 'd' }
     33 |  fullname = fullpath+'/'+atomlabel+'.'+llabel[l]+'.py'
     34 |  f = open(fullname,'w+')
     35 |  #f.write(atomlabel+'.'+llabel[l]+ '=' + repr(PCG) + '\n' )
     36 |  f.write('cgto=' + repr(PCG) + '\n' )
     37 |  f.close()
```

</details>

---

### `bench_031` — `scipy.interpolate.interp2d` (scipy)
- **Sample ID**: `scipy_1171` | **Line**: 4:6 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `fabs = scipy.interpolate.interp2d(self.lam, self.a, \`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1171)</summary>

```python
      1 |     def calculate_opacity(self, a):
      2 |         mdust = 4*pi*a**3/3*self.rho
      3 | 
-->   4 |         fabs = scipy.interpolate.interp2d(self.lam, self.a, \
      5 |                 numpy.log10(self.Qabs), kind='linear')
      6 |         fsca = scipy.interpolate.interp2d(self.lam, self.a, \
      7 |                 numpy.log10(self.Qsca), kind='linear')
      8 |         fext = scipy.interpolate.interp2d(self.lam, self.a, \
      9 |                 numpy.log10(self.Qext), kind='linear')
     10 | 
     11 |         Qabs = 10.**fabs(self.lam, a)
     12 |         Qsca = 10.**fsca(self.lam, a)
     13 |         Qext = 10.**fext(self.lam, a)
     14 |         
     15 |         self.kabs = pi*a**2*Qabs/mdust
     16 |         self.ksca = pi*a**2*Qsca/mdust
     17 |         self.kext = pi*a**2*Qext/mdust
     18 |     
     19 |         self.albedo = self.ksca / self.kext
```

</details>

---

### `bench_032` — `scipy.misc.factorial` (scipy)
- **Sample ID**: `scipy_1573` | **Line**: 19:0 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `lKf[tmp] = np.log( scipy.misc.factorial(K[tmp], exact=False) )`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1573)</summary>

```python
      1 | def photon_error(evt, type_data, key_data, type_fit, key_fit, adu_per_photon):
      2 |     import scipy.misc
      3 |     data = np.array(evt[type_data][key_data].data / (1.*adu_per_photon), dtype="float")
      4 |     fit = np.array(evt[type_fit][key_fit].data / (1.*adu_per_photon), dtype="float")
      5 |     data_best = fit.round()
      6 |     data = data.copy()
      7 |     M = fit != 0
      8 |     M *= data > 0
      9 |     M *= data_best > 0
     10 | 
     11 |     K = data[M]
     12 |     W = fit[M]
     13 |     Ks = data_best[M]
     14 |     
     15 |     # Stirling
     16 |     lKf = K*np.log(K)-K
     17 |     tmp = K < 5
     18 |     if tmp.sum():
-->  19 |         lKf[tmp] = np.log( scipy.misc.factorial(K[tmp], exact=False) )
     20 | 
     21 |     # Stirling
     22 |     lKsf = Ks*np.log(Ks)-Ks
     23 |     tmp = Ks < 5
     24 |     if tmp.sum():
     25 |         lKsf[tmp] = np.log( scipy.misc.factorial(Ks[tmp], exact=False) )
     26 |     
     27 |     error = ( Ks * np.log(W) - lKsf ) - ( K * np.log(W) - lKf )
     28 |     error = error.sum()
     29 |     add_record(evt["analysis"], "analysis", "photon error", error, unit='')
```

</details>

---

### `bench_033` — `scipy.misc.logsumexp` (scipy)
- **Sample ID**: `scipy_602` | **Line**: 11:0 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `res = scipy.misc.logsumexp(ps + np.log(self.a), axis=1) if log else np.dot(ps, self.a)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_602)</summary>

```python
      1 |     def eval(self, x, ii=None, log=True):
      2 |         """
      3 |         Evaluates the mog pdf.
      4 |         :param x: rows are inputs to evaluate at
      5 |         :param ii: a list of indices specifying which marginal to evaluate. if None, the joint pdf is evaluated
      6 |         :param log: if True, the log pdf is evaluated
      7 |         :return: pdf or log pdf
      8 |         """
      9 | 
     10 |         ps = np.array([c.eval(x, ii, log) for c in self.xs]).T
-->  11 |         res = scipy.misc.logsumexp(ps + np.log(self.a), axis=1) if log else np.dot(ps, self.a)
     12 | 
     13 |         return res
```

</details>

---

### `bench_034` — `pandas.DataFrame.iteritems` (pandas)
- **Sample ID**: `pandas_87` | **Line**: 5:19 | **Stratum**: `hard_negative` (`wrapper_lookalike_pyspark`)
- **Call Site**: `self.assert_eq(pdf.first("1D"), psdf.first("1D"))`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: third-party PySpark/Koalas DataFrame/Series method.*

<details>
<summary>View Enclosing Code (pandas_87)</summary>

```python
      1 |     def test_first(self):
      2 |         index = pd.date_range("2018-04-09", periods=4, freq="2D")
      3 |         pdf = pd.DataFrame([1, 2, 3, 4], index=index)
      4 |         psdf = ps.from_pandas(pdf)
-->   5 |         self.assert_eq(pdf.first("1D"), psdf.first("1D"))
      6 |         self.assert_eq(pdf.first(DateOffset(days=1)), psdf.first(DateOffset(days=1)))
      7 |         with self.assertRaisesRegex(TypeError, "'first' only supports a DatetimeIndex"):
      8 |             ps.DataFrame([1, 2, 3, 4]).first("1D")
```

</details>

---

### `bench_035` — `numpy.product` (numpy)
- **Sample ID**: `numpy_2597` | **Line**: 41:15 | **Stratum**: `hard_negative` (`stdlib_name_collision_product`)
- **Call Site**: `for idx in itertools.product(*map(range, operand.shape[:-2])):`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: itertools.product standard library collision.*

<details>
<summary>View Enclosing Code (numpy_2597)</summary>

```python
      1 |   def eig(cls, harness: test_harnesses.Harness):
      2 |     compute_left_eigenvectors = harness.params["compute_left_eigenvectors"]
      3 |     compute_right_eigenvectors = harness.params["compute_right_eigenvectors"]
      4 |     dtype = harness.dtype
      5 | 
      6 |     def custom_assert(tst, result_jax, result_tf, *, args, tol, err_msg):
      7 |       operand, = args
      8 |       inner_dimension = operand.shape[-1]
      9 | 
     10 |       # Test ported from tests.linlag_test.testEig
     11 |       # Norm, adjusted for dimension and type.
     12 |       def norm(x):
     13 |         norm = np.linalg.norm(x, axis=(-2, -1))
     14 |         return norm / ((inner_dimension + 1) * jnp.finfo(dtype).eps)
     15 | 
     16 |       def check_right_eigenvectors(a, w, vr):
     17 |         tst.assertTrue(
     18 |             np.all(norm(np.matmul(a, vr) - w[..., None, :] * vr) < 100))
     19 | 
     20 |       def check_left_eigenvectors(a, w, vl):
     21 |         rank = len(a.shape)
     22 |         aH = jnp.conj(a.transpose(list(range(rank - 2)) + [rank - 1, rank - 2]))
     23 |         wC = jnp.conj(w)
     24 |         check_right_eigenvectors(aH, wC, vl)
     25 | 
     26 |       def check_eigenvalue_is_in_array(eigenvalue, eigenvalues_array):
     27 |         tol = None
     28 |         # TODO(bchetioui): numerical discrepancies
     29 |         if dtype in [np.float32, np.complex64]:
     30 |           tol = 1e-4
     31 |         elif dtype in [np.float64, np.complex128]:
     32 |           tol = 1e-13
     33 |         closest_diff = min(abs(eigenvalues_array - eigenvalue))
     34 |         tst.assertAllClose(
     35 |             closest_diff,
     36 |             np.array(0., closest_diff.dtype),
     37 |             atol=tol,
     38 |             err_msg=err_msg)
     39 | 
     40 |       all_w_jax, all_w_tf = result_jax[0], result_tf[0]
-->  41 |       for idx in itertools.product(*map(range, operand.shape[:-2])):
     42 |         w_jax, w_tf = all_w_jax[idx], all_w_tf[idx]
     43 |         for i in range(inner_dimension):
     44 |           check_eigenvalue_is_in_array(w_jax[i], w_tf)
     45 |           check_eigenvalue_is_in_array(w_tf[i], w_jax)
     46 | 
     47 |       if compute_left_eigenvectors:
     48 |         check_left_eigenvectors(operand, all_w_tf, result_tf[1])
     49 |       if compute_right_eigenvectors:
     50 |         check_right_eigenvectors(operand, all_w_tf,
     51 |                                  result_tf[1 + compute_left_eigenvectors])
     52 | 
     53 |     return [
     54 |         # Eig does not work in JAX on gpu or tpu
     55 |         Jax2TfLimitation(
     56 |             "function not compilable", modes="compiled", devices="cpu"),
     57 |         Jax2TfLimitation(
     58 |             "TF Conversion of eig is not implemented when both compute_left_eigenvectors and compute_right_eigenvectors are set to True",
     59 |             enabled=(compute_left_eigenvectors and compute_right_eigenvectors)),
     60 |         custom_numeric(
     61 |             custom_assert=custom_assert,
     62 |             description=("May return the eigenvalues and eigenvectors in a "
     63 |                          "potentially different order. The eigenvectors may "
     64 |                          "also be different, but equally valid."))
     65 |     ]
```

</details>

---

### `bench_036` — `scipy.misc.comb` (scipy)
- **Sample ID**: `scipy_134` | **Line**: 18:12 | **Stratum**: `hard_negative` (`replacement_overload_comb`)
- **Call Site**: `scipy.special.comb(sample_count, sample_array)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: modern replacement scipy.special.comb.*

<details>
<summary>View Enclosing Code (scipy_134)</summary>

```python
      1 | def expected_entropy_estimate(probabilities, sample_count):
      2 |     """Compute the expected entropy estimate sampling N elements underlying
      3 |     probabilities `p`
      4 | 
      5 | 
      6 |     Arguments:
      7 |         probabilities (numpy.ndarray): probabilities (p)
      8 |                                        (assumed to sum to 1.0)
      9 |         sample_count (int): number of samples (N)
     10 | 
     11 |     Returns:
     12 |         entropy (float): expected entropy
     13 |     """
     14 |     entropy = 0.0
     15 |     for probability in probabilities:
     16 |         sample_array = np.arange(1, sample_count)
     17 |         entropy += np.sum(
-->  18 |             scipy.special.comb(sample_count, sample_array)
     19 |             * np.power(probability, sample_array)
     20 |             * np.power(1 - probability, sample_count - sample_array)
     21 |             # should be sample_array/sample_count
     22 |             # but we moved the 1/sample_count out:
     23 |             * sample_array
     24 |             * np.log(sample_array / sample_count)
     25 |         )
     26 |     return -entropy / sample_count
```

</details>

---

### `bench_037` — `scipy.integrate.cumtrapz` (scipy)
- **Sample ID**: `scipy_1528` | **Line**: 9:13 | **Stratum**: `hard_negative` (`replacement_overload_integrate`)
- **Call Site**: `px_desired = integrate.cumulative_trapezoid(vx_desired, dx=dt, initial=0)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: modern replacement cumulative_trapezoid.*

<details>
<summary>View Enclosing Code (scipy_1528)</summary>

```python
      1 | alpha_desired[800::,0]  = np.linspace(np.pi/8, -np.pi, 200)
      2 | 
      3 | 
      4 | freq = np.median(freq)
      5 | 
      6 | vx_desired = (a0 * freq * np.cos(alpha_desired)).reshape(1,-1).flatten()
      7 | vy_desired = (a0 * freq * np.sin(alpha_desired)).reshape(1,-1).flatten()
      8 | 
-->   9 | px_desired = integrate.cumulative_trapezoid(vx_desired, dx=dt, initial=0)
```

</details>

---

### `bench_038` — `scipy.stats.chisqprob` (scipy)
- **Sample ID**: `scipy_1671` | **Line**: 28:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `p = scipy.stats.chisqprob(D, df)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1671)</summary>

```python
      1 | def deviance(X, y_true, y_pred):
      2 |     """Computes the deviance statistic.
      3 | 
      4 |     Parameters
      5 |     ----------
      6 |     X : array
      7 |         Design matrix.
      8 |     y_true : array
      9 |         Observed labels, either 0 or 1.
     10 |     y_pred : array
     11 |         Predicted probabilities, floats on [0, 1].
     12 | 
     13 |     Returns
     14 |     -------
     15 |     D : float
     16 |         The test statistic :math:`D`.
     17 |     df : int
     18 |         The degrees of freedom of the test.
     19 |     p : float
     20 |         The p-value of the test.
     21 |     """
     22 |     __, p = X.shape
     23 | 
     24 |     d = deviance_residuals(y_true, y_pred)
     25 | 
     26 |     D = np.sum(np.square(d))
     27 |     df = len(y_true) - (p + 1)
-->  28 |     p = scipy.stats.chisqprob(D, df)
     29 | 
     30 |     TestResult = namedtuple('Deviance', ('D', 'df', 'p'))
     31 |     return TestResult(D, df, p)
```

</details>

---

### `bench_039` — `scipy.misc.comb` (scipy)
- **Sample ID**: `scipy_132` | **Line**: 24:18 | **Stratum**: `hard_negative` (`replacement_overload_comb`)
- **Call Site**: `choose_2p_p = scipy.special.comb(2*p, p, exact=True)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: modern replacement scipy.special.comb.*

<details>
<summary>View Enclosing Code (scipy_132)</summary>

```python
      1 | def _ell(A, m):
      2 |     """A helper function for expm_2009.
      3 | 
      4 |     Parameters
      5 |     ----------
      6 |     A : linear operator
      7 |         A linear operator whose norm of power we care about.
      8 |     m : int
      9 |         The power of the linear operator
     10 | 
     11 |     Returns
     12 |     -------
     13 |     value : int
     14 |         A value related to a bound.
     15 | 
     16 |     """
     17 |     if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
     18 |         raise ValueError('expected A to be like a square matrix')
     19 | 
     20 |     p = 2*m + 1
     21 | 
     22 |     # The c_i are explained in (2.2) and (2.6) of the 2005 expm paper.
     23 |     # They are coefficients of terms of a generating function series expansion.
-->  24 |     choose_2p_p = scipy.special.comb(2*p, p, exact=True)
     25 |     abs_c_recip = float(choose_2p_p * math.factorial(2*p + 1))
     26 | 
     27 |     # This is explained after Eq. (1.2) of the 2009 expm paper.
     28 |     # It is the "unit roundoff" of IEEE double precision arithmetic.
     29 |     u = 2.**-24
     30 | 
     31 |     # Compute the one-norm of matrix power p of abs(A).
     32 |     A_abs_onenorm = _onenorm_matrix_power_nnm(abs(A), p)
     33 | 
     34 |     # Treat zero norm as a special case.
     35 |     if not A_abs_onenorm:
     36 |         return 0
     37 | 
     38 |     alpha = A_abs_onenorm / (_onenorm(A) * abs_c_recip)
     39 |     return max(int(np.ceil(np.log2(alpha/u) / (2 * m))), 0)
```

</details>

---

### `bench_040` — `scipy.misc.factorial` (scipy)
- **Sample ID**: `scipy_1557` | **Line**: 14:23 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `convPml = np.sqrt(2. * factorial(mls[:,:,1]-mls[:,:,0])/factorial(mls[:,:,1]+mls[:,:,0]))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1557)</summary>

```python
      1 | CosMphi = np.cos(phiMs * Phis.reshape([nPhi,-1]))
      2 | # Array holding the values of m and l in a 2d grid to make matrix mult easier
      3 | mls = np.empty([nHarmonics+1,nHarmonics+1,2])
      4 | for i in range(nHarmonics+1):
      5 |     mls[:,i,0] = range(nHarmonics+1) #ms
      6 |     mls[i,:,1] = range(nHarmonics+1) #ls
      7 |     
      8 |     
      9 | # Need to be able to convert from scipy poly normalization to 
     10 | # PFSS version, which depends on some large factorials
     11 | # Save the log of convPml without the (-1)^m factor otherwise will round to
     12 | # zero for m+l above 170, will take exponent later    
     13 | convPml = np.empty([nHarmonics+1,nHarmonics+1], dtype=np.longdouble)
-->  14 | convPml = np.sqrt(2. * factorial(mls[:,:,1]-mls[:,:,0])/factorial(mls[:,:,1]+mls[:,:,0]))
```

</details>

---

### `bench_041` — `scipy.stats.betai` (scipy)
- **Sample ID**: `scipy_471` | **Line**: 21:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `prob = [scipy.stats.betai(0.5 * dfden, 0.5 * dfnum, dfden/float(dfden+dfnum*f)) if f is not ma.masked and dfden/float(dfden+dfnum*f) <= 1.0 \`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_471)</summary>

```python
      1 | def aF_oneway(*args, **kwargs):
      2 |     dim = kwargs.get("dim", None)
      3 |     arrays = args
      4 |     means = [ma.mean(a, dim) for a in arrays]
      5 |     vars = [ma.var(a, dim) for a in arrays]
      6 |     lens = [ma.sum(ma.array(ma.ones(a.shape), mask=ma.asarray(a).mask), dim) for a in arrays]
      7 |     alldata = ma.concatenate(arrays, dim if dim is not None else 0)
      8 |     bign =  ma.sum(ma.array(ma.ones(alldata.shape), mask=alldata.mask), dim)
      9 |     sstot = ma.sum(alldata ** 2, dim) - (ma.sum(alldata, dim) ** 2) / bign
     10 |     ssbn = ma.sum([(ma.sum(a, dim) ** 2) / L for a, L in zip(arrays, lens)], dim)
     11 | #    print ma.sum(alldata, dim) ** 2 / bign, ssbn
     12 |     ssbn -= ma.sum(alldata, dim) ** 2 / bign
     13 |     sswn = sstot - ssbn
     14 |     dfbn = dfnum = float(len(args) - 1.0)
     15 |     dfwn = bign - len(args) # + 1.0
     16 |     F = (ssbn / dfbn) / (sswn / dfwn)
     17 |     if F.ndim == 0 and dfwn.ndim == 0:
     18 |         return (F,scipy.stats.betai(0.5 * dfwn, 0.5 * dfnum, dfwn/float(dfwn+dfnum*F)) if F is not ma.masked and dfwn/float(dfwn+dfnum*F) <= 1.0 \
     19 |                 and dfwn/float(dfwn+dfnum*F) >= 0.0 else ma.masked)
     20 |     else:
-->  21 |         prob = [scipy.stats.betai(0.5 * dfden, 0.5 * dfnum, dfden/float(dfden+dfnum*f)) if f is not ma.masked and dfden/float(dfden+dfnum*f) <= 1.0 \
     22 |             and dfden/float(dfden+dfnum*f) >= 0.0 else ma.masked for dfden, f in zip (dfwn, F)]
     23 |         return F, prob
```

</details>

---

### `bench_042` — `scipy.special.errprint` (scipy)
- **Sample ID**: `scipy_268` | **Line**: 4:8 | **Stratum**: `pipeline_miss` (`cython_native_ufunc`)
- **Call Site**: `c = special.errprint(b)  # returns last state 'a'`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Pipeline missed: compiled Cython native ufunc unparseable by AST.*

<details>
<summary>View Enclosing Code (scipy_268)</summary>

```python
      1 |     def test_errprint(self):
      2 |         a = special.errprint()
      3 |         b = 1-a  # a is the state 1-a inverts state
-->   4 |         c = special.errprint(b)  # returns last state 'a'
      5 |         assert_equal(a,c)
      6 |         d = special.errprint(a)  # returns to original state
      7 |         assert_equal(d,b)
```

</details>

---

### `bench_043` — `scipy.misc.face` (scipy)
- **Sample ID**: `scipy_2145` | **Line**: 7:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `face = misc.face(gray=True)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_2145)</summary>

```python
      1 | def test_connect_regions_with_grid():
      2 |     try:
      3 |         face = sp.face(gray=True)
      4 |     except AttributeError:
      5 |         # Newer versions of scipy have face in misc
      6 |         from scipy import misc
-->   7 |         face = misc.face(gray=True)
      8 |     mask = face > 50
      9 |     graph = grid_to_graph(*face.shape, mask=mask)
     10 |     assert_equal(ndimage.label(mask)[1], connected_components(graph)[0])
     11 | 
     12 |     mask = face > 150
     13 |     graph = grid_to_graph(*face.shape, mask=mask, dtype=None)
     14 |     assert_equal(ndimage.label(mask)[1], connected_components(graph)[0])
```

</details>

---

### `bench_044` — `pandas.DataFrame.swapaxes` (pandas)
- **Sample ID**: `pandas_27` | **Line**: 4:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `df2 = df.swapaxes(ax, ax)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_27)</summary>

```python
      1 | def test_swapaxes_noop(using_copy_on_write, ax):
      2 |     df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
      3 |     df_orig = df.copy()
-->   4 |     df2 = df.swapaxes(ax, ax)
      5 | 
      6 |     if using_copy_on_write:
      7 |         assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
      8 |     else:
      9 |         assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
     10 | 
     11 |     # mutating df2 triggers a copy-on-write for that column/block
     12 |     df2.iloc[0, 0] = 0
     13 |     if using_copy_on_write:
     14 |         assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
     15 |     tm.assert_frame_equal(df, df_orig)
```

</details>

---

### `bench_045` — `scipy.stats.rvs_ratio_uniforms` (scipy)
- **Sample ID**: `scipy_460` | **Line**: 6:9 | **Stratum**: `pipeline_miss` (`preamble_filter_omission`)
- **Call Site**: `r1 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=(3, 4))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Pipeline missed: receiver 'stats' omitted from preamble, callee dropped by manifest short-name filter.*

<details>
<summary>View Enclosing Code (scipy_460)</summary>

```python
      1 |     def test_random_state(self):
      2 |         f = stats.norm.pdf
      3 |         v_bound = np.sqrt(f(np.sqrt(2))) * np.sqrt(2)
      4 |         umax, vmin, vmax = np.sqrt(f(0)), -v_bound, v_bound
      5 |         np.random.seed(1234)
-->   6 |         r1 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=(3, 4))
      7 |         r2 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=(3, 4),
      8 |                                       random_state=1234)
      9 |         assert_equal(r1, r2)
```

</details>

---

### `bench_046` — `scipy.misc.face` (scipy)
- **Sample ID**: `scipy_2157` | **Line**: 2:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `face = scipy.misc.face()`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_2157)</summary>

```python
      1 | logger.configure(os.path.realpath('.'))
-->   2 | face = scipy.misc.face()
```

</details>

---

### `bench_047` — `scipy.integrate.cumtrapz` (scipy)
- **Sample ID**: `scipy_1430` | **Line**: 9:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `pylab.semilogx(total_time[k],scipy.integrate.cumtrapz(Melt_volume[k],x=total_time[k],initial=0)/Vmantle0,'b')`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1430)</summary>

```python
      1 | pylab.xlim([x_low, x_high])
      2 | pylab.legend(frameon=False)
      3 | pylab.subplot(2,1,2)
      4 | Vmantle0 = ((4.0*np.pi/3.0) * (rp**3 - rc**3))/1e9
      5 | for k in range(0,len(inputs)):
      6 |     for i in range(0,len(Melt_volume[k])): 
      7 |         if (total_time[k][i]<inputs_for_MC[k][5].interiorg):
      8 |             Melt_volume[k][i] = 0.0
-->   9 |     pylab.semilogx(total_time[k],scipy.integrate.cumtrapz(Melt_volume[k],x=total_time[k],initial=0)/Vmantle0,'b')
```

</details>

---

### `bench_048` — `pandas.DataFrame.iteritems` (pandas)
- **Sample ID**: `pandas_45` | **Line**: 12:19 | **Stratum**: `hard_negative` (`wrapper_lookalike_pyspark`)
- **Call Site**: `self.assert_eq(pdf1.transpose().sort_index(), psdf1.transpose().sort_index())`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: third-party PySpark/Koalas DataFrame/Series method.*

<details>
<summary>View Enclosing Code (pandas_45)</summary>

```python
      1 |     def test_transpose(self):
      2 |         # TODO: what if with random index?
      3 |         pdf1 = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]}, columns=["col1", "col2"])
      4 |         psdf1 = ps.from_pandas(pdf1)
      5 | 
      6 |         pdf2 = pd.DataFrame(
      7 |             data={"score": [9, 8], "kids": [0, 0], "age": [12, 22]},
      8 |             columns=["score", "kids", "age"],
      9 |         )
     10 |         psdf2 = ps.from_pandas(pdf2)
     11 | 
-->  12 |         self.assert_eq(pdf1.transpose().sort_index(), psdf1.transpose().sort_index())
     13 |         self.assert_eq(pdf2.transpose().sort_index(), psdf2.transpose().sort_index())
     14 | 
     15 |         with option_context("compute.max_rows", None):
     16 |             self.assert_eq(pdf1.transpose().sort_index(), psdf1.transpose().sort_index())
     17 | 
     18 |             self.assert_eq(pdf2.transpose().sort_index(), psdf2.transpose().sort_index())
     19 | 
     20 |         pdf3 = pd.DataFrame(
     21 |             {
     22 |                 ("cg1", "a"): [1, 2, 3],
     23 |                 ("cg1", "b"): [4, 5, 6],
     24 |                 ("cg2", "c"): [7, 8, 9],
     25 |                 ("cg3", "d"): [9, 9, 9],
     26 |             },
     27 |             index=pd.MultiIndex.from_tuples([("rg1", "x"), ("rg1", "y"), ("rg2", "z")]),
     28 |         )
     29 |         psdf3 = ps.from_pandas(pdf3)
     30 | 
     31 |         self.assert_eq(pdf3.transpose().sort_index(), psdf3.transpose().sort_index())
     32 | 
     33 |         with option_context("compute.max_rows", None):
     34 |             self.assert_eq(pdf3.transpose().sort_index(), psdf3.transpose().sort_index())
```

</details>

---

### `bench_049` — `scipy.special.sph_yn` (scipy)
- **Sample ID**: `scipy_1047` | **Line**: 2:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `return scipy.special.sph_yn(n, z)[0][-1]`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1047)</summary>

```python
      1 | def sph_yn(n, z):
-->   2 |     return scipy.special.sph_yn(n, z)[0][-1]
```

</details>

---

### `bench_050` — `scipy.misc.factorial` (scipy)
- **Sample ID**: `scipy_1557` | **Line**: 14:23 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `convPml = np.sqrt(2. * factorial(mls[:,:,1]-mls[:,:,0])/factorial(mls[:,:,1]+mls[:,:,0]))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1557)</summary>

```python
      1 | CosMphi = np.cos(phiMs * Phis.reshape([nPhi,-1]))
      2 | # Array holding the values of m and l in a 2d grid to make matrix mult easier
      3 | mls = np.empty([nHarmonics+1,nHarmonics+1,2])
      4 | for i in range(nHarmonics+1):
      5 |     mls[:,i,0] = range(nHarmonics+1) #ms
      6 |     mls[i,:,1] = range(nHarmonics+1) #ls
      7 |     
      8 |     
      9 | # Need to be able to convert from scipy poly normalization to 
     10 | # PFSS version, which depends on some large factorials
     11 | # Save the log of convPml without the (-1)^m factor otherwise will round to
     12 | # zero for m+l above 170, will take exponent later    
     13 | convPml = np.empty([nHarmonics+1,nHarmonics+1], dtype=np.longdouble)
-->  14 | convPml = np.sqrt(2. * factorial(mls[:,:,1]-mls[:,:,0])/factorial(mls[:,:,1]+mls[:,:,0]))
```

</details>

---

### `bench_051` — `pandas.Series.pad` (pandas)
- **Sample ID**: `pandas_126` | **Line**: 7:4 | **Stratum**: `hard_negative` (`replacement_overload_pandas`)
- **Call Site**: `df.ffill(inplace=True)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: modern replacement ffill.*

<details>
<summary>View Enclosing Code (pandas_126)</summary>

```python
      1 | def test_interpolate_creates_copy(using_copy_on_write):
      2 |     # GH#51126
      3 |     df = DataFrame({"a": [1.5, np.nan, 3]})
      4 |     view = df[:]
      5 |     expected = df.copy()
      6 | 
-->   7 |     df.ffill(inplace=True)
      8 |     df.iloc[0, 0] = 100.5
      9 | 
     10 |     if using_copy_on_write:
     11 |         tm.assert_frame_equal(view, expected)
     12 |     else:
     13 |         expected = DataFrame({"a": [100.5, 1.5, 3]})
     14 |         tm.assert_frame_equal(view, expected)
```

</details>

---

### `bench_052` — `scipy.special.sph_yn` (scipy)
- **Sample ID**: `scipy_986` | **Line**: 5:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `ynb,yprb =  scipy.special.sph_yn(n,beta)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_986)</summary>

```python
      1 | def phaseshifts(n,alpha,beta):
      2 |     gamm = Gam(n,alpha)
      3 |     num = np.zeros((n),float); den = np.zeros((n),float)
      4 |     jnb,jnpr =   scipy.special.sph_jn(n,beta)
-->   5 |     ynb,yprb =  scipy.special.sph_yn(n,beta)
      6 |     for i in range(0,n):
      7 |         num1=gamm[i]*jnb[i]
      8 |         den1=gamm[i]*ynb[i]
      9 |         num[i]=beta*jnpr[i]-num1
     10 |         den[i]=beta*yprb[i]-den1  
     11 |         td=atan2(num[i],den[i])      
     12 |         delta[i]=td
     13 |     return delta
```

</details>

---

### `bench_053` — `scipy.special.sph_jn` (scipy)
- **Sample ID**: `scipy_989` | **Line**: 3:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `for nn in range(0,n): jn,jpr = scipy.special.sph_jn(nn,xx)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_989)</summary>

```python
      1 | def Gam(n,xx): # Spherical Bessel ratio
      2 |     gamma = np.zeros((n),float)
-->   3 |     for nn in range(0,n): jn,jpr = scipy.special.sph_jn(nn,xx)  
      4 |     gamma = alpha*jpr/jn   # gamma match psi outside-inside
      5 |     return gamma
```

</details>

---

### `bench_054` — `pandas.Series.pad` (pandas)
- **Sample ID**: `pandas_123` | **Line**: 13:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `self.assert_eq(pdf.pad(), psdf.pad())`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_123)</summary>

```python
      1 |     def test_pad(self):
      2 |         pdf = pd.DataFrame(
      3 |             {
      4 |                 "A": [None, 3, None, None],
      5 |                 "B": [2, 4, None, 3],
      6 |                 "C": [None, None, None, 1],
      7 |                 "D": [0, 1, 5, 4],
      8 |             },
      9 |             columns=["A", "B", "C", "D"],
     10 |         )
     11 |         psdf = ps.from_pandas(pdf)
     12 | 
-->  13 |         self.assert_eq(pdf.pad(), psdf.pad())
     14 | 
     15 |         # Test `inplace=True`
     16 |         pdf.pad(inplace=True)
     17 |         psdf.pad(inplace=True)
     18 |         self.assert_eq(pdf, psdf)
```

</details>

---

### `bench_055` — `scipy.misc.comb` (scipy)
- **Sample ID**: `scipy_147` | **Line**: 4:26 | **Stratum**: `hard_negative` (`replacement_overload_comb`)
- **Call Site**: `out[i] = 1 / (n * scipy.special.comb(n-1,i))`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: modern replacement scipy.special.comb.*

<details>
<summary>View Enclosing Code (scipy_147)</summary>

```python
      1 | def shapley_coefficients(n):
      2 |     out = np.zeros(n)
      3 |     for i in range(n):
-->   4 |         out[i] = 1 / (n * scipy.special.comb(n-1,i))
      5 |     return out
```

</details>

---

### `bench_056` — `numpy.product` (numpy)
- **Sample ID**: `numpy_169` | **Line**: 23:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `(numpy.product(value.shape) == 1 and numpy.product(attr.shape) == 1):`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (numpy_169)</summary>

```python
      1 |     def modify(self, name, value):
      2 |         """ Change the value of an attribute while preserving its type.
      3 | 
      4 |         Differs from __setitem__ in that if the attribute already exists, its
      5 |         type is preserved.  This can be very useful for interacting with
      6 |         externally generated files.
      7 | 
      8 |         If the attribute doesn't exist, it will be automatically created.
      9 |         """
     10 |         with phil:
     11 |             if not name in self:
     12 |                 self[name] = value
     13 |             else:
     14 |                 value = numpy.asarray(value, order='C')
     15 | 
     16 |                 attr = h5a.open(self._id, self._e(name))
     17 | 
     18 |                 if attr.get_space().get_simple_extent_type() == h5s.NULL:
     19 |                     raise IOError("Empty attributes can't be modified")
     20 | 
     21 |                 # Allow the case of () <-> (1,)
     22 |                 if (value.shape != attr.shape) and not \
-->  23 |                    (numpy.product(value.shape) == 1 and numpy.product(attr.shape) == 1):
     24 |                     raise TypeError("Shape of data is incompatible with existing attribute")
     25 |                 attr.write(value)
```

</details>

---

### `bench_057` — `pandas.DataFrame.swapaxes` (pandas)
- **Sample ID**: `pandas_32` | **Line**: 10:24 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `self.assert_eq(psdf.swapaxes("columns", "index"), pdf.swapaxes("columns", "index"))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_32)</summary>

```python
      1 |     def test_swapaxes(self):
      2 |         pdf = pd.DataFrame(
      3 |             [[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=["x", "y", "z"], columns=["a", "b", "c"]
      4 |         )
      5 |         psdf = ps.from_pandas(pdf)
      6 | 
      7 |         self.assert_eq(psdf.swapaxes(0, 1), pdf.swapaxes(0, 1))
      8 |         self.assert_eq(psdf.swapaxes(1, 0), pdf.swapaxes(1, 0))
      9 |         self.assert_eq(psdf.swapaxes("index", "columns"), pdf.swapaxes("index", "columns"))
-->  10 |         self.assert_eq(psdf.swapaxes("columns", "index"), pdf.swapaxes("columns", "index"))
     11 |         self.assert_eq((psdf + 1).swapaxes(0, 1), (pdf + 1).swapaxes(0, 1))
     12 | 
     13 |         self.assertRaises(AssertionError, lambda: psdf.swapaxes(0, 1, copy=False))
     14 |         self.assertRaises(ValueError, lambda: psdf.swapaxes(0, -1))
```

</details>

---

### `bench_058` — `pandas.DataFrame.iteritems` (pandas)
- **Sample ID**: `numpy_150` | **Line**: 27:31 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `for key, val in tensor.iteritems():`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (numpy_150)</summary>

```python
      1 |     def coverage(self, bin_size=50):
      2 |         '''
      3 |         Compute the coverage of the blend space by the input tensors.
      4 | 
      5 |         Returns NumPy 2D arrays ``(fill, magnitude, src)``. ``fill``
      6 |         indicates how densely filled each "bin" is, from 0.0 (empty)
      7 |         to 1.0 (full). ``magnitude`` accumulates the absolute values
      8 |         of the items within the bin. ``src`` indicates which tensor
      9 |         each item comes from, specified by its index in the
     10 |         ``tensors`` array.  (If multiple tensors write in the same
     11 |         bin, the last one wins.)
     12 |         '''
     13 |         if not isinstance(bin_size, (list, tuple)):
     14 |             bin_size = [bin_size]*self.ndim
     15 | 
     16 |         import numpy
     17 |         src = numpy.zeros(tuple(numpy.ceil(float(items) / float(bins))
     18 |                                 for items, bins in izip(self.shape, bin_size)),
     19 |                           dtype=numpy.uint8)
     20 |         magnitude = numpy.zeros(src.shape)
     21 |         fill = numpy.zeros(src.shape)
     22 |         inc = 1.0 / numpy.product(bin_size)
     23 | 
     24 |         # This loop should look a lot like the one in FakeTensor.
     25 |         labels = self._labels
     26 |         for tensor_idx, tensor in enumerate(self._tensors):
-->  27 |             for key, val in tensor.iteritems():
     28 |                 idx = tuple(label_list.index(label) // bins for label_list, label, bins in izip(labels, key, bin_size))
     29 |                 src[idx] = tensor_idx
     30 |                 fill[idx] += inc
     31 |                 magnitude[idx] += abs(val)
     32 |         return fill, magnitude, src
```

</details>

---

### `bench_059` — `scipy.stats.itemfreq` (scipy)
- **Sample ID**: `scipy_376` | **Line**: 7:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `return scipy.stats.itemfreq(kos)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_376)</summary>

```python
      1 |     def frequencies(self):
      2 |         kos = []
      3 |         for solution in self.solutions:
      4 |             for ko in solution.knockout_list:
      5 |                 kos.append(ko.id)
      6 | 
-->   7 |         return scipy.stats.itemfreq(kos)
```

</details>

---

### `bench_060` — `scipy.integrate.simps` (scipy)
- **Sample ID**: `scipy_1868` | **Line**: 30:28 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `v_w_im = simps(f_im,dx=dt)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1868)</summary>

```python
      1 | def fourier_t2w(v_t,tgrd,wgrd,method):
      2 |     
      3 |     methods = ['linear','quadratic','cubic','simpson']
      4 |     assert (method in methods), '@fourier_t2w. This method does not exist'
      5 |     
      6 |     xj = 1.0j
      7 |     Nt = len(v_t)-1
      8 |     dt = tgrd[1]-tgrd[0]
      9 |     
     10 |     v_w = np.zeros(len(wgrd),dtype='complex')
     11 | 
     12 |     if method == 'linear':
     13 |         v_w = linear_fourier_t2w(v_t,tgrd,wgrd)
     14 | 
     15 |     elif method == 'quadratic' or method == 'cubic':
     16 |  
     17 |         f_t = interpolate.interp1d(tgrd,v_t, kind=method, bounds_error=False,fill_value=0.0)
     18 | 
     19 |         for iw in range(len(wgrd)):
     20 |             def f_t_exp_iwt(t):
     21 |                 return f_t(t)*np.exp(xj*wgrd[iw]*t)
     22 | 
     23 |             v_w[iw],error = complex_quad(f_t_exp_iwt,tgrd[0],tgrd[-1])
     24 |     
     25 |     elif method == 'simpson':
     26 |         for iw in range(len(wgrd)):
     27 |             f_re = np.cos(wgrd[iw]*tgrd[:])*v_t[:]
     28 |             f_im = np.sin(wgrd[iw]*tgrd[:])*v_t[:]
     29 |             v_w_re = simps(f_re,dx=dt)
-->  30 |             v_w_im = simps(f_im,dx=dt)
     31 |             v_w[iw] = v_w_re + xj*v_w_im
     32 |                 
     33 |     return v_w
```

</details>

---

### `bench_061` — `numpy.alltrue` (numpy)
- **Sample ID**: `numpy_1577` | **Line**: 6:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `assert_(np.alltrue(b == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (numpy_1577)</summary>

```python
      1 |     def test_fromiter_bytes(self):
      2 |         """Ticket #1058"""
      3 |         a = np.fromiter(list(range(10)), dtype='b')
      4 |         b = np.fromiter(list(range(10)), dtype='B')
      5 |         assert_(np.alltrue(a == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
-->   6 |         assert_(np.alltrue(b == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
```

</details>

---

### `bench_062` — `scipy.integrate.simps` (scipy)
- **Sample ID**: `scipy_1778` | **Line**: 26:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `sum2[j] = scipy.integrate.simps(ftemp2[:lastn[j]+1], dx=step)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1778)</summary>

```python
      1 | def _cumul_simpson(theta, cl, nmax, thetamax, cl_mask=None):
      2 |     '''
      3 |     Notes
      4 |     -----
      5 |     This calculates the two integrals in Eq. 90
      6 |     '''
      7 | 
      8 |     npoints = 2.0**(nmax-1)
      9 |     step = thetamax / npoints
     10 | 
     11 |     beta = np.linspace(0.0, thetamax, num=npoints+1)
     12 | 
     13 |     c_plus = _cplus(np.cos(beta), cl, cl_mask=cl_mask)
     14 |     ftemp1 = np.sin(beta)/np.cos(beta/2.0)**4 * c_plus
     15 |     ftemp2 = np.tan(beta/2.0)**3 * c_plus
     16 | 
     17 |     nell = len(cl[0])
     18 | 
     19 |     sum1 = np.zeros(nell)
     20 |     sum2 = np.zeros(nell)
     21 |     lastn = 2*np.array(np.round(theta/(2.0*step)), dtype=int)
     22 | 
     23 |     for j in range(nell):
     24 |         if lastn[j] != 0:
     25 |             sum1[j] = scipy.integrate.simps(ftemp1[:lastn[j]+1], dx=step)
-->  26 |             sum2[j] = scipy.integrate.simps(ftemp2[:lastn[j]+1], dx=step)
     27 | 
     28 |     #Since beta[lastn[j]] is not exactly theta[j], we apply the basic Simpson's rule on
     29 |     #the small interval between those two values
     30 |     step2 = (theta - step*lastn) / 2.0
     31 |     theta0 = theta-step2
     32 |     theta1 = theta
     33 | 
     34 |     c_plus0 = _cplus(np.cos(theta0), cl, cl_mask=cl_mask)
     35 |     c_plus1 = _cplus(np.cos(theta1), cl, cl_mask=cl_mask)
     36 | 
     37 |     int1_0 = np.sin(theta0)/np.cos(theta0/2.0)**4 * c_plus0
     38 |     int1_1 = np.sin(theta1)/np.cos(theta1/2.0)**4 * c_plus1
     39 |     int2_0 = np.tan(theta0/2.0)**3 * c_plus0
     40 |     int2_1 = np.tan(theta1/2.0)**3 * c_plus1
     41 | 
     42 |     endpoint = step2*(ftemp1[lastn] + 4.0 * int1_0 + int1_1) / 3.0
     43 |     sum1 += endpoint
     44 | 
     45 |     endpoint = step2*(ftemp2[lastn] + 4.0 * int2_0 + int2_1) / 3.0
     46 |     sum2 += endpoint
     47 | 
     48 |     return sum1, sum2
```

</details>

---

### `bench_063` — `pandas.io.formats.style.Styler.render` (pandas)
- **Sample ID**: `pandas_13` | **Line**: 7:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `'level0 row0" rowspan="2">l0</th>' in s.render()`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_13)</summary>

```python
      1 |     def test_rowspan_w3(self):
      2 |         # GH 38533
      3 |         df = DataFrame(data=[[1, 2]], index=[["l0", "l0"], ["l1a", "l1b"]])
      4 |         s = Styler(df, uuid="_", cell_ids=False)
      5 |         assert (
      6 |             '<th id="T___level0_row0" class="row_heading '
-->   7 |             'level0 row0" rowspan="2">l0</th>' in s.render()
      8 |         )
```

</details>

---

### `bench_064` — `numpy.product` (numpy)
- **Sample ID**: `numpy_246` | **Line**: 41:30 | **Stratum**: `hard_negative` (`stdlib_name_collision_product`)
- **Call Site**: `for matsize, batchdims in itertools.product([0, 3, 5], [(0,), (3,), (5, 3)]):`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: itertools.product standard library collision.*

<details>
<summary>View Enclosing Code (numpy_246)</summary>

```python
      1 |     def test_slogdet(self, device, dtype):
      2 |         from torch.testing._internal.common_utils import (random_hermitian_matrix, random_hermitian_psd_matrix,
      3 |                                                           random_hermitian_pd_matrix, random_square_matrix_of_rank)
      4 | 
      5 |         # mat_chars denotes matrix characteristics
      6 |         # possible values are: hermitian, hermitian_psd, hermitian_pd, singular, non_singular
      7 |         def run_test(matsize, batchdims, mat_chars):
      8 |             num_matrices = np.prod(batchdims)
      9 |             list_of_matrices = []
     10 |             if num_matrices != 0:
     11 |                 for idx in range(num_matrices):
     12 |                     mat_type = idx % len(mat_chars)
     13 |                     if mat_chars[mat_type] == 'hermitian':
     14 |                         list_of_matrices.append(random_hermitian_matrix(matsize, dtype=dtype, device=device))
     15 |                     elif mat_chars[mat_type] == 'hermitian_psd':
     16 |                         list_of_matrices.append(random_hermitian_psd_matrix(matsize, dtype=dtype, device=device))
     17 |                     elif mat_chars[mat_type] == 'hermitian_pd':
     18 |                         list_of_matrices.append(random_hermitian_pd_matrix(matsize, dtype=dtype, device=device))
     19 |                     elif mat_chars[mat_type] == 'singular':
     20 |                         list_of_matrices.append(torch.ones(matsize, matsize, dtype=dtype, device=device))
     21 |                     elif mat_chars[mat_type] == 'non_singular':
     22 |                         list_of_matrices.append(random_square_matrix_of_rank(matsize, matsize, dtype=dtype, device=device))
     23 |                 full_tensor = torch.stack(list_of_matrices, dim=0).reshape(batchdims + (matsize, matsize))
     24 |             else:
     25 |                 full_tensor = torch.randn(*batchdims, matsize, matsize, dtype=dtype, device=device)
     26 | 
     27 |             actual_value = torch.linalg.slogdet(full_tensor)
     28 |             expected_value = np.linalg.slogdet(full_tensor.cpu().numpy())
     29 |             self.assertEqual(expected_value[0], actual_value[0], atol=self.precision, rtol=self.precision)
     30 |             self.assertEqual(expected_value[1], actual_value[1], atol=self.precision, rtol=self.precision)
     31 | 
     32 |             # test out=variant
     33 |             sign_out = torch.empty_like(actual_value[0])
     34 |             logabsdet_out = torch.empty_like(actual_value[1])
     35 |             ans = torch.linalg.slogdet(full_tensor, out=(sign_out, logabsdet_out))
     36 |             self.assertEqual(ans[0], sign_out)
     37 |             self.assertEqual(ans[1], logabsdet_out)
     38 |             self.assertEqual(sign_out, actual_value[0])
     39 |             self.assertEqual(logabsdet_out, actual_value[1])
     40 | 
-->  41 |         for matsize, batchdims in itertools.product([0, 3, 5], [(0,), (3,), (5, 3)]):
     42 |             run_test(matsize, batchdims, mat_chars=['hermitian_pd'])
     43 |             run_test(matsize, batchdims, mat_chars=['singular'])
     44 |             run_test(matsize, batchdims, mat_chars=['non_singular'])
     45 |             run_test(matsize, batchdims, mat_chars=['hermitian', 'hermitian_pd', 'hermitian_psd'])
     46 |             run_test(matsize, batchdims, mat_chars=['singular', 'non_singular'])
```

</details>

---

### `bench_065` — `scipy.misc.logsumexp` (scipy)
- **Sample ID**: `scipy_535` | **Line**: 8:21 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `misc.logsumexp(self.model_and_run_length_log_distr[m,:]))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_535)</summary>

```python
      1 |     def update_priors(self, t):
      2 |         """update the priors for each model in the model universe, provided 
      3 |         that the boolean auto_prior_update is true"""
      4 |         for (m, model) in zip(range(0,self.Q), self.model_universe):
      5 |             if model.auto_prior_update == True:
      6 |                 """We need to weigh quantities acc. to rld"""
      7 |                 model_specific_rld = (self.model_and_run_length_log_distr[m,:] -
-->   8 |                     misc.logsumexp(self.model_and_run_length_log_distr[m,:]))
      9 |                 model.prior_update(t, model_specific_rld)
```

</details>

---

### `bench_066` — `pandas.DataFrame.iteritems` (pandas)
- **Sample ID**: `pandas_58` | **Line**: 5:52 | **Stratum**: `hard_negative` (`wrapper_lookalike_pyspark`)
- **Call Site**: `for (p_name, p_items), (k_name, k_items) in zip(pser.items(), psser.items()):`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: third-party PySpark/Koalas DataFrame/Series method.*

<details>
<summary>View Enclosing Code (pandas_58)</summary>

```python
      1 |     def test_items(self):
      2 |         pser = pd.Series(["A", "B", "C"])
      3 |         psser = ps.from_pandas(pser)
      4 | 
-->   5 |         for (p_name, p_items), (k_name, k_items) in zip(pser.items(), psser.items()):
      6 |             self.assert_eq(p_name, k_name)
      7 |             self.assert_eq(p_items, k_items)
```

</details>

---

### `bench_067` — `scipy.signal.hanning` (scipy)
- **Sample ID**: `scipy_314` | **Line**: 15:21 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `qfactor = np.convolve(qfactor, scipy.signal.hanning(sw*2)/sw, mode='same')`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_314)</summary>

```python
      1 | def repetition(SSM):
      2 |     sw = 20 # The search range (search width) to find repetition in the SSM        
      3 |     qfactor = np.zeros(SSM.shape[0]) 
      4 |     for i in range(1,int(SSM.shape[0]-sw/2-1)):
      5 |         trigger = 0
      6 |         num_of_peaks = 0
      7 |         for j in range(i+1, min(i+sw, SSM.shape[0]-1)): #max(i-sw,1)
      8 |             if SSM[i,j]>SSM[i,j-1] and SSM[i,j]>=np.amax(SSM)/2.0:
      9 |                 trigger = 1
     10 |             elif SSM[i,j]<SSM[i,j-1] and trigger == 1:
     11 |                 num_of_peaks = num_of_peaks + 1
     12 |                 trigger = 0
     13 |         qfactor[int(i+sw/2)]= np.mean(SSM[i,i+1:min(i+sw, SSM.shape[0]-1)])*(num_of_peaks) # max(i-sw,1)#        
     14 | 
-->  15 |     qfactor = np.convolve(qfactor, scipy.signal.hanning(sw*2)/sw, mode='same')
     16 |     qfactor = qfactor/np.amax(qfactor)
     17 |     
     18 |     return qfactor
```

</details>

---

### `bench_068` — `scipy.stats.betai` (scipy)
- **Sample ID**: `scipy_471` | **Line**: 18:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `return (F,scipy.stats.betai(0.5 * dfwn, 0.5 * dfnum, dfwn/float(dfwn+dfnum*F)) if F is not ma.masked and dfwn/float(dfwn+dfnum*F) <= 1.0 \`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_471)</summary>

```python
      1 | def aF_oneway(*args, **kwargs):
      2 |     dim = kwargs.get("dim", None)
      3 |     arrays = args
      4 |     means = [ma.mean(a, dim) for a in arrays]
      5 |     vars = [ma.var(a, dim) for a in arrays]
      6 |     lens = [ma.sum(ma.array(ma.ones(a.shape), mask=ma.asarray(a).mask), dim) for a in arrays]
      7 |     alldata = ma.concatenate(arrays, dim if dim is not None else 0)
      8 |     bign =  ma.sum(ma.array(ma.ones(alldata.shape), mask=alldata.mask), dim)
      9 |     sstot = ma.sum(alldata ** 2, dim) - (ma.sum(alldata, dim) ** 2) / bign
     10 |     ssbn = ma.sum([(ma.sum(a, dim) ** 2) / L for a, L in zip(arrays, lens)], dim)
     11 | #    print ma.sum(alldata, dim) ** 2 / bign, ssbn
     12 |     ssbn -= ma.sum(alldata, dim) ** 2 / bign
     13 |     sswn = sstot - ssbn
     14 |     dfbn = dfnum = float(len(args) - 1.0)
     15 |     dfwn = bign - len(args) # + 1.0
     16 |     F = (ssbn / dfbn) / (sswn / dfwn)
     17 |     if F.ndim == 0 and dfwn.ndim == 0:
-->  18 |         return (F,scipy.stats.betai(0.5 * dfwn, 0.5 * dfnum, dfwn/float(dfwn+dfnum*F)) if F is not ma.masked and dfwn/float(dfwn+dfnum*F) <= 1.0 \
     19 |                 and dfwn/float(dfwn+dfnum*F) >= 0.0 else ma.masked)
     20 |     else:
     21 |         prob = [scipy.stats.betai(0.5 * dfden, 0.5 * dfnum, dfden/float(dfden+dfnum*f)) if f is not ma.masked and dfden/float(dfden+dfnum*f) <= 1.0 \
     22 |             and dfden/float(dfden+dfnum*f) >= 0.0 else ma.masked for dfden, f in zip (dfwn, F)]
     23 |         return F, prob
```

</details>

---

### `bench_069` — `numpy.product` (numpy)
- **Sample ID**: `numpy_1029` | **Line**: 14:34 | **Stratum**: `hard_negative` (`stdlib_name_collision_product`)
- **Call Site**: `all_possible_digits = itertools.product(*domain_size_subset)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: itertools.product standard library collision.*

<details>
<summary>View Enclosing Code (numpy_1029)</summary>

```python
      1 |     def estimate_sr(perturbed_datas, fos, all_domain_sizes):
      2 |         n = len(perturbed_datas)
      3 |         dimension = len(all_domain_sizes)
      4 |         result = OrderedDict()
      5 | 
      6 |         for d in range(1, dimension + 1):
      7 | 
      8 |             result[d] = OrderedDict()
      9 |             attr_sets = itertools.combinations(range(dimension), d)
     10 | 
     11 |             for attr_set in attr_sets:
     12 |                 result[d][attr_set] = OrderedDict()
     13 |                 domain_size_subset = [range(all_domain_sizes[x]) for x in attr_set]
-->  14 |                 all_possible_digits = itertools.product(*domain_size_subset)
     15 | 
     16 |                 for digits in all_possible_digits:
     17 |                     same_count = 0
     18 |                     for i in range(n):
     19 |                         same = True
     20 |                         for j in range(len(attr_set)):
     21 |                             if not fos[attr_set[j]].support_sr(perturbed_datas[i][attr_set[j]], digits[j]):
     22 |                                 same = False
     23 |                                 break
     24 |                         same_count += same
     25 | 
     26 |                     minus_count = n * np.prod([fos[x].q for x in attr_set])
     27 |                     for d_ in range(1, d):
     28 |                         count_d_ = 0
     29 |                         chosen_sets_ = itertools.combinations(range(d), d_)
     30 |                         for chosen_set_ in chosen_sets_:
     31 |                             coeff = np.prod([fos[d].q for d in attr_set])
     32 |                             coeff *= np.prod(
     33 |                                 [(fos[attr_set[x]].p / fos[attr_set[x]].q - 1) for x in chosen_set_])
     34 | 
     35 |                             attr_set_ = tuple([attr_set[chosen_set_[k]] for k in range(d_)])
     36 |                             digits_ = tuple([digits[chosen_set_[k]] for k in range(d_)])
     37 | 
     38 |                             count_d_ += result[d_][attr_set_][digits_] * coeff
     39 |                         minus_count += count_d_
     40 | 
     41 |                     est = (same_count - minus_count) / np.prod([(fos[x].p - fos[x].q) for x in attr_set])
     42 |                     result[d][attr_set][digits] = est
     43 | 
     44 |         return result
```

</details>

---

### `bench_070` — `scipy.special.sph_jn` (scipy)
- **Sample ID**: `scipy_990` | **Line**: 8:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `jnf,jnfr = scipy.special.sph_jn(n,beta)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_990)</summary>

```python
      1 | def wavefunction():  # computes internal r<1 and externalr>1 wavef.
      2 |      delta = phaseshifts(n,alpha,beta)
      3 |      BL = np.zeros((n),complex)
      4 |      Rin = np.zeros((n,Nin),float) #internal wavefunction r<1 for n partwaves
      5 |      Rex = np.zeros((n,nexpts),float)
      6 |      for i in range (0,10):     # to find BL for matching
      7 |           jnb,jnpr = scipy.special.sph_jn(n,alpha)# SphBessel functions
-->   8 |           jnf,jnfr = scipy.special.sph_jn(n,beta)
      9 |           ynb,yprb = r = scipy.special.sph_yn(n,beta)
     10 |           cosd = cos(delta[i])
     11 |           sind = sin(delta[i])
     12 |           zz = complex(cosd,-sind)
     13 |           num = jnb[i]*zz
     14 |           den = cosd*jnf[i]-sind*ynb[i]
     15 |           BL[i] = num/den  # to match internal and external wavefunctions
     16 |      intr = 1.0/Nin # 50points inside, increment          
     17 |      for i in range(0,n):   # internal wavefunc 
     18 |           rin = intr
     19 |           for ri in range(0,Nin): #plot internal func
     20 |               alpr = alpha*rin
     21 |               jnint,jnintpr = scipy.special.sph_jn(n,alpr)
     22 |               Rin[i,ri] = rin*jnint[i]
     23 |               rin = rin+intr
     24 |      extr = 2./nexpts       # from r =  1 to 3   
     25 |      for i in range(0,n):   
     26 |           rex = 1.0
     27 |           for rx in range(0,nexpts): #plot internal func
     28 |               argu = beta*rex
     29 |               jnxt,jnintpr = scipy.special.sph_jn(n,argu)
     30 |               nxt,jnintpr = scipy.special.sph_yn(n,argu)
     31 |               factr = jnxt[i]*cos(delta[i])-nxt[i]*sin(delta[i])
     32 |               fsin = sin(delta[i])*factr
     33 |               fcos = cos(delta[i])*factr
     34 |               Rex[i,rx] = rex*(fcos*BL.real[i]-fsin*BL.imag[i] )
     35 |               
     36 |               rex = rex+extr       
     37 |      ai = np.arange(0,1,intr)       
     38 |      nwaf = 0  #partial wavef to plotCHANGE FOR OTHER WAVES 1 2 3..
     39 |      f3 = plt.figure()
     40 |      ax3 = f3.add_subplot(111)
     41 |      plt.plot(ai,Rin[nwaf, :])  # only plot s wavefunction
     42 |      ae = np.arange(1,3,extr)   
     43 |      plt.title("  r * partial wavefunction S ")
     44 |      plt.xlabel ("r")
     45 |      plt.plot(ae,Rex[nwaf, :])
```

</details>

---

### `bench_071` — `pandas.DataFrame.first` (pandas)
- **Sample ID**: `pandas_83` | **Line**: 9:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `df_equals(modin_result, pandas_df.first("3D"))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_83)</summary>

```python
      1 | def test_first():
      2 |     i = pd.date_range("2010-04-09", periods=400, freq="2D")
      3 |     modin_df = pd.DataFrame({"A": list(range(400)), "B": list(range(400))}, index=i)
      4 |     pandas_df = pandas.DataFrame(
      5 |         {"A": list(range(400)), "B": list(range(400))}, index=i
      6 |     )
      7 |     with pytest.warns(FutureWarning, match="first is deprecated and will be removed"):
      8 |         modin_result = modin_df.first("3D")
-->   9 |     df_equals(modin_result, pandas_df.first("3D"))
     10 |     df_equals(modin_df.first("20D"), pandas_df.first("20D"))
```

</details>

---

### `bench_072` — `scipy.special.errprint` (scipy)
- **Sample ID**: `scipy_267` | **Line**: 4:15 | **Stratum**: `pipeline_miss` (`cython_native_ufunc`)
- **Call Site**: `flag = sc.errprint(True)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Pipeline missed: compiled Cython native ufunc unparseable by AST.*

<details>
<summary>View Enclosing Code (scipy_267)</summary>

```python
      1 | def test_errprint():
      2 |     with suppress_warnings() as sup:
      3 |         sup.filter(DeprecationWarning, "`errprint` is deprecated!")
-->   4 |         flag = sc.errprint(True)
      5 | 
      6 |     try:
      7 |         assert_(isinstance(flag, bool))
      8 |         with pytest.warns(sc.SpecialFunctionWarning):
      9 |             sc.loggamma(0)
     10 |     finally:
     11 |         with suppress_warnings() as sup:
     12 |             sup.filter(DeprecationWarning, "`errprint` is deprecated!")
     13 |             sc.errprint(flag)
```

</details>

---

### `bench_073` — `pandas.Series.iteritems` (pandas)
- **Sample ID**: `pandas_57` | **Line**: 3:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `for el, item in s.iteritems():`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_57)</summary>

```python
      1 |     def test_float_index_at_iat(self):
      2 |         s = pd.Series([1, 2, 3], index=[0.1, 0.2, 0.3])
-->   3 |         for el, item in s.iteritems():
      4 |             assert s.at[el] == item
      5 |         for i in range(len(s)):
      6 |             assert s.iat[i] == i + 1
```

</details>

---

### `bench_074` — `numpy.cumproduct` (numpy)
- **Sample ID**: `numpy_3461` | **Line**: 17:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `self.coefs = np.cumproduct((1,) + shape_[:-1])`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (numpy_3461)</summary>

```python
      1 |     def __init__(self, shape_, order='C', **keywords):
      2 |         shape_ = tointtuple(shape_)
      3 |         ndim = len(shape_)
      4 |         if ndim == 1:
      5 |             raise NotImplementedError('ndim == 1 is not implemented.')
      6 |         if order.upper() not in ('C', 'F'):
      7 |             raise ValueError(f"Invalid order {order!r}. Expected order is 'C' or 'F'.")
      8 |         order = order.upper()
      9 | 
     10 |         Operator.__init__(self, **keywords)
     11 |         self.shape_ = shape_
     12 |         self.order = order
     13 |         self.ndim = ndim
     14 |         if order == 'C':
     15 |             self.coefs = np.cumproduct((1,) + shape_[:0:-1])[::-1]
     16 |         elif order == 'F':
-->  17 |             self.coefs = np.cumproduct((1,) + shape_[:-1])
```

</details>

---

### `bench_075` — `pandas.DataFrame.last` (pandas)
- **Sample ID**: `pandas_88` | **Line**: 6:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `self.assert_eq(pdf.last(DateOffset(days=1)), kdf.last(DateOffset(days=1)))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_88)</summary>

```python
      1 |     def test_last(self):
      2 |         index = pd.date_range("2018-04-09", periods=4, freq="2D")
      3 |         pdf = pd.DataFrame([1, 2, 3, 4], index=index)
      4 |         kdf = ks.from_pandas(pdf)
      5 |         self.assert_eq(pdf.last("1D"), kdf.last("1D"))
-->   6 |         self.assert_eq(pdf.last(DateOffset(days=1)), kdf.last(DateOffset(days=1)))
      7 |         with self.assertRaisesRegex(TypeError, "'last' only supports a DatetimeIndex"):
      8 |             ks.DataFrame([1, 2, 3, 4]).last("1D")
```

</details>

---

### `bench_076` — `pandas.DataFrame.iteritems` (pandas)
- **Sample ID**: `pandas_124` | **Line**: 17:8 | **Stratum**: `hard_negative` (`wrapper_lookalike_pyspark`)
- **Call Site**: `pdf.pad(inplace=True)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: third-party PySpark/Koalas DataFrame/Series method.*

<details>
<summary>View Enclosing Code (pandas_124)</summary>

```python
      1 |     def test_pad(self):
      2 |         pdf = pd.DataFrame(
      3 |             {
      4 |                 "A": [None, 3, None, None],
      5 |                 "B": [2, 4, None, 3],
      6 |                 "C": [None, None, None, 1],
      7 |                 "D": [0, 1, 5, 4],
      8 |             },
      9 |             columns=["A", "B", "C", "D"],
     10 |         )
     11 |         psdf = ps.from_pandas(pdf)
     12 | 
     13 |         if LooseVersion(pd.__version__) >= LooseVersion("1.1"):
     14 |             self.assert_eq(pdf.pad(), psdf.pad())
     15 | 
     16 |             # Test `inplace=True`
-->  17 |             pdf.pad(inplace=True)
     18 |             psdf.pad(inplace=True)
     19 |             self.assert_eq(pdf, psdf)
     20 |         else:
     21 |             expected = ps.DataFrame(
     22 |                 {
     23 |                     "A": [None, 3, 3, 3],
     24 |                     "B": [2.0, 4.0, 4.0, 3.0],
     25 |                     "C": [None, None, None, 1],
     26 |                     "D": [0, 1, 5, 4],
     27 |                 },
     28 |                 columns=["A", "B", "C", "D"],
     29 |             )
     30 |             self.assert_eq(expected, psdf.pad())
     31 | 
     32 |             # Test `inplace=True`
     33 |             psdf.pad(inplace=True)
     34 |             self.assert_eq(expected, psdf)
```

</details>

---

### `bench_077` — `scipy.misc.comb` (scipy)
- **Sample ID**: `scipy_107` | **Line**: 6:19 | **Stratum**: `hard_negative` (`replacement_overload_comb`)
- **Call Site**: `result += scipy.special.comb(N+n, n) * scipy.special.comb(2*N+1, N-n) * (-x)**n`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: modern replacement scipy.special.comb.*

<details>
<summary>View Enclosing Code (scipy_107)</summary>

```python
      1 | def smoothstep(x, NN=1., xmin=0., xmax=1.):
      2 |     N = math.ceil(NN)
      3 |     x = np.clip((x - xmin) / (xmax - xmin), 0, 1)
      4 |     result = 0
      5 |     for n in range(0, N+1):
-->   6 |          result += scipy.special.comb(N+n, n) * scipy.special.comb(2*N+1, N-n) * (-x)**n
      7 |     result *= x**(N+1)
      8 |     if NN != N: result = (x + result) / 2
      9 |     return result
```

</details>

---

### `bench_078` — `scipy.integrate.cumtrapz` (scipy)
- **Sample ID**: `scipy_1366` | **Line**: 38:32 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `spec_cdf = np.hstack((np.zeros(1), cumtrapz(emp_spect, freq)))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1366)</summary>

```python
      1 |     def initialize_from_data_empspect(self, train_x: torch.Tensor, train_y: torch.Tensor):
      2 |         """
      3 |         Initialize mixture components based on the empirical spectrum of the data.
      4 |         This will often be better than the standard initialize_from_data method, but it assumes
      5 |         that your inputs are evenly spaced.
      6 | 
      7 |         :param torch.Tensor train_x: Training inputs
      8 |         :param torch.Tensor train_y: Training outputs
      9 |         """
     10 | 
     11 |         import numpy as np
     12 |         from scipy.fftpack import fft
     13 |         from scipy.integrate import cumtrapz
     14 | 
     15 |         with torch.no_grad():
     16 |             if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
     17 |                 raise RuntimeError("train_x and train_y should be tensors")
     18 |             if train_x.ndimension() == 1:
     19 |                 train_x = train_x.unsqueeze(-1)
     20 |             if self.active_dims is not None:
     21 |                 train_x = train_x[..., self.active_dims]
     22 | 
     23 |             # Flatten batch dimensions
     24 |             train_x = train_x.view(-1, train_x.size(-1))
     25 |             train_y = train_y.view(-1)
     26 | 
     27 |             N = train_x.size(-2)
     28 |             emp_spect = np.abs(fft(train_y.cpu().detach().numpy())) ** 2 / N
     29 |             M = math.floor(N / 2)
     30 | 
     31 |             freq1 = np.arange(M + 1)
     32 |             freq2 = np.arange(-M + 1, 0)
     33 |             freq = np.hstack((freq1, freq2)) / N
     34 |             freq = freq[: M + 1]
     35 |             emp_spect = emp_spect[: M + 1]
     36 | 
     37 |             total_area = np.trapz(emp_spect, freq)
-->  38 |             spec_cdf = np.hstack((np.zeros(1), cumtrapz(emp_spect, freq)))
     39 |             spec_cdf = spec_cdf / total_area
     40 | 
     41 |             a = np.random.rand(1000, self.ard_num_dims)
     42 |             p, q = np.histogram(a, spec_cdf)
     43 |             bins = np.digitize(a, q)
     44 |             slopes = (spec_cdf[bins] - spec_cdf[bins - 1]) / (freq[bins] - freq[bins - 1])
     45 |             intercepts = spec_cdf[bins - 1] - slopes * freq[bins - 1]
     46 |             inv_spec = (a - intercepts) / slopes
     47 | 
     48 |             from sklearn.mixture import GaussianMixture
     49 | 
     50 |             GMM = GaussianMixture(n_components=self.num_mixtures, covariance_type="diag").fit(inv_spec)
     51 |             means = GMM.means_
     52 |             varz = GMM.covariances_
     53 |             weights = GMM.weights_
     54 | 
     55 |             dtype = self.raw_mixture_means.dtype
     56 |             device = self.raw_mixture_means.device
     57 |             self.mixture_means = torch.tensor(means, dtype=dtype, device=device).unsqueeze(-2)
     58 |             self.mixture_scales = torch.tensor(varz, dtype=dtype, device=device).unsqueeze(-2)
     59 |             self.mixture_weights = torch.tensor(weights, dtype=dtype, device=device)
```

</details>

---

### `bench_079` — `pandas.Series.pad` (pandas)
- **Sample ID**: `pandas_131` | **Line**: 17:15 | **Stratum**: `hard_negative` (`replacement_overload_pandas`)
- **Call Site**: `expected = df.ffill()`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: modern replacement ffill.*

<details>
<summary>View Enclosing Code (pandas_131)</summary>

```python
      1 |     def test_fillna_inplace(self):
      2 |         df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
      3 |         df.loc[:4, 1] = np.nan
      4 |         df.loc[-4:, 3] = np.nan
      5 | 
      6 |         expected = df.fillna(value=0)
      7 |         assert expected is not df
      8 | 
      9 |         df.fillna(value=0, inplace=True)
     10 |         tm.assert_frame_equal(df, expected)
     11 | 
     12 |         expected = df.fillna(value={0: 0}, inplace=True)
     13 |         assert expected is None
     14 | 
     15 |         df.loc[:4, 1] = np.nan
     16 |         df.loc[-4:, 3] = np.nan
-->  17 |         expected = df.ffill()
     18 |         assert expected is not df
     19 | 
     20 |         df.ffill(inplace=True)
     21 |         tm.assert_frame_equal(df, expected)
```

</details>

---

### `bench_080` — `scipy.misc.factorial2` (scipy)
- **Sample ID**: `scipy_1928` | **Line**: 12:23 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `c = 1.0 * factorial2(n-3)**2 / np.sqrt(2*np.pi*np.math.factorial(n))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1928)</summary>

```python
      1 |     def get_initializations(self, num_pol = 5, copy_fun = 'relu'):
      2 |         k = []
      3 |         if copy_fun == 'relu':
      4 |             for n in range(num_pol):
      5 |                 if n == 0:
      6 |                     k.append(1.0/np.sqrt(2*np.pi))
      7 |                 elif n == 1:
      8 |                     k.append(1.0/2)
      9 |                 elif n == 2:
     10 |                     k.append(1.0/np.sqrt(4*np.pi))
     11 |                 elif n > 2 and n % 2 == 0:
-->  12 |                     c = 1.0 * factorial2(n-3)**2 / np.sqrt(2*np.pi*np.math.factorial(n))
     13 |                     k.append(c)
     14 |                 elif n >= 2 and n % 2 != 0:
     15 |                     k.append(0.0)
     16 |         return k
```

</details>

---

### `bench_081` — `scipy.stats.rvs_ratio_uniforms` (scipy)
- **Sample ID**: `scipy_458` | **Line**: 7:10 | **Stratum**: `pipeline_miss` (`preamble_filter_omission`)
- **Call Site**: `rvs = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=2500,`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Pipeline missed: receiver 'stats' omitted from preamble, callee dropped by manifest short-name filter.*

<details>
<summary>View Enclosing Code (scipy_458)</summary>

```python
      1 |     def test_rv_generation(self):
      2 |         # use KS test to check distribution of rvs
      3 |         # normal distribution
      4 |         f = stats.norm.pdf
      5 |         v_bound = np.sqrt(f(np.sqrt(2))) * np.sqrt(2)
      6 |         umax, vmin, vmax = np.sqrt(f(0)), -v_bound, v_bound
-->   7 |         rvs = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=2500,
      8 |                                        random_state=12345)
      9 |         assert_equal(stats.kstest(rvs, 'norm')[1] > 0.25, True)
     10 | 
     11 |         # exponential distribution
     12 |         rvs = stats.rvs_ratio_uniforms(lambda x: np.exp(-x), umax=1,
     13 |                                        vmin=0, vmax=2*np.exp(-1),
     14 |                                        size=1000, random_state=12345)
     15 |         assert_equal(stats.kstest(rvs, 'expon')[1] > 0.25, True)
```

</details>

---

### `bench_082` — `scipy.integrate.simps` (scipy)
- **Sample ID**: `scipy_1894` | **Line**: 18:16 | **Stratum**: `hard_negative` (`replacement_overload_integrate`)
- **Call Site**: `intg2 = integrate.simpson(tmp_y, tmp_x, tmp_x[1] - tmp_x[0]) / L`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: modern replacement simpson.*

<details>
<summary>View Enclosing Code (scipy_1894)</summary>

```python
      1 | def do_average(L, x, S):
      2 |     AVG = []
      3 |     XX = []
      4 |     for xx in x:
      5 |         if xx - L / 2.0 < x[0]:
      6 |             continue
      7 |         if xx + L / 2.0 > x[-1]:
      8 |             continue
      9 |         XX.append(xx)
     10 |         # print('step',xx)
     11 |         tmp_x = np.arange(xx - L / 2.0, xx + L / 2.0, 0.001)
     12 |         # x=np.arange(xx-L/2.0, xx+L/2.0,0.1)
     13 |         tmp_y = S(tmp_x)
     14 |         # intg2=np.trapz(tmp_y,tmp_x,tmp_x[1]-tmp_x[0])/L
     15 |         # intg2=integrate.quad(S, xx-L/2.0, xx+L/2.0)[0] / L
     16 |         # intg2=integrate.quad(S, xx-L/2.0, xx+L/2.0)[0] / L
     17 |         # print('tmp_x[1]-tmp_x[0]',tmp_x[1]-tmp_x[0])
-->  18 |         intg2 = integrate.simpson(tmp_y, tmp_x, tmp_x[1] - tmp_x[0]) / L
     19 |         # print('intg',int)
     20 |         AVG.append(intg2)  # integration
     21 |     #        print("int ", integrate.quad(S, xx-L/2.0, xx+L/2.0))
     22 |     return XX, AVG
```

</details>

---

### `bench_083` — `scipy.special.errprint` (scipy)
- **Sample ID**: `scipy_275` | **Line**: 51:13 | **Stratum**: `pipeline_miss` (`cython_native_ufunc`)
- **Call Site**: `sv = special.errprint(sv)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Pipeline missed: compiled Cython native ufunc unparseable by AST.*

<details>
<summary>View Enclosing Code (scipy_275)</summary>

```python
      1 | def comb(N,k,exact=0):
      2 |     """
      3 |     The number of combinations of N things taken k at a time.
      4 | 
      5 |     This is often expressed as "N choose k".
      6 | 
      7 |     Parameters
      8 |     ----------
      9 |     N : int, ndarray
     10 |         Number of things.
     11 |     k : int, ndarray
     12 |         Number of elements taken.
     13 |     exact : int, optional
     14 |         If `exact` is 0, then floating point precision is used, otherwise
     15 |         exact long integer is computed.
     16 | 
     17 |     Returns
     18 |     -------
     19 |     val : int, ndarray
     20 |         The total number of combinations.
     21 | 
     22 |     Notes
     23 |     -----
     24 |     - Array arguments accepted only for exact=0 case.
     25 |     - If k > N, N < 0, or k < 0, then a 0 is returned.
     26 | 
     27 |     Examples
     28 |     --------
     29 |     >>> k = np.array([3, 4])
     30 |     >>> n = np.array([10, 10])
     31 |     >>> sc.comb(n, k, exact=False)
     32 |     array([ 120.,  210.])
     33 |     >>> sc.comb(10, 3, exact=True)
     34 |     120L
     35 | 
     36 |     """
     37 |     if exact:
     38 |         if (k > N) or (N < 0) or (k < 0):
     39 |             return 0
     40 |         val = 1
     41 |         for j in xrange(min(k, N-k)):
     42 |             val = (val*(N-j))//(j+1)
     43 |         return val
     44 |     else:
     45 |         from scipy import special
     46 |         k,N = asarray(k), asarray(N)
     47 |         lgam = special.gammaln
     48 |         cond = (k <= N) & (N >= 0) & (k >= 0)
     49 |         sv = special.errprint(0)
     50 |         vals = exp(lgam(N+1) - lgam(N-k+1) - lgam(k+1))
-->  51 |         sv = special.errprint(sv)
     52 |         return where(cond, vals, 0.0)
```

</details>

---

### `bench_084` — `scipy.misc.comb` (scipy)
- **Sample ID**: `scipy_201` | **Line**: 29:26 | **Stratum**: `hard_negative` (`replacement_overload_comb`)
- **Call Site**: `real_pairs += scipy.special.comb(count, 2)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: modern replacement scipy.special.comb.*

<details>
<summary>View Enclosing Code (scipy_201)</summary>

```python
      1 | def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
      2 |     """ Compute accuracy metrics between log parsing results and ground truth
      3 |     
      4 |     Arguments
      5 |     ---------
      6 |         series_groundtruth : pandas.Series
      7 |             A sequence of groundtruth event Ids
      8 |         series_parsedlog : pandas.Series
      9 |             A sequence of parsed event Ids
     10 |         debug : bool, default False
     11 |             print error log messages when set to True
     12 | 
     13 |     Returns
     14 |     -------
     15 |         precision : float
     16 |         recall : float
     17 |         f_measure : float
     18 |         accuracy : float
     19 |     """
     20 |     EVENT = [0, 0]
     21 |     TEXT  = [0, 0]
     22 |     TABLE = [0, 0]
     23 | 
     24 |     series_groundtruth_valuecounts = series_groundtruth.value_counts()
     25 |     # print("series_groundtruth_valuecounts:{}".format(series_groundtruth_valuecounts))
     26 |     real_pairs = 0
     27 |     for count in series_groundtruth_valuecounts:
     28 |         if count > 1:
-->  29 |             real_pairs += scipy.special.comb(count, 2)
     30 |     # print("real_pairs:{}".format(real_pairs))
     31 |     series_parsedlog_valuecounts = series_parsedlog.value_counts()
     32 |     # print("series_parsedlog_valuecounts:{}".format(series_parsedlog_valuecounts))
     33 |     parsed_pairs = 0
     34 |     for count in series_parsedlog_valuecounts:
     35 |         if count > 1:
     36 |             parsed_pairs += scipy.special.comb(count, 2)
     37 |     # print("parsed_pairs:{}".format(parsed_pairs))
     38 |     accurate_pairs = 0
     39 |     accurate_events = 0 # determine how many lines are correctly parsed
     40 |     accurate_templates = 0
     41 |     for parsed_eventId in series_parsedlog_valuecounts.index:
     42 |         logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
     43 |         # print("logIds:{}".format(logIds))
     44 |         series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
     45 |         # print("series_groundtruth_logId_valuecounts:{}".format(series_groundtruth_logId_valuecounts))
     46 |         error_eventIds = (parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
     47 |         error = True
     48 |         if series_groundtruth_logId_valuecounts.size == 1:
     49 |             groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
     50 |             # print("groundtruth_eventId:{}".format(groundtruth_eventId))
     51 |             if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
     52 |                 # print("series_groundtruth:{}".format(series_groundtruth))
     53 |                 # print("WTF:{}".format( series_groundtruth[series_groundtruth == groundtruth_eventId]))
     54 |                 # print("logIds.size:{}".format(logIds.size))
     55 | 
     56 |                 if groundtruth_eventId[0] == 's':
     57 |                     EVENT[0] += 1
     58 |                     EVENT[1] += logIds.size
     59 |                 elif groundtruth_eventId[0] == 'm':
     60 |                     TEXT[0] += 1
     61 |                     TEXT[1] += logIds.size
     62 |                 else:
     63 |                     TABLE[0] += 1
     64 |                     TABLE[1] += logIds.size
     65 |                     
     66 | 
     67 |                 accurate_events += logIds.size
     68 |                 accurate_templates += 1
     69 |                 error = False
     70 |         if error and debug:
     71 |             print('(parsed_eventId, groundtruth_eventId) =', error_eventIds, 'failed', logIds.size, 'messages')
     72 |         for count in series_groundtruth_logId_valuecounts:
     73 |             if count > 1:
     74 |                 accurate_pairs += scipy.special.comb(count, 2)
     75 | 
     76 |     precision = float(accurate_templates) / len(series_parsedlog_valuecounts)
     77 |     recall = float(accurate_templates) / len(series_groundtruth_valuecounts)
     78 |     f_measure = 2 * precision * recall / (precision + recall)
     79 |     accuracy = float(accurate_events) / series_groundtruth.size
     80 | 
     81 |     print(EVENT, TEXT, TABLE)
     82 |     return precision, recall, f_measure, accuracy
```

</details>

---

### `bench_085` — `scipy.stats.chisqprob` (scipy)
- **Sample ID**: `scipy_1682` | **Line**: 16:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `p = scipy.stats.chisqprob(chi, df)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1682)</summary>

```python
      1 | def doLogLikelihoodTest(complex_ll, complex_np,
      2 |                         simple_ll, simple_np,
      3 |                         significance_threshold=0.05):
      4 |     """perform log-likelihood test between model1 and model2.
      5 |     """
      6 | 
      7 |     assert complex_ll >= simple_ll, "log likelihood of complex model smaller than for simple model: %f > %f" % (
      8 |         complex_ll, simple_ll)
      9 | 
     10 |     chi = 2 * (complex_ll - simple_ll)
     11 |     df = complex_np - simple_np
     12 | 
     13 |     if df <= 0:
     14 |         raise ValueError("difference of degrees of freedom not larger than 0")
     15 | 
-->  16 |     p = scipy.stats.chisqprob(chi, df)
     17 | 
     18 |     l = LogLikelihoodTest()
     19 | 
     20 |     l.mComplexLogLikelihood = complex_ll
     21 |     l.mSimpleLogLikelihood = simple_ll
     22 |     l.mComplexNumParameters = complex_np
     23 |     l.mSimpleNumParameters = simple_np
     24 |     l.mSignificanceThreshold = significance_threshold
     25 |     l.mProbability = p
     26 |     l.mChiSquaredValue = chi
     27 |     l.mDegreesFreedom = df
     28 | 
     29 |     if p < significance_threshold:
     30 |         l.mPassed = True
     31 |     else:
     32 |         l.mPassed = False
     33 | 
     34 |     return l
```

</details>

---

### `bench_086` — `scipy.special.sph_yn` (scipy)
- **Sample ID**: `scipy_990` | **Line**: 30:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `nxt,jnintpr = scipy.special.sph_yn(n,argu)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_990)</summary>

```python
      1 | def wavefunction():  # computes internal r<1 and externalr>1 wavef.
      2 |      delta = phaseshifts(n,alpha,beta)
      3 |      BL = np.zeros((n),complex)
      4 |      Rin = np.zeros((n,Nin),float) #internal wavefunction r<1 for n partwaves
      5 |      Rex = np.zeros((n,nexpts),float)
      6 |      for i in range (0,10):     # to find BL for matching
      7 |           jnb,jnpr = scipy.special.sph_jn(n,alpha)# SphBessel functions
      8 |           jnf,jnfr = scipy.special.sph_jn(n,beta)
      9 |           ynb,yprb = r = scipy.special.sph_yn(n,beta)
     10 |           cosd = cos(delta[i])
     11 |           sind = sin(delta[i])
     12 |           zz = complex(cosd,-sind)
     13 |           num = jnb[i]*zz
     14 |           den = cosd*jnf[i]-sind*ynb[i]
     15 |           BL[i] = num/den  # to match internal and external wavefunctions
     16 |      intr = 1.0/Nin # 50points inside, increment          
     17 |      for i in range(0,n):   # internal wavefunc 
     18 |           rin = intr
     19 |           for ri in range(0,Nin): #plot internal func
     20 |               alpr = alpha*rin
     21 |               jnint,jnintpr = scipy.special.sph_jn(n,alpr)
     22 |               Rin[i,ri] = rin*jnint[i]
     23 |               rin = rin+intr
     24 |      extr = 2./nexpts       # from r =  1 to 3   
     25 |      for i in range(0,n):   
     26 |           rex = 1.0
     27 |           for rx in range(0,nexpts): #plot internal func
     28 |               argu = beta*rex
     29 |               jnxt,jnintpr = scipy.special.sph_jn(n,argu)
-->  30 |               nxt,jnintpr = scipy.special.sph_yn(n,argu)
     31 |               factr = jnxt[i]*cos(delta[i])-nxt[i]*sin(delta[i])
     32 |               fsin = sin(delta[i])*factr
     33 |               fcos = cos(delta[i])*factr
     34 |               Rex[i,rx] = rex*(fcos*BL.real[i]-fsin*BL.imag[i] )
     35 |               
     36 |               rex = rex+extr       
     37 |      ai = np.arange(0,1,intr)       
     38 |      nwaf = 0  #partial wavef to plotCHANGE FOR OTHER WAVES 1 2 3..
     39 |      f3 = plt.figure()
     40 |      ax3 = f3.add_subplot(111)
     41 |      plt.plot(ai,Rin[nwaf, :])  # only plot s wavefunction
     42 |      ae = np.arange(1,3,extr)   
     43 |      plt.title("  r * partial wavefunction S ")
     44 |      plt.xlabel ("r")
     45 |      plt.plot(ae,Rex[nwaf, :])
```

</details>

---

### `bench_087` — `scipy.misc.comb` (scipy)
- **Sample ID**: `scipy_89` | **Line**: 16:16 | **Stratum**: `hard_negative` (`replacement_overload_comb`)
- **Call Site**: `pref = 1. / scipy.special.comb(L, L // 2 + charge_sector)**0.5`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: modern replacement scipy.special.comb.*

<details>
<summary>View Enclosing Code (scipy_89)</summary>

```python
      1 | def test_canoncial_purification(conserve_ancilla, L=6, charge_sector=0, eps=1.e-14):
      2 |     site = spin_half
      3 |     psi = purification_mps.PurificationMPS.from_infiniteT_canonical(
      4 |         [site] * L, [charge_sector], conserve_ancilla_charge=conserve_ancilla)
      5 |     psi.test_sanity()
      6 |     Szs = psi.expectation_value('Sz')
      7 |     assert abs(sum(Szs) - charge_sector) < 1.e-13
      8 |     total_psi = psi.get_theta(0, L).take_slice(0, 'vL').take_slice(0, 'vR')
      9 |     total_psi.itranspose(['p' + str(i) for i in range(L)] + ['q' + str(i) for i in range(L)])
     10 |     # note: don't `combine_legs`: it will permute the p legs differently than q due to charges
     11 |     total_psi_dense = total_psi.to_ndarray().reshape(2**L, 2**L)
     12 |     # now it should be diagonal
     13 |     diag = np.diag(total_psi_dense)
     14 |     assert np.all(np.abs(total_psi_dense - np.diag(diag) < eps))  # is it diagonal?
     15 |     # and the diagonal should be sqrt(L choose L//2) for states with fitting numbers
-->  16 |     pref = 1. / scipy.special.comb(L, L // 2 + charge_sector)**0.5
     17 |     Q_p = site.leg.to_qflat()[:, 0]
     18 |     for i, entry in enumerate(diag):
     19 |         Q_i = sum([Q_p[int(b)] for b in format(i, 'b').zfill(L)])
     20 |         if Q_i == charge_sector:
     21 |             assert abs(entry - pref) < eps
     22 |         else:
     23 |             assert abs(entry) < eps
     24 | 
     25 |     # and one quick test of TEBD
     26 |     xxz_pars = dict(L=L, Jxx=1., Jz=3., hz=0., bc_MPS='finite', sort_charge=False)
     27 |     # sort_charge should be same as for global `spin_half`.
     28 |     M = XXZChain(xxz_pars)
     29 |     TEBD_params = {
     30 |         'trunc_params': {
     31 |             'chi_max': 16,
     32 |             'svd_min': 1.e-8
     33 |         },
     34 |         'disentangle': None,  # 'renyi' should work as well, 'backwards' not.
     35 |         'dt': 0.1,
     36 |         'N_steps': 2
     37 |     }
     38 |     if conserve_ancilla:
     39 |         M = purification_mps.convert_model_purification_canonical_conserve_ancilla_charge(M)
     40 |     eng = PurificationTEBD(psi, M, TEBD_params)
     41 |     eng.run_imaginary(0.2)
     42 |     eng.run()
     43 |     N = psi.expectation_value('Id')  # check normalization : <1> =?= 1
     44 |     npt.assert_array_almost_equal_nulp(N, np.ones([L]), 100)
```

</details>

---

### `bench_088` — `pandas.DataFrame.select` (pandas)
- **Sample ID**: `pandas_85` | **Line**: 8:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `df1 = df.select(lambda u: u[0] in ['f2', 'f3'], axis=1)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_85)</summary>

```python
      1 |     def test_groupby_level_no_obs(self):
      2 |         # #1697
      3 |         midx = MultiIndex.from_tuples([('f1', 's1'), ('f1', 's2'),
      4 |                                        ('f2', 's1'), ('f2', 's2'),
      5 |                                        ('f3', 's1'), ('f3', 's2')])
      6 |         df = DataFrame(
      7 |             [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], columns=midx)
-->   8 |         df1 = df.select(lambda u: u[0] in ['f2', 'f3'], axis=1)
      9 | 
     10 |         grouped = df1.groupby(axis=1, level=0)
     11 |         result = grouped.sum()
     12 |         self.assertTrue((result.columns == ['f2', 'f3']).all())
```

</details>

---

### `bench_089` — `scipy.stats.chisqprob` (scipy)
- **Sample ID**: `scipy_1676` | **Line**: 30:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `prob = scipy.stats.chisqprob(chi2, dof)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1676)</summary>

```python
      1 |     def _calc_stats(self):
      2 |         pars, pcov = extract_mcmc_stats(self.trials)
      3 |         perr = sqrt(diag(pcov))
      4 | 
      5 |         npars = len(pars)
      6 | 
      7 |         lnprob = self.get_lnprob(pars)
      8 | 
      9 |         chi2 = lnprob / (-0.5)
     10 |         dof = self.x.size - npars
     11 |         chi2per = chi2 / dof
     12 | 
     13 |         aic = -2 * lnprob + 2 * npars
     14 |         bic = -2 * lnprob + npars * log(self.x.size)
     15 | 
     16 |         res = {
     17 |             "pars": pars,
     18 |             "perr": perr,
     19 |             "pcov": pcov,
     20 |             "lnprob": lnprob,
     21 |             "aic": aic,
     22 |             "bic": bic,
     23 |             "chi2": chi2,
     24 |             "dof": dof,
     25 |             "chi2per": chi2per,
     26 |         }
     27 |         try:
     28 |             import scipy.stats
     29 | 
-->  30 |             prob = scipy.stats.chisqprob(chi2, dof)
     31 | 
     32 |             res["prob"] = prob
     33 |         except Exception:
     34 |             pass
     35 | 
     36 |         self._result = res
     37 |         try:
     38 |             self._ply = np.poly1d(res["pars"])
     39 |         except Exception:
     40 |             print("could not set poly")
     41 |             self._ply = None
```

</details>

---

### `bench_090` — `scipy.stats.rvs_ratio_uniforms` (scipy)
- **Sample ID**: `scipy_458` | **Line**: 12:10 | **Stratum**: `pipeline_miss` (`preamble_filter_omission`)
- **Call Site**: `rvs = stats.rvs_ratio_uniforms(lambda x: np.exp(-x), umax=1,`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Pipeline missed: receiver 'stats' omitted from preamble, callee dropped by manifest short-name filter.*

<details>
<summary>View Enclosing Code (scipy_458)</summary>

```python
      1 |     def test_rv_generation(self):
      2 |         # use KS test to check distribution of rvs
      3 |         # normal distribution
      4 |         f = stats.norm.pdf
      5 |         v_bound = np.sqrt(f(np.sqrt(2))) * np.sqrt(2)
      6 |         umax, vmin, vmax = np.sqrt(f(0)), -v_bound, v_bound
      7 |         rvs = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=2500,
      8 |                                        random_state=12345)
      9 |         assert_equal(stats.kstest(rvs, 'norm')[1] > 0.25, True)
     10 | 
     11 |         # exponential distribution
-->  12 |         rvs = stats.rvs_ratio_uniforms(lambda x: np.exp(-x), umax=1,
     13 |                                        vmin=0, vmax=2*np.exp(-1),
     14 |                                        size=1000, random_state=12345)
     15 |         assert_equal(stats.kstest(rvs, 'expon')[1] > 0.25, True)
```

</details>

---

### `bench_091` — `pandas.DataFrame.iteritems` (pandas)
- **Sample ID**: `pandas_127` | **Line**: 19:4 | **Stratum**: `hard_negative` (`wrapper_lookalike_pyspark`)
- **Call Site**: `psdf.ffill(inplace=True)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: third-party PySpark/Koalas DataFrame/Series method.*

<details>
<summary>View Enclosing Code (pandas_127)</summary>

```python
      1 |     def test_ffill(self):
      2 |         idx = np.random.rand(6)
      3 |         pdf = pd.DataFrame(
      4 |             {
      5 |                 "x": [np.nan, 2, 3, 4, np.nan, 6],
      6 |                 "y": [1, 2, np.nan, 4, np.nan, np.nan],
      7 |                 "z": [1, 2, 3, 4, np.nan, np.nan],
      8 |             },
      9 |             index=idx,
     10 |         )
     11 |         psdf = ps.from_pandas(pdf)
     12 | 
     13 |         self.assert_eq(psdf.ffill(), pdf.ffill())
     14 |         self.assert_eq(psdf.ffill(limit=1), pdf.ffill(limit=1))
     15 | 
     16 |         pser = pdf.y
     17 |         psser = psdf.y
     18 | 
-->  19 |         psdf.ffill(inplace=True)
     20 |         pdf.ffill(inplace=True)
     21 | 
     22 |         self.assert_eq(psdf, pdf)
     23 |         self.assert_eq(psser, pser)
     24 |         self.assert_eq(psser[idx[2]], pser[idx[2]])
```

</details>

---

### `bench_092` — `scipy.misc.face` (scipy)
- **Sample ID**: `scipy_2149` | **Line**: 5:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `image_tensor = tf.constant(scipy.misc.face())`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_2149)</summary>

```python
      1 |     def test_plot_many(self):
      2 |         '''1.4 plot_many'''
      3 |         # make a fake batch
      4 |         batch_size = 3
-->   5 |         image_tensor = tf.constant(scipy.misc.face())
      6 |         try:
      7 |             attention_batch = tf.random.gamma([batch_size, 7, 7], alpha=0.3, seed=42)
      8 |         except AttributeError:  # legacy TF versions
      9 |             attention_batch = tf.random_gamma([batch_size, 7, 7], alpha=0.3, seed=42)
     10 | 
     11 |         image_batch = tf.tile(tf.expand_dims(image_tensor, 0),
     12 |                               [batch_size, 1, 1, 1], name='image_batch') # copy
     13 | 
     14 |         plot_op = tfplot.plot_many(_overlay_attention, [attention_batch, image_batch])
     15 |         r = self._execute_plot_op(plot_op, print_image=False)
     16 |         #for i in range(3): imgcat(r[i])
     17 |         self.assertEqual(r.shape, (3, 400, 400, 4))
```

</details>

---

### `bench_093` — `pandas.DataFrame.iteritems` (pandas)
- **Sample ID**: `pandas_71` | **Line**: 9:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `for (p_name, p_items), (k_name, k_items) in zip(pdf.iteritems(), kdf.iteritems()):`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_71)</summary>

```python
      1 |     def test_iteritems(self):
      2 |         pdf = pd.DataFrame(
      3 |             {"species": ["bear", "bear", "marsupial"], "population": [1864, 22000, 80000]},
      4 |             index=["panda", "polar", "koala"],
      5 |             columns=["species", "population"],
      6 |         )
      7 |         kdf = ks.from_pandas(pdf)
      8 | 
-->   9 |         for (p_name, p_items), (k_name, k_items) in zip(pdf.iteritems(), kdf.iteritems()):
     10 |             self.assert_eq(p_name, k_name)
     11 |             self.assert_eq(p_items, k_items)
```

</details>

---

### `bench_094` — `scipy.signal.hanning` (scipy)
- **Sample ID**: `scipy_290` | **Line**: 15:25 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `mX, _ = TimeFrequencyDecomposition.STFT(xn, hanning(self.nfft/2 + 1), self.nfft, self.nfft/4)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_290)</summary>

```python
      1 |     def NMREval(self, xn, xnhat):
      2 |         """ Method to perform NMR perceptual evaluation of audio quality between two signals.
      3 |         Args        :
      4 |             xn      :   (ndarray) 1D Array containing the true time domain signal.
      5 |             xnhat   :   (ndarray) 1D Array containing the estimated time domain signal.
      6 |         Returns     :
      7 |             NMR     :   (float)   A float measurement in dB providing a perceptually weighted
      8 |                         evaluation. Below -9 dB can be considered as in-audible difference/error.
      9 |         As appears in :
     10 |         - K. Brandenburg and T. Sporer,  “NMR and Masking Flag: Evaluation of Quality Using Perceptual Criteria,” in
     11 |         Proceedings of the AES 11th International Conference on Test and Measurement, Portland, USA, May 1992, pp. 169–179
     12 |         - J. Nikunen and T. Virtanen, "Noise-to-mask ratio minimization by weighted non-negative matrix factorization," in
     13 |          Acoustics Speech and Signal Processing (ICASSP), 2010 IEEE International Conference on, Dallas, TX, 2010, pp. 25-28.
     14 |         """
-->  15 |         mX, _ = TimeFrequencyDecomposition.STFT(xn, hanning(self.nfft/2 + 1), self.nfft, self.nfft/4)
     16 |         mXhat, _ = TimeFrequencyDecomposition.STFT(xnhat, hanning(self.nfft/2 + 1), self.nfft, self.nfft/4)
     17 | 
     18 |         # Compute Error
     19 |         Err = np.abs(mX - mXhat) ** 2.
     20 | 
     21 |         # Acquire Masking Threshold
     22 |         mT = self.maskingThreshold(mX)
     23 | 
     24 |         # Inverse the filter of masking threshold
     25 |         imT = 1./(mT + eps)
     26 | 
     27 |         # Outer/Middle Ear transfer function on the diagonal
     28 |         LTq = 10 ** (self.MOEar()/20.)
     29 | 
     30 |         # NMR computation
     31 |         NMR = 10. * np.log10((1./mX.shape[0]) * self._maxb * np.sum((imT * (Err*LTq))))
     32 |         print(NMR)
     33 |         return NMR
```

</details>

---

### `bench_095` — `scipy.misc.comb` (scipy)
- **Sample ID**: `scipy_38` | **Line**: 24:23 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `c[j] += factor * comb(j, k-a)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_38)</summary>

```python
      1 |     def from_power_basis(cls, pp, extrapolate=None):
      2 |         """
      3 |         Construct a piecewise polynomial in Bernstein basis
      4 |         from a power basis polynomial.
      5 | 
      6 |         Parameters
      7 |         ----------
      8 |         pp : PPoly
      9 |             A piecewise polynomial in the power basis
     10 |         extrapolate : bool, optional
     11 |             Whether to extrapolate to ouf-of-bounds points based on first
     12 |             and last intervals, or to return NaNs. Default: True.
     13 | 
     14 |         """
     15 |         dx = np.diff(pp.x)
     16 |         k = pp.c.shape[0] - 1   # polynomial order
     17 | 
     18 |         rest = (None,)*(pp.c.ndim-2)
     19 | 
     20 |         c = np.zeros_like(pp.c)
     21 |         for a in range(k+1):
     22 |             factor = pp.c[a] / comb(k, k-a) * dx[(slice(None),)+rest]**(k-a)
     23 |             for j in range(k-a, k+1):
-->  24 |                 c[j] += factor * comb(j, k-a)
     25 | 
     26 |         if extrapolate is None:
     27 |             extrapolate = pp.extrapolate
     28 | 
     29 |         return cls.construct_fast(c, pp.x, extrapolate)
```

</details>

---

### `bench_096` — `scipy.special.sph_jn` (scipy)
- **Sample ID**: `scipy_992` | **Line**: 3:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `for nn in range(0,n):  jn,jpr = scipy.special.sph_jn(nn,xx)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_992)</summary>

```python
      1 | def Gam(n,xx): # Spherical Bessel ratio
      2 |     gamma = np.zeros((n),float)
-->   3 |     for nn in range(0,n):  jn,jpr = scipy.special.sph_jn(nn,xx)  
      4 |     gamma = alpha*jpr/jn   # gamma match psi outside-inside
      5 |     return gamma
```

</details>

---

### `bench_097` — `scipy.stats.betai` (scipy)
- **Sample ID**: `scipy_470` | **Line**: 13:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `return (t, scipy.stats.betai(0.5*df,0.5,df/(df+t**2)) if t is not ma.masked and df/(df+t**2) <= 1.0 else ma.masked)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_470)</summary>

```python
      1 | def attest_ind(a, b, dim=None):
      2 |     """ Return the t-test statistics on arrays a and b over the dim axis.
      3 |     Returns both the t statistic as well as the p-value
      4 |     """
      5 | #    dim = a.ndim - 1 if dim is None else dim
      6 |     x1, x2 = ma.mean(a, dim), ma.mean(b, dim)
      7 |     v1, v2 = ma.var(a, dim), ma.var(b, dim)
      8 |     n1, n2 = (a.shape[dim], b.shape[dim]) if dim is not None else (a.size, b.size)
      9 |     df = float(n1+n2-2)
     10 |     svar = ((n1-1)*v1+(n2-1)*v2) / df
     11 |     t = (x1-x2)/ma.sqrt(svar*(1.0/n1 + 1.0/n2))
     12 |     if t.ndim == 0:
-->  13 |         return (t, scipy.stats.betai(0.5*df,0.5,df/(df+t**2)) if t is not ma.masked and df/(df+t**2) <= 1.0 else ma.masked)
     14 |     else:
     15 |         prob = [scipy.stats.betai(0.5*df,0.5,df/(df+tsq)) if tsq is not ma.masked and df/(df+tsq) <= 1.0 else ma.masked  for tsq in t*t]
     16 |         return t, prob
```

</details>

---

### `bench_098` — `pandas.Series.pad` (pandas)
- **Sample ID**: `pandas_125` | **Line**: 14:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `self.assert_eq(pdf.pad(), kdf.pad())`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_125)</summary>

```python
      1 |     def test_pad(self):
      2 |         pdf = pd.DataFrame(
      3 |             {
      4 |                 "A": [None, 3, None, None],
      5 |                 "B": [2, 4, None, 3],
      6 |                 "C": [None, None, None, 1],
      7 |                 "D": [0, 1, 5, 4],
      8 |             },
      9 |             columns=["A", "B", "C", "D"],
     10 |         )
     11 |         kdf = ks.from_pandas(pdf)
     12 | 
     13 |         if LooseVersion(pd.__version__) >= LooseVersion("1.1"):
-->  14 |             self.assert_eq(pdf.pad(), kdf.pad())
     15 | 
     16 |             # Test `inplace=True`
     17 |             pdf.pad(inplace=True)
     18 |             kdf.pad(inplace=True)
     19 |             self.assert_eq(pdf, kdf)
     20 |         else:
     21 |             expected = ks.DataFrame(
     22 |                 {
     23 |                     "A": [None, 3, 3, 3],
     24 |                     "B": [2.0, 4.0, 4.0, 3.0],
     25 |                     "C": [None, None, None, 1],
     26 |                     "D": [0, 1, 5, 4],
     27 |                 },
     28 |                 columns=["A", "B", "C", "D"],
     29 |             )
     30 |             self.assert_eq(expected, kdf.pad())
     31 | 
     32 |             # Test `inplace=True`
     33 |             kdf.pad(inplace=True)
     34 |             self.assert_eq(expected, kdf)
```

</details>

---

### `bench_099` — `pandas.DataFrame.iteritems` (pandas)
- **Sample ID**: `pandas_86` | **Line**: 5:19 | **Stratum**: `hard_negative` (`wrapper_lookalike_pyspark`)
- **Call Site**: `self.assert_eq(pdf.last("1D"), psdf.last("1D"))`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: third-party PySpark/Koalas DataFrame/Series method.*

<details>
<summary>View Enclosing Code (pandas_86)</summary>

```python
      1 |     def test_last(self):
      2 |         index = pd.date_range("2018-04-09", periods=4, freq="2D")
      3 |         pdf = pd.DataFrame([1, 2, 3, 4], index=index)
      4 |         psdf = ps.from_pandas(pdf)
-->   5 |         self.assert_eq(pdf.last("1D"), psdf.last("1D"))
      6 |         self.assert_eq(pdf.last(DateOffset(days=1)), psdf.last(DateOffset(days=1)))
      7 |         with self.assertRaisesRegex(TypeError, "'last' only supports a DatetimeIndex"):
      8 |             ps.DataFrame([1, 2, 3, 4]).last("1D")
```

</details>

---

### `bench_100` — `numpy.cumproduct` (numpy)
- **Sample ID**: `numpy_3437` | **Line**: 2:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `self.assert_deprecated(lambda: np.cumproduct(np.array([1, 2, 3])))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (numpy_3437)</summary>

```python
      1 |     def test_cumproduct(self):
-->   2 |         self.assert_deprecated(lambda: np.cumproduct(np.array([1, 2, 3])))
```

</details>

---

### `bench_101` — `scipy.misc.comb` (scipy)
- **Sample ID**: `scipy_192` | **Line**: 4:11 | **Stratum**: `hard_negative` (`replacement_overload_comb`)
- **Call Site**: `return _spspecial.comb(n, r)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: modern replacement scipy.special.comb.*

<details>
<summary>View Enclosing Code (scipy_192)</summary>

```python
      1 | def _nCr(n, r):                                                                           # noqa
      2 |     """Number of combinations of r items out of a set of n.  Equals n!/(r!(n-r)!)"""
      3 |     #f = _math.factorial; return f(n) / f(r) / f(n-r)
-->   4 |     return _spspecial.comb(n, r)
```

</details>

---

### `bench_102` — `pandas.DataFrame.swapaxes` (pandas)
- **Sample ID**: `pandas_33` | **Line**: 11:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `self.assert_eq((kdf + 1).swapaxes(0, 1), (pdf + 1).swapaxes(0, 1))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_33)</summary>

```python
      1 |     def test_swapaxes(self):
      2 |         pdf = pd.DataFrame(
      3 |             [[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=["x", "y", "z"], columns=["a", "b", "c"]
      4 |         )
      5 |         kdf = ks.from_pandas(pdf)
      6 | 
      7 |         self.assert_eq(kdf.swapaxes(0, 1), pdf.swapaxes(0, 1))
      8 |         self.assert_eq(kdf.swapaxes(1, 0), pdf.swapaxes(1, 0))
      9 |         self.assert_eq(kdf.swapaxes("index", "columns"), pdf.swapaxes("index", "columns"))
     10 |         self.assert_eq(kdf.swapaxes("columns", "index"), pdf.swapaxes("columns", "index"))
-->  11 |         self.assert_eq((kdf + 1).swapaxes(0, 1), (pdf + 1).swapaxes(0, 1))
     12 | 
     13 |         self.assertRaises(AssertionError, lambda: kdf.swapaxes(0, 1, copy=False))
     14 |         self.assertRaises(ValueError, lambda: kdf.swapaxes(0, -1))
```

</details>

---

### `bench_103` — `scipy.stats.itemfreq` (scipy)
- **Sample ID**: `scipy_375` | **Line**: 10:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `frequencies = scipy.stats.itemfreq(M)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_375)</summary>

```python
      1 | def check_ks(pop1, pop2, label, alpha, params):
      2 |     clipped_dists = ['exponential_clipped',
      3 |                      'gamma_clipped', 'lognormal_clipped']
      4 |     discrete_dists = ['binomial', 'poisson', 'uniform_int',
      5 |                       'binomial_clipped', 'poisson_clipped']
      6 |     M = get_weighted_connectivity_matrix(pop1, pop2, label)
      7 |     M = M.flatten()
      8 | 
      9 |     if params['distribution'] in discrete_dists:
-->  10 |         frequencies = scipy.stats.itemfreq(M)
     11 |         expected = get_expected_freqs(params, frequencies[:, 0], len(M))
     12 |         chi, p = scipy.stats.chisquare(frequencies[:, 1], expected)
     13 |     elif params['distribution'] in clipped_dists:
     14 |         D, p = scipy.stats.kstest(M, get_clipped_cdf(
     15 |             params), alternative='two-sided')
     16 |     else:
     17 |         if params['distribution'] == 'normal':
     18 |             distrib = scipy.stats.norm
     19 |             args = params['mu'], params['sigma']
     20 |         elif params['distribution'] == 'normal_clipped':
     21 |             distrib = scipy.stats.truncnorm
     22 |             args = (
     23 |                 (params['low'] - params['mu']) /
     24 |                 params['sigma'] if 'low' in params else -np.inf,
     25 |                 (params['high'] - params['mu']) /
     26 |                 params['sigma'] if 'high' in params else -np.inf,
     27 |                 params['mu'],
     28 |                 params['sigma']
     29 |             )
     30 |         elif params['distribution'] == 'exponential':
     31 |             distrib = scipy.stats.expon
     32 |             args = 0, 1. / params['lambda']
     33 |         elif params['distribution'] == 'gamma':
     34 |             distrib = scipy.stats.gamma
     35 |             args = params['order'], 0, params['scale']
     36 |         elif params['distribution'] == 'lognormal':
     37 |             distrib = scipy.stats.lognorm
     38 |             args = params['sigma'], 0, np.exp(params['mu'])
     39 |         elif params['distribution'] == 'uniform':
     40 |             distrib = scipy.stats.uniform
     41 |             args = params['low'], params['high'] - params['low']
     42 |         else:
     43 |             raise ValueError("{} distribution not supported.".format(
     44 |                 params['distribution']))
     45 |         D, p = scipy.stats.kstest(
     46 |             M, distrib.cdf, args=args, alternative='two-sided')
     47 | 
     48 |     return p > alpha
```

</details>

---

### `bench_104` — `scipy.integrate.trapz` (scipy)
- **Sample ID**: `scipy_2013` | **Line**: 26:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `SS = scipyint.trapz(AccTemp, PeriodTemp)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_2013)</summary>

```python
      1 | def MethodNameGotDeleted(Folder_Location, GMid, T1, T2, Tnorm):
      2 |     O = GetResponseSpectrumDataPoints(Folder_Location,GMid)
      3 |     Period = O.Period
      4 |     Acc = O.Sa
      5 |     Vel = O.Sv
      6 |     Disp = O.Sd
      7 | 
      8 |     PeriodTemp = []
      9 |     AccTemp = []
     10 | 
     11 |     if T1 == T2:
     12 |         return 0.0
     13 | 
     14 |     import numpy as np
     15 | 
     16 |     SaNorm = np.interp(Tnorm,Period,Acc)
     17 | 
     18 |     for i in range(len(Period)):
     19 |         if Period[i] <= T2 and Period[i] >= T1:
     20 |             PeriodTemp.append(Period[i])
     21 |             AccTemp.append(Acc[i])
     22 | 
     23 | 
     24 | 
     25 |     import scipy.integrate as scipyint
-->  26 |     SS = scipyint.trapz(AccTemp, PeriodTemp)
     27 |     SS = SS/(SaNorm*(T2-T1))
     28 | 
     29 |     return SS
```

</details>

---

### `bench_105` — `scipy.misc.factorial2` (scipy)
- **Sample ID**: `scipy_1930` | **Line**: 19:23 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `c = 1.0 * factorial2(n-3)**2 / np.sqrt(2*np.pi*np.math.factorial(n))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1930)</summary>

```python
      1 |     def get_initializations(self, num_pol = 5, copy_fun = 'relu'):
      2 |         k = []
      3 |         if copy_fun == 'relu':
      4 |             for n in range(num_pol):
      5 |                 if n == 0:
      6 |                     k.append(1.0/np.sqrt(2*np.pi))
      7 |                     #k.append(0.0)
      8 |                     #k.append(0.3821)
      9 |                 elif n == 1:
     10 |                     k.append(1.0/2)
     11 |                     #k.append(0.0)
     12 |                     #k.append(0.3775)
     13 |                 elif n == 2:
     14 |                     k.append(1.0/np.sqrt(4*np.pi))
     15 |                     #k.append(0.0)
     16 |                     #k.append(0.5535)
     17 |                 elif n > 2 and n % 2 == 0:
     18 |                     #c = 1.0 * np.math.factorial(np.math.factorial(n-3))**2 / np.sqrt(2*np.pi*np.math.factorial(n))
-->  19 |                     c = 1.0 * factorial2(n-3)**2 / np.sqrt(2*np.pi*np.math.factorial(n))
     20 |                     k.append(c)
     21 |                     #k.append(0.0)
     22 |                     #k.append(-0.4244)
     23 |                 elif n >= 2 and n % 2 != 0:
     24 |                     k.append(0.0)
     25 |                     #k.append(0.2126)
     26 |                     #k.append(0.0655)
     27 |         return k
```

</details>

---

### `bench_106` — `numpy.product` (numpy)
- **Sample ID**: `numpy_142` | **Line**: 3:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `return numpy.product(numpy.product(self.image.shape[:-1]))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (numpy_142)</summary>

```python
      1 |     def _feature_shape_image_area(self):
      2 |         if self.multichannel:
-->   3 |             return numpy.product(numpy.product(self.image.shape[:-1]))
      4 |         else:
      5 |             return numpy.product(numpy.product(self.image.shape))
```

</details>

---

### `bench_107` — `pandas.Series.pad` (pandas)
- **Sample ID**: `pandas_130` | **Line**: 23:10 | **Stratum**: `hard_negative` (`replacement_overload_pandas`)
- **Call Site**: `res = df.ffill()`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: modern replacement ffill.*

<details>
<summary>View Enclosing Code (pandas_130)</summary>

```python
      1 |     def test_na_actions_categorical(self):
      2 |         cat = Categorical([1, 2, 3, np.nan], categories=[1, 2, 3])
      3 |         vals = ["a", "b", np.nan, "d"]
      4 |         df = DataFrame({"cats": cat, "vals": vals})
      5 |         cat2 = Categorical([1, 2, 3, 3], categories=[1, 2, 3])
      6 |         vals2 = ["a", "b", "b", "d"]
      7 |         df_exp_fill = DataFrame({"cats": cat2, "vals": vals2})
      8 |         cat3 = Categorical([1, 2, 3], categories=[1, 2, 3])
      9 |         vals3 = ["a", "b", np.nan]
     10 |         df_exp_drop_cats = DataFrame({"cats": cat3, "vals": vals3})
     11 |         cat4 = Categorical([1, 2], categories=[1, 2, 3])
     12 |         vals4 = ["a", "b"]
     13 |         df_exp_drop_all = DataFrame({"cats": cat4, "vals": vals4})
     14 | 
     15 |         # fillna
     16 |         res = df.fillna(value={"cats": 3, "vals": "b"})
     17 |         tm.assert_frame_equal(res, df_exp_fill)
     18 | 
     19 |         msg = "Cannot setitem on a Categorical with a new category"
     20 |         with pytest.raises(TypeError, match=msg):
     21 |             df.fillna(value={"cats": 4, "vals": "c"})
     22 | 
-->  23 |         res = df.ffill()
     24 |         tm.assert_frame_equal(res, df_exp_fill)
     25 | 
     26 |         # dropna
     27 |         res = df.dropna(subset=["cats"])
     28 |         tm.assert_frame_equal(res, df_exp_drop_cats)
     29 | 
     30 |         res = df.dropna()
     31 |         tm.assert_frame_equal(res, df_exp_drop_all)
     32 | 
     33 |         # make sure that fillna takes missing values into account
     34 |         c = Categorical([np.nan, "b", np.nan], categories=["a", "b"])
     35 |         df = DataFrame({"cats": c, "vals": [1, 2, 3]})
     36 | 
     37 |         cat_exp = Categorical(["a", "b", "a"], categories=["a", "b"])
     38 |         df_exp = DataFrame({"cats": cat_exp, "vals": [1, 2, 3]})
     39 | 
     40 |         res = df.fillna("a")
     41 |         tm.assert_frame_equal(res, df_exp)
```

</details>

---

### `bench_108` — `pandas.DataFrame.last` (pandas)
- **Sample ID**: `pandas_84` | **Line**: 11:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `modin_result = modin_df.last("3D")`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_84)</summary>

```python
      1 | def test_last():
      2 |     modin_index = pd.date_range("2010-04-09", periods=400, freq="2D")
      3 |     pandas_index = pandas.date_range("2010-04-09", periods=400, freq="2D")
      4 |     modin_df = pd.DataFrame(
      5 |         {"A": list(range(400)), "B": list(range(400))}, index=modin_index
      6 |     )
      7 |     pandas_df = pandas.DataFrame(
      8 |         {"A": list(range(400)), "B": list(range(400))}, index=pandas_index
      9 |     )
     10 |     with pytest.warns(FutureWarning, match="last is deprecated and will be removed"):
-->  11 |         modin_result = modin_df.last("3D")
     12 |     df_equals(modin_result, pandas_df.last("3D"))
     13 |     df_equals(modin_df.last("20D"), pandas_df.last("20D"))
```

</details>

---

### `bench_109` — `scipy.special.sph_jn` (scipy)
- **Sample ID**: `scipy_987` | **Line**: 29:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `jnxt,jnintpr=scipy.special.sph_jn(n,argu)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_987)</summary>

```python
      1 | def wavefunction():  # computes internal r<1 and externalr>1 wavef.
      2 |      delta=phaseshifts(n,alpha,beta)
      3 |      BL=np.zeros((n),complex)
      4 |      Rin=np.zeros((n,ninpts),float) #internal wavefunction r<1 for n partwaves
      5 |      Rex=np.zeros((n,nexpts),float)
      6 |      for i in range (0,10):     # to find BL for matching
      7 |           jnb,jnpr=scipy.special.sph_jn(n,alpha)# SphBessel functions
      8 |           jnf,jnfr=scipy.special.sph_jn(n,beta)
      9 |           ynb,yprb=r=scipy.special.sph_yn(n,beta)
     10 |           cosd=cos(delta[i])
     11 |           sind=sin(delta[i])
     12 |           zz=complex(cosd,-sind)
     13 |           num=jnb[i]*zz
     14 |           den=cosd*jnf[i]-sind*ynb[i]
     15 |           BL[i]=num/den  # to match internal and external wavefunctions
     16 |      intr=1.0/ninpts # 50points inside, increment          
     17 |      for i in range(0,n):   # internal wavefunc 
     18 |           rin=intr
     19 |           for ri in range(0,ninpts): #plot internal func
     20 |               alpr=alpha*rin
     21 |               jnint,jnintpr=scipy.special.sph_jn(n,alpr)
     22 |               Rin[i,ri]=rin*jnint[i]
     23 |               rin=rin+intr
     24 |      extr=2./nexpts       # from r= 1 to 3   
     25 |      for i in range(0,n):   
     26 |           rex=1.0
     27 |           for rx in range(0,nexpts): #plot internal func
     28 |               argu=beta*rex
-->  29 |               jnxt,jnintpr=scipy.special.sph_jn(n,argu)
     30 |               nxt,jnintpr=scipy.special.sph_yn(n,argu)
     31 |               factr=jnxt[i]*cos(delta[i])-nxt[i]*sin(delta[i])
     32 |               fsin=sin(delta[i])*factr
     33 |               fcos=cos(delta[i])*factr
     34 |               Rex[i,rx]=rex*(fcos*BL.real[i]-fsin*BL.imag[i] )
     35 |               
     36 |               rex=rex+extr       
     37 |      ai=np.arange(0,1,intr)       
     38 |      nwaf=0  #partial wavef to plotCHANGE FOR OTHER WAVES 1 2 3..
     39 |      f3=plt.figure()
     40 |      ax3=f3.add_subplot(111)
     41 |      plt.plot(ai,Rin[nwaf, :])  # only plot s wavefunction
     42 |      ae=np.arange(1,3,extr)   
     43 |      plt.title("  r * partial wavefunction S ")
     44 |      plt.xlabel ("r")
     45 |      plt.plot(ae,Rex[nwaf, :])
```

</details>

---

### `bench_110` — `scipy.misc.comb` (scipy)
- **Sample ID**: `scipy_52` | **Line**: 9:24 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `pi = ( misc.comb(n,i) * misc.comb((N-n), (m-i)) ) / m`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_52)</summary>

```python
      1 | def pval_old(k,n,m,N, adhoc=False, denm=False, logchoose=False):
      2 |     #logger.info("%s %s %s %s" % (k,n,m,N))
      3 |     pv = 0.0
      4 |     for i in range(k,int(min(m,n)+1)):
      5 |         #kdrew: use adhoc choose method instead of scipy.misc.comb
      6 |         if adhoc:
      7 |             pi = ( choose(n,i) * choose((N-n), (m-i)) ) / choose(N,m)
      8 |         elif denm:
-->   9 |             pi = ( misc.comb(n,i) * misc.comb((N-n), (m-i)) ) / m
     10 |         elif logchoose:
     11 |             r1 = logchoose_func(n, i)
     12 |             try:
     13 |                 r2 = logchoose_func(N-n, m-i)
     14 |             except ValueError as ve:
     15 |                 print(str(ve))
     16 |                 return np.nan
     17 |             r3 = logchoose_func(N,m)
     18 | 
     19 |             pi = scipy.exp(r1 + r2 - r3)
     20 |         else:
     21 |             pi = ( misc.comb(n,i) * misc.comb((N-n), (m-i)) ) / misc.comb(N,m)
     22 |         pv += pi
     23 |     return pv
```

</details>

---

### `bench_111` — `numpy.product` (numpy)
- **Sample ID**: `numpy_1412` | **Line**: 29:30 | **Stratum**: `hard_negative` (`stdlib_name_collision_product`)
- **Call Site**: `for m, codes in enumerate(itertools.product(*all_codes)):`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: itertools.product standard library collision.*

<details>
<summary>View Enclosing Code (numpy_1412)</summary>

```python
      1 |     def _get_nan_decs(self):
      2 |         """Set all variables to nan for specializations of fused types for
      3 |         which don't have signatures.
      4 | 
      5 |         """
      6 |         # Set non fused-type variables to nan
      7 |         tab = " "*4
      8 |         fused_types, lines = [], [tab + "else:"]
      9 |         seen = set()
     10 |         for outvar, outtype, code in zip(self.outvars, self.outtypes,
     11 |                                          self.outcodes):
     12 |             if len(code) == 1:
     13 |                 line = f"{outvar}[0] = {NAN_VALUE[code]}"
     14 |                 lines.append(2*tab + line)
     15 |             else:
     16 |                 fused_type = outtype
     17 |                 name, _ = fused_type
     18 |                 if name not in seen:
     19 |                     fused_types.append(fused_type)
     20 |                     seen.add(name)
     21 |         if not fused_types:
     22 |             return lines
     23 | 
     24 |         # Set fused-type variables to nan
     25 |         all_codes = tuple([codes for _unused, codes in fused_types])
     26 | 
     27 |         codelens = [len(x) for x in all_codes]
     28 |         last = numpy.prod(codelens) - 1
-->  29 |         for m, codes in enumerate(itertools.product(*all_codes)):
     30 |             fused_codes, decs = [], []
     31 |             for n, fused_type in enumerate(fused_types):
     32 |                 code = codes[n]
     33 |                 fused_codes.append(underscore(CY_TYPES[code]))
     34 |                 for nn, outvar in enumerate(self.outvars):
     35 |                     if self.outtypes[nn] == fused_type:
     36 |                         line = f"{outvar}[0] = {NAN_VALUE[code]}"
     37 |                         decs.append(line)
     38 |             if m == 0:
     39 |                 adverb = "if"
     40 |             elif m == last:
     41 |                 adverb = "else"
     42 |             else:
     43 |                 adverb = "elif"
     44 |             cond = self._get_conditional(fused_types, codes, adverb)
     45 |             lines.append(2*tab + cond)
     46 |             lines.extend([3*tab + x for x in decs])
     47 |         return lines
```

</details>

---

### `bench_112` — `scipy.misc.comb` (scipy)
- **Sample ID**: `scipy_20` | **Line**: 18:23 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `v -= comb(k, i) * aw[i][-1] * ck[k - i]`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_20)</summary>

```python
      1 |     def __call__(self, u, d=0):
      2 |         """
      3 |         Calculate the point corresponding to given parameter.
      4 |         :param u: Target parameter.
      5 |         :param d: Degree of derivation.
      6 |         :type d: int
      7 |         :return: Value at u with d times derivation.
      8 |         """
      9 | 
     10 |         # TODO: Optimize the calculation of derivatives.
     11 | 
     12 |         aw = np.copy(list(map(lambda der: self.spl(u, der), range(d + 1))))
     13 |         ck = np.empty((d + 1, 3), float)
     14 | 
     15 |         for k in range(d + 1):
     16 |             v = aw[k][:3]
     17 |             for i in range(1, k + 1):
-->  18 |                 v -= comb(k, i) * aw[i][-1] * ck[k - i]
     19 |             ck[k] = v / aw[0][-1]
     20 | 
     21 |         return ck[d]
```

</details>

---

### `bench_113` — `pandas.io.formats.style.Styler.render` (pandas)
- **Sample ID**: `pandas_0` | **Line**: 4:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `es.render()`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_0)</summary>

```python
      1 |     def test_render_empty_dfs(self):
      2 |         empty_df = DataFrame()
      3 |         es = Styler(empty_df)
-->   4 |         es.render()
      5 |         # An index but no columns
      6 |         DataFrame(columns=["a"]).style.render()
      7 |         # A column but no index
      8 |         DataFrame(index=["a"]).style.render()
```

</details>

---

### `bench_114` — `scipy.integrate.cumtrapz` (scipy)
- **Sample ID**: `scipy_1508` | **Line**: 31:36 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `integral = cumtrapz(rfunc, ygr, axis=1,initial=0)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1508)</summary>

```python
      1 |     def _precompute_interpolators(self):
      2 |         """Interpolate each response function and store interpolators.
      3 | 
      4 |         Uses :func:`prince_cr.util.get_interp_object` as interpolator.
      5 |         This might result in too many knots and can be subject to
      6 |         future optimization.
      7 |         """
      8 | 
      9 |         info(2, 'Computing interpolators for response functions')
     10 | 
     11 |         info(5, 'Nonelastic response functions f(y)')
     12 |         self.nonel_intp = {}
     13 |         for mother in self.nonel_idcs:
     14 |             self.nonel_intp[mother] = get_interp_object(
     15 |                 *self.get_channel(mother))
     16 | 
     17 |         info(5, 'Inclusive (boost conserving) response functions g(y)')
     18 |         self.incl_intp = {}
     19 |         for mother, daughter in self.incl_idcs:
     20 |             self.incl_intp[(mother, daughter)] = get_interp_object(
     21 |                 *self.get_channel(mother, daughter))
     22 | 
     23 |         info(5, 'Inclusive (redistributed) response functions h(y)')
     24 |         self.incl_diff_intp = {}
     25 |         for mother, daughter in self.incl_diff_idcs:
     26 |             ygr, rfunc = self.get_channel(mother, daughter)
     27 |             self.incl_diff_intp[(mother, daughter)] = get_2Dinterp_object(
     28 |                 self.xcenters, ygr, rfunc, self.cross_section.xbins)
     29 | 
     30 |             from scipy.integrate import cumtrapz
-->  31 |             integral = cumtrapz(rfunc, ygr, axis=1,initial=0)
     32 |             integral = cumtrapz(integral, self.xcenters, axis=0,initial=0)
     33 | 
     34 |             self.incl_diff_intp_integral[(mother, daughter)] = get_2Dinterp_object(
     35 |                 self.xcenters, ygr, integral, self.cross_section.xbins)
```

</details>

---

### `bench_115` — `numpy.cumproduct` (numpy)
- **Sample ID**: `numpy_1931` | **Line**: 5:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `assert_(np.all(np.cumproduct(A) == expected))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (numpy_1931)</summary>

```python
      1 |     def test_cumproduct(self):
      2 |         A = [[1, 2, 3], [4, 5, 6]]
      3 |         with assert_warns(DeprecationWarning):
      4 |             expected = np.array([1, 2, 6, 24, 120, 720])
-->   5 |             assert_(np.all(np.cumproduct(A) == expected))
```

</details>

---

### `bench_116` — `pandas.io.formats.style.Styler.render` (pandas)
- **Sample ID**: `pandas_12` | **Line**: 5:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `assert '<th class="col_heading level0 col0" colspan="2">l0</th>' in s.render()`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_12)</summary>

```python
      1 |     def test_colspan_w3(self):
      2 |         # GH 36223
      3 |         df = DataFrame(data=[[1, 2]], columns=[["l0", "l0"], ["l1a", "l1b"]])
      4 |         s = Styler(df, uuid="_", cell_ids=False)
-->   5 |         assert '<th class="col_heading level0 col0" colspan="2">l0</th>' in s.render()
```

</details>

---

### `bench_117` — `scipy.integrate.cumtrapz` (scipy)
- **Sample ID**: `scipy_1545` | **Line**: 22:16 | **Stratum**: `hard_negative` (`replacement_overload_integrate`)
- **Call Site**: `primitive = scipy.integrate.cumulative_trapezoid(y.real, 2 * q.real, initial=0)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: modern replacement cumulative_trapezoid.*

<details>
<summary>View Enclosing Code (scipy_1545)</summary>

```python
      1 | def compute_A2_nonlocal(Q, microstructure):
      2 | 
      3 |     margin = 4  # this number must be higher for small grains, because
      4 |     # the FT of gamma is wider, so is the integrand of ReF. It means that using the long range version
      5 |     # is not recommanded for small grains
      6 |     # this value should be adaptive
      7 | 
      8 |     maxq = margin * Q
      9 | 
     10 |     k = 12  # number of samples. This should be adaptive depending on the size/wavelength
     11 |     n = 2**k
     12 |     nQ = n // margin
     13 |     q = np.linspace(0, maxq, n + 1)
     14 | 
     15 |     assert q[nQ] == Q
     16 | 
     17 |     # start with the imaginary part - Eq (70)
     18 |     y = 2 * q * microstructure.ft_autocorrelation_function(2 * q)
     19 | 
     20 |     # integrate from 0 to 2Q (for Q in 0 to qmax)
     21 |     # take the real part to avoid warnings... but this remains to be explored
-->  22 |     primitive = scipy.integrate.cumulative_trapezoid(y.real, 2 * q.real, initial=0)
     23 | 
     24 |     ImF = - 1 / (2 * (2 * np.pi)**1.5) * q * primitive
     25 | 
     26 |     # continue with the real part. Eq (71) is much more difficult to compute than the imaginary part
     27 |     # because of the principal value, but Torquato 2021, suppmat gives a hint with Eq S111.
     28 |     # here: M = maxq
     29 | 
     30 |     with np.errstate(invalid='ignore'):
     31 |         y1 = ImF / ((Q + q) * q)
     32 |         y1[0] = 0  # remove the singularity in 0
     33 | 
     34 |         y2 = (ImF - ImF[nQ]) / (Q**2 - q**2)
     35 | 
     36 |         y2[nQ] = (y2[nQ - 1] + y2[nQ + 1]) / 2   # remove the singularity in Q
     37 | 
     38 |     y = y1 + y2
     39 | 
     40 |     asymptotic_integral = (ImF[nQ] - Q / maxq * ImF[-1]) * np.log(np.abs((maxq + Q) / (maxq - Q)))
     41 | 
     42 |     ReF = - 2 / np.pi * Q * scipy.integrate.romb(y.real, maxq / n) - 1 / np.pi * asymptotic_integral
     43 | 
     44 |     gamma_3_2 = 0.5 * np.sqrt(np.pi)
     45 |     A2 = - (2 * np.pi) / (2**1.5 * gamma_3_2) * (ReF + 1j * ImF[nQ])  # the factor is from eq 67
     46 | 
     47 |     # check ImF(Q)
     48 |     # q = np.linspace(0, 2 * Q, n + 1)
     49 |     # y = q * microstructure.ft_autocorrelation_funscipyction(q)
     50 |     # ImFQ_direct = - 1 / (2 * (2 * np.pi)**1.5) * Q * scipy.integrate.romb(y, 2 * Q / n)
     51 |     # print(ImF[nQ], ImFQ_direct)
     52 | 
     53 |     return A2
```

</details>

---

### `bench_118` — `scipy.linalg.pinv2` (scipy)
- **Sample ID**: `scipy_1263` | **Line**: 7:12 | **Stratum**: `hard_negative` (`submodule_non_deprecated_pinv`)
- **Call Site**: `inv_r = scipy.linalg.pinv(r)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: non-deprecated standard pinv, not pinv2.*

<details>
<summary>View Enclosing Code (scipy_1263)</summary>

```python
      1 | def savitzky_golay_filter(X, order, degree):
      2 |     # Calculate vandermonde matrix.
      3 |     rng = np.arange(-order, order + 1, dtype='float64')
      4 |     s = np.vander(rng)[:, ::-1]
      5 |     S = s[:, :degree + 1]
      6 |     r, = scipy.linalg.qr(S, mode='r')[:degree + 1]
-->   7 |     inv_r = scipy.linalg.pinv(r)
      8 |     g = np.dot(S, np.dot(inv_r, inv_r.T)).astype('float64')
      9 | 
     10 |     filtered = np.zeros_like(X)
     11 | 
     12 |     for i in range(1, X.shape[0]):
     13 |         if i < order:
     14 |             window = np.zeros((order, X.shape[1]))
     15 |             window[-i:] = X[:i]
     16 |         else:
     17 |             window = X[i - order:i]
     18 | 
     19 |         filtered[i] = np.dot(g[:order, 0].T, window) * 2
     20 | 
     21 |     return filtered
```

</details>

---

### `bench_119` — `pandas.DataFrame.select` (pandas)
- **Sample ID**: `pandas_90` | **Line**: 32:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `expected = df.select(crit)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_90)</summary>

```python
      1 |     def test_select(self):
      2 | 
      3 |         # deprecated: gh-12410
      4 |         f = lambda x: x.weekday() == 2
      5 |         index = self.tsframe.index[[f(x) for x in self.tsframe.index]]
      6 |         expected_weekdays = self.tsframe.reindex(index=index)
      7 | 
      8 |         with tm.assert_produces_warning(FutureWarning,
      9 |                                         check_stacklevel=False):
     10 |             result = self.tsframe.select(f, axis=0)
     11 |             assert_frame_equal(result, expected_weekdays)
     12 | 
     13 |             result = self.frame.select(lambda x: x in ('B', 'D'), axis=1)
     14 |             expected = self.frame.reindex(columns=['B', 'D'])
     15 |             assert_frame_equal(result, expected, check_names=False)
     16 | 
     17 |         # replacement
     18 |         f = lambda x: x.weekday == 2
     19 |         result = self.tsframe.loc(axis=0)[f(self.tsframe.index)]
     20 |         assert_frame_equal(result, expected_weekdays)
     21 | 
     22 |         crit = lambda x: x in ['B', 'D']
     23 |         result = self.frame.loc(axis=1)[(self.frame.columns.map(crit))]
     24 |         expected = self.frame.reindex(columns=['B', 'D'])
     25 |         assert_frame_equal(result, expected, check_names=False)
     26 | 
     27 |         # doc example
     28 |         df = DataFrame({'A': [1, 2, 3]}, index=['foo', 'bar', 'baz'])
     29 | 
     30 |         crit = lambda x: x in ['bar', 'baz']
     31 |         with tm.assert_produces_warning(FutureWarning):
-->  32 |             expected = df.select(crit)
     33 |         result = df.loc[df.index.map(crit)]
     34 |         assert_frame_equal(result, expected, check_names=False)
```

</details>

---

### `bench_120` — `pandas.DataFrame.applymap` (pandas)
- **Sample ID**: `pandas_115` | **Line**: 27:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `result = df.applymap(str)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_115)</summary>

```python
      1 |     def test_applymap(self):
      2 |         applied = self.frame.applymap(lambda x: x * 2)
      3 |         assert_frame_equal(applied, self.frame * 2)
      4 |         result = self.frame.applymap(type)
      5 | 
      6 |         # GH #465, function returning tuples
      7 |         result = self.frame.applymap(lambda x: (x, x))
      8 |         tm.assertIsInstance(result['A'][0], tuple)
      9 | 
     10 |         # GH 2909, object conversion to float in constructor?
     11 |         df = DataFrame(data=[1,'a'])
     12 |         result = df.applymap(lambda x: x)
     13 |         self.assertEqual(result.dtypes[0], object)
     14 | 
     15 |         df = DataFrame(data=[1.,'a'])
     16 |         result = df.applymap(lambda x: x)
     17 |         self.assertEqual(result.dtypes[0], object)
     18 | 
     19 |         # GH2786
     20 |         df  = DataFrame(np.random.random((3,4)))
     21 |         df2 = df.copy()
     22 |         cols = ['a','a','a','a']
     23 |         df.columns = cols
     24 | 
     25 |         expected = df2.applymap(str)
     26 |         expected.columns = cols
-->  27 |         result = df.applymap(str)
     28 |         assert_frame_equal(result,expected)
     29 | 
     30 |         # datetime/timedelta
     31 |         df['datetime'] = Timestamp('20130101')
     32 |         df['timedelta'] = Timedelta('1 min')
     33 |         result = df.applymap(str)
     34 |         for f in ['datetime','timedelta']:
     35 |             self.assertEqual(result.loc[0,f],str(df.loc[0,f]))
```

</details>

---

### `bench_121` — `pandas.DataFrame.iteritems` (pandas)
- **Sample ID**: `pandas_74` | **Line**: 9:52 | **Stratum**: `hard_negative` (`wrapper_lookalike_pyspark`)
- **Call Site**: `for (p_name, p_items), (k_name, k_items) in zip(pdf.items(), psdf.items()):`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: third-party PySpark/Koalas DataFrame/Series method.*

<details>
<summary>View Enclosing Code (pandas_74)</summary>

```python
      1 |     def test_items(self):
      2 |         pdf = pd.DataFrame(
      3 |             {"species": ["bear", "bear", "marsupial"], "population": [1864, 22000, 80000]},
      4 |             index=["panda", "polar", "koala"],
      5 |             columns=["species", "population"],
      6 |         )
      7 |         psdf = ps.from_pandas(pdf)
      8 | 
-->   9 |         for (p_name, p_items), (k_name, k_items) in zip(pdf.items(), psdf.items()):
     10 |             self.assert_eq(p_name, k_name)
     11 |             self.assert_eq(p_items, k_items)
```

</details>

---

### `bench_122` — `scipy.misc.comb` (scipy)
- **Sample ID**: `scipy_44` | **Line**: 19:0 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `return scipy.misc.comb(len(target), ham_dist) * 3 ** ham_dist`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_44)</summary>

```python
      1 | def plot_library_comp_by_hamming_distance(ax,
      2 |                                           target,
      3 |                                           max_ham,
      4 |                                           min_reads,
      5 |                                           interesting_reads,
      6 |                                           interesting_seqs):
      7 |     read_counts_given_ham_dist = defaultdict(list)
      8 |     for seq in interesting_seqs:
      9 |         if len(seq) != len(target):
     10 |             continue
     11 |         ham_dist = simple_hamming_distance(target, seq)
     12 |         if ham_dist > max_ham:
     13 |             continue
     14 |         nreads = len(interesting_reads[seq])
     15 |         if nreads >= min_reads:
     16 |             read_counts_given_ham_dist[ham_dist].append(nreads)
     17 | 
     18 |     def nseqs_given_ham(ham_dist):
-->  19 |         return scipy.misc.comb(len(target), ham_dist) * 3 ** ham_dist
     20 | 
     21 |     def make_colormap(seq):
     22 |         """Return a LinearSegmentedColormap
     23 |         seq: a sequence of floats and RGB-tuples. The floats should be increasing
     24 |         and in the interval (0,1).
     25 |         """
     26 |         seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
     27 |         cdict = {'red': [], 'green': [], 'blue': []}
     28 |         for i, item in enumerate(seq):
     29 |             if isinstance(item, float):
     30 |                 r1, g1, b1 = seq[i - 1]
     31 |                 r2, g2, b2 = seq[i + 1]
     32 |                 cdict['red'].append([item, r1, r2])
     33 |                 cdict['green'].append([item, g1, g2])
     34 |                 cdict['blue'].append([item, b1, b2])
     35 |         return mcolors.LinearSegmentedColormap('CustomMap', cdict)
     36 | 
     37 |     bar_w = 0.8
     38 |     viol_w = 0.5
     39 | 
     40 |     max_count = 0
     41 |     ham_dists, nseqs, log_fracs = [], [], []
     42 |     for ham_dist, good_count_list in sorted(read_counts_given_ham_dist.items()):
     43 |         frac_found = float(len(good_count_list)) / nseqs_given_ham(ham_dist)
     44 |         ham_dists.append(ham_dist)
     45 |         nseqs.append(len(good_count_list))
     46 |         log_fracs.append(np.log10(frac_found))
     47 |         max_count = max(good_count_list + [max_count])
     48 |     upper_xlim = 10 ** (int(np.log10(max_count)) + 2)
     49 |     text_x = 10 ** ((np.log10(upper_xlim) + np.log10(max_count)) / 2.0)
     50 | 
     51 |     min_log_frac = -5.0
     52 |     high_color = np.array([0.3, 0.3, 1])
     53 |     low_color = 0.8 * np.array([1, 1, 1])
     54 |     for ham_dist, nseqs, log_frac in zip(ham_dists, nseqs, log_fracs):
     55 |         if log_frac < min_log_frac:
     56 |             log_frac = min_log_frac
     57 |         color = low_color + (log_frac - min_log_frac) / (-min_log_frac) * (high_color - low_color)
     58 |         ax.barh(ham_dist - bar_w / 2.0, nseqs, height=bar_w, color=color, zorder=-1)
     59 |         ax.text(text_x, ham_dist, str(nseqs), ha='center', va='center')
     60 | 
     61 |     viol_color = 0.6 * np.array([1, 1, 1])
     62 |     for ham_dist, good_count_list in sorted(read_counts_given_ham_dist.items()):
     63 |         if len(good_count_list) > 1:
     64 |             viol_d = ax.violinplot(good_count_list, [ham_dist], showextrema=False, vert=False, widths=viol_w)
     65 |             viol_d['bodies'][0].set_color(viol_color)
     66 |             viol_d['bodies'][0].set_alpha(1)
     67 |             viol_d['bodies'][0].set_edgecolor('k')
     68 |         else:
     69 |             ax.plot([good_count_list[0]] * 2, [ham_dist - viol_w / 2.0, ham_dist + viol_w / 2.0], color='k', linewidth=1)
     70 | 
     71 |     ax.set_xscale('log')
     72 |     ax.set_xlim((0.7, upper_xlim))
     73 |     ax.plot([min_reads] * 2, ax.get_ylim(), ':k')
     74 | 
     75 |     ax.set_ylabel('Substitutions', fontsize=18)
     76 |     ax.set_yticks(range(max_ham + 1))
     77 |     ax.set_ylim((max_ham + 1, -1))
     78 |     ax.set_facecolor('white')
     79 |     ax.grid(False)
     80 |     for item in ax.get_xticklabels() + ax.get_yticklabels():
     81 |         item.set_fontsize(16)
     82 | 
     83 |     ax.set_xlabel('Unique Sequences (bars)\nClusters per Sequence (violins)', fontsize=18)
     84 | 
     85 |     cax, kw = mpl.colorbar.make_axes(ax)
     86 |     cmap = make_colormap([low_color, high_color])
     87 |     norm = mpl.colors.Normalize(vmin=min_log_frac, vmax=0)
     88 |     cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
     89 |     cbar.set_label('Fraction of Sequences Recovered')
     90 |     ticks = range(int(min_log_frac), 1)
     91 |     cbar.set_ticks(ticks)
     92 |     cbar.set_ticklabels(['$\leq 10^{%d}$' % min_log_frac] + ['$10^{%d}$' % xx for xx in ticks[1:]])
     93 |     cbar.ax.tick_params(labelsize=14)
```

</details>

---

### `bench_123` — `scipy.integrate.simps` (scipy)
- **Sample ID**: `scipy_1737` | **Line**: 9:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `self.assertAllClose(scipy.integrate.simps(y=y, dx=dx), 1.,`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1737)</summary>

```python
      1 |   def testBatesPDFisNormalized(self, total_count, bounds):
      2 |     low, high = tf.cast(bounds[0], tf.float64), tf.cast(bounds[1], tf.float64)
      3 |     d = bates.Bates(total_count=total_count, low=low, high=high)
      4 |     # This is about as high as JAX can go and still finish in time.
      5 |     nx = 100
      6 |     x = tf.linspace(low, high, nx)
      7 |     y = self.evaluate(d.prob(x))
      8 |     dx = self.evaluate(x[1] - x[0])
-->   9 |     self.assertAllClose(scipy.integrate.simps(y=y, dx=dx), 1.,
     10 |                         atol=5e-05, rtol=5e-05)
```

</details>

---

### `bench_124` — `scipy.misc.face` (scipy)
- **Sample ID**: `scipy_2163` | **Line**: 8:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `face = scipy.misc.face()`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_2163)</summary>

```python
      1 | def test_pyplot(setup):
      2 |     import scipy.misc
      3 |     import matplotlib
      4 |     matplotlib.use('Agg')
      5 |     import matplotlib.pyplot as plt
      6 |     import numpy as np
      7 | 
-->   8 |     face = scipy.misc.face()
      9 |     logger.save_image(face, "face.png")
     10 | 
     11 |     fig = plt.figure(figsize=(4, 2))
     12 |     xs = np.linspace(0, 5, 1000)
     13 |     plt.plot(xs, np.cos(xs))
     14 |     logger.savefig("face_02.png", fig=fig)
     15 |     plt.close()
     16 | 
     17 |     fig = plt.figure(figsize=(4, 2))
     18 |     xs = np.linspace(0, 5, 1000)
     19 |     plt.plot(xs, np.cos(xs))
     20 |     logger.savefig('sine.pdf')
```

</details>

---

### `bench_125` — `pandas.DataFrame.applymap` (pandas)
- **Sample ID**: `pandas_107` | **Line**: 5:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `result = df.applymap(str)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_107)</summary>

```python
      1 | def test_applymap_datetimelike(col, val):
      2 |     # datetime/timedelta
      3 |     df = DataFrame(np.random.random((3, 4)))
      4 |     df[col] = val
-->   5 |     result = df.applymap(str)
      6 |     assert result.loc[0, col] == str(df.loc[0, col])
```

</details>

---

### `bench_126` — `scipy.special.errprint` (scipy)
- **Sample ID**: `scipy_270` | **Line**: 6:12 | **Stratum**: `pipeline_miss` (`cython_native_ufunc`)
- **Call Site**: `c = special.errprint(b)  # returns last state 'a'`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Pipeline missed: compiled Cython native ufunc unparseable by AST.*

<details>
<summary>View Enclosing Code (scipy_270)</summary>

```python
      1 |     def test_errprint(self):
      2 |         with suppress_warnings() as sup:
      3 |             sup.filter(DeprecationWarning, "`errprint` is deprecated!")
      4 |             a = special.errprint()
      5 |             b = 1-a  # a is the state 1-a inverts state
-->   6 |             c = special.errprint(b)  # returns last state 'a'
      7 |             d = special.errprint(a)  # returns to original state
      8 |         assert_equal(a,c)
      9 |         assert_equal(d,b)
```

</details>

---

### `bench_127` — `scipy.special.sph_yn` (scipy)
- **Sample ID**: `scipy_987` | **Line**: 30:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `nxt,jnintpr=scipy.special.sph_yn(n,argu)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_987)</summary>

```python
      1 | def wavefunction():  # computes internal r<1 and externalr>1 wavef.
      2 |      delta=phaseshifts(n,alpha,beta)
      3 |      BL=np.zeros((n),complex)
      4 |      Rin=np.zeros((n,ninpts),float) #internal wavefunction r<1 for n partwaves
      5 |      Rex=np.zeros((n,nexpts),float)
      6 |      for i in range (0,10):     # to find BL for matching
      7 |           jnb,jnpr=scipy.special.sph_jn(n,alpha)# SphBessel functions
      8 |           jnf,jnfr=scipy.special.sph_jn(n,beta)
      9 |           ynb,yprb=r=scipy.special.sph_yn(n,beta)
     10 |           cosd=cos(delta[i])
     11 |           sind=sin(delta[i])
     12 |           zz=complex(cosd,-sind)
     13 |           num=jnb[i]*zz
     14 |           den=cosd*jnf[i]-sind*ynb[i]
     15 |           BL[i]=num/den  # to match internal and external wavefunctions
     16 |      intr=1.0/ninpts # 50points inside, increment          
     17 |      for i in range(0,n):   # internal wavefunc 
     18 |           rin=intr
     19 |           for ri in range(0,ninpts): #plot internal func
     20 |               alpr=alpha*rin
     21 |               jnint,jnintpr=scipy.special.sph_jn(n,alpr)
     22 |               Rin[i,ri]=rin*jnint[i]
     23 |               rin=rin+intr
     24 |      extr=2./nexpts       # from r= 1 to 3   
     25 |      for i in range(0,n):   
     26 |           rex=1.0
     27 |           for rx in range(0,nexpts): #plot internal func
     28 |               argu=beta*rex
     29 |               jnxt,jnintpr=scipy.special.sph_jn(n,argu)
-->  30 |               nxt,jnintpr=scipy.special.sph_yn(n,argu)
     31 |               factr=jnxt[i]*cos(delta[i])-nxt[i]*sin(delta[i])
     32 |               fsin=sin(delta[i])*factr
     33 |               fcos=cos(delta[i])*factr
     34 |               Rex[i,rx]=rex*(fcos*BL.real[i]-fsin*BL.imag[i] )
     35 |               
     36 |               rex=rex+extr       
     37 |      ai=np.arange(0,1,intr)       
     38 |      nwaf=0  #partial wavef to plotCHANGE FOR OTHER WAVES 1 2 3..
     39 |      f3=plt.figure()
     40 |      ax3=f3.add_subplot(111)
     41 |      plt.plot(ai,Rin[nwaf, :])  # only plot s wavefunction
     42 |      ae=np.arange(1,3,extr)   
     43 |      plt.title("  r * partial wavefunction S ")
     44 |      plt.xlabel ("r")
     45 |      plt.plot(ae,Rex[nwaf, :])
```

</details>

---

### `bench_128` — `numpy.alltrue` (numpy)
- **Sample ID**: `numpy_1443` | **Line**: 9:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `self.assert_(numpy.alltrue(a[0] == b[0]))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (numpy_1443)</summary>

```python
      1 |     def test_1d_two_channels_roundtrip(self):
      2 |         """ roundtrip test call with two channels in a frame """
      3 |         a = Fr.frgetvect1d("./test.dat","Adc1")
      4 |         Fr.frputvect('writetest.gwf', [{'name':'Adc1', 'data':a[0],
      5 |         'start':a[1], 'dx':a[3], 'kind':'ADC', 'x_unit':a[4],
      6 |         'y_unit':a[5]},{'name':'reverse', 'data':a[0][::-1], 'start':a[1],
      7 |         'dx':a[3], 'kind':'ADC', 'x_unit':a[4], 'y_unit': a[5]}])
      8 |         b = Fr.frgetvect1d("writetest.gwf", "Adc1")
-->   9 |         self.assert_(numpy.alltrue(a[0] == b[0]))
     10 |         self.assert_(numpy.alltrue(a[1:] == b[1:]))
     11 | 
     12 |         c = Fr.frgetvect1d("writetest.gwf", "reverse")
     13 |         self.assert_(numpy.alltrue(a[0][::-1] == c[0]))
     14 |         self.assert_(numpy.alltrue(a[1:] == c[1:]))
     15 |         os.remove("writetest.gwf")
```

</details>

---

### `bench_129` — `scipy.stats.itemfreq` (scipy)
- **Sample ID**: `scipy_371` | **Line**: 2:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `freq = scipy.stats.itemfreq(a)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_371)</summary>

```python
      1 | def freq_count(a, **kwargs):
-->   2 |     freq = scipy.stats.itemfreq(a)
      3 |     if 'classes' in kwargs:
      4 |         cls = kwargs['classes']
      5 |         x = pd.Series(zeros(cls.shape, dtype=int), index=cls)
      6 |     else:
      7 |         x = pd.Series(zeros(freq.shape[0]), index=freq[:, 0], dtype=int)
      8 | 
      9 |     for l in freq: x[l[0]] = l[1]
     10 |     return x
```

</details>

---

### `bench_130` — `scipy.misc.comb` (scipy)
- **Sample ID**: `scipy_182` | **Line**: 2:17 | **Stratum**: `hard_negative` (`replacement_overload_comb`)
- **Call Site**: `comb = jnp.array(scipy.special.comb(_r50[:, None], _r50[None]))`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: modern replacement scipy.special.comb.*

<details>
<summary>View Enclosing Code (scipy_182)</summary>

```python
      1 | _r50 = np.arange(50)
-->   2 | comb = jnp.array(scipy.special.comb(_r50[:, None], _r50[None]))
```

</details>

---

### `bench_131` — `scipy.integrate.trapz` (scipy)
- **Sample ID**: `scipy_2027` | **Line**: 33:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `y_hat[i] = scipy.integrate.trapz(y_py, t_list)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_2027)</summary>

```python
      1 | def get_y_hat(t_list, density_hat):
      2 |     """
      3 |     Compute y_hat based on t_list and density values.
      4 | 
      5 |     Parameters
      6 |     ----------
      7 |     t_list : np.ndarray
      8 |         Array of t-values.
      9 |     density_hat : np.ndarray
     10 |         Array of density values.
     11 | 
     12 |     Returns
     13 |     -------
     14 |     np.ndarray
     15 |         y_hat for each entry in density_hat.
     16 |     """
     17 |     n_y, n_t = np.shape(density_hat)
     18 | 
     19 |     t_list_hat = (t_list[0:-1] + t_list[1:]) / 2
     20 | 
     21 |     y_hat = np.zeros(n_y)
     22 | 
     23 |     if len(t_list_hat) == n_t:
     24 |         for i in range(0, n_y):
     25 |             y_py = t_list_hat * density_hat[i, :]
     26 | 
     27 |             y_hat[i] = scipy.integrate.trapz(y_py, t_list_hat)
     28 | 
     29 |     else:
     30 |         for i in range(0, n_y):
     31 |             y_py = t_list * density_hat[i, :]
     32 | 
-->  33 |             y_hat[i] = scipy.integrate.trapz(y_py, t_list)
     34 | 
     35 |     return y_hat
```

</details>

---

### `bench_132` — `scipy.stats.rvs_ratio_uniforms` (scipy)
- **Sample ID**: `scipy_459` | **Line**: 11:9 | **Stratum**: `pipeline_miss` (`preamble_filter_omission`)
- **Call Site**: `r3 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=(3, 1),`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Pipeline missed: receiver 'stats' omitted from preamble, callee dropped by manifest short-name filter.*

<details>
<summary>View Enclosing Code (scipy_459)</summary>

```python
      1 |     def test_shape(self):
      2 |         # test shape of return value depending on size parameter
      3 |         f = stats.norm.pdf
      4 |         v_bound = np.sqrt(f(np.sqrt(2))) * np.sqrt(2)
      5 |         umax, vmin, vmax = np.sqrt(f(0)), -v_bound, v_bound
      6 | 
      7 |         r1 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=3,
      8 |                                       random_state=1234)
      9 |         r2 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=(3,),
     10 |                                       random_state=1234)
-->  11 |         r3 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=(3, 1),
     12 |                                       random_state=1234)
     13 |         assert_equal(r1, r2)
     14 |         assert_equal(r2, r3.flatten())
     15 |         assert_equal(r1.shape, (3,))
     16 |         assert_equal(r3.shape, (3, 1))
     17 | 
     18 |         r4 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=(3, 3, 3),
     19 |                                       random_state=12)
     20 |         r5 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=27,
     21 |                                       random_state=12)
     22 |         assert_equal(r4.flatten(), r5)
     23 |         assert_equal(r4.shape, (3, 3, 3))
     24 | 
     25 |         r6 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, random_state=1234)
     26 |         r7 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=1,
     27 |                                       random_state=1234)
     28 |         r8 = stats.rvs_ratio_uniforms(f, umax, vmin, vmax, size=(1, ),
     29 |                                       random_state=1234)
     30 |         assert_equal(r6, r7)
     31 |         assert_equal(r7, r8)
```

</details>

---

### `bench_133` — `scipy.integrate.cumtrapz` (scipy)
- **Sample ID**: `scipy_1456` | **Line**: 27:32 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `spec_cdf = np.hstack((np.zeros(1), cumtrapz(emp_spect, freq)))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1456)</summary>

```python
      1 |     def initialize_from_data_empspect(self, train_x, train_y):
      2 |         """
      3 |         Initialize mixture components based on the empirical spectrum of the data.
      4 | 
      5 |         This will often be better than the standard initialize_from_data method.
      6 |         """
      7 |         import numpy as np
      8 |         from scipy.fftpack import fft
      9 |         from scipy.integrate import cumtrapz
     10 | 
     11 |         if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
     12 |             raise RuntimeError("train_x and train_y should be tensors")
     13 |         if train_x.ndimension() == 1:
     14 |             train_x = train_x.unsqueeze(-1)
     15 | 
     16 |         N = train_x.size(-2)
     17 |         emp_spect = np.abs(fft(train_y.cpu().detach().numpy())) ** 2 / N
     18 |         M = math.floor(N / 2)
     19 | 
     20 |         freq1 = np.arange(M + 1)
     21 |         freq2 = np.arange(-M + 1, 0)
     22 |         freq = np.hstack((freq1, freq2)) / N
     23 |         freq = freq[: M + 1]
     24 |         emp_spect = emp_spect[: M + 1]
     25 | 
     26 |         total_area = np.trapz(emp_spect, freq)
-->  27 |         spec_cdf = np.hstack((np.zeros(1), cumtrapz(emp_spect, freq)))
     28 |         spec_cdf = spec_cdf / total_area
     29 | 
     30 |         a = np.random.rand(1000, self.ard_num_dims)
     31 |         p, q = np.histogram(a, spec_cdf)
     32 |         bins = np.digitize(a, q)
     33 |         slopes = (spec_cdf[bins] - spec_cdf[bins - 1]) / (freq[bins] - freq[bins - 1])
     34 |         intercepts = spec_cdf[bins - 1] - slopes * freq[bins - 1]
     35 |         inv_spec = (a - intercepts) / slopes
     36 | 
     37 |         from sklearn.mixture import GaussianMixture
     38 | 
     39 |         GMM = GaussianMixture(n_components=self.num_mixtures, covariance_type="diag").fit(inv_spec)
     40 |         means = GMM.means_
     41 |         varz = GMM.covariances_
     42 |         weights = GMM.weights_
     43 | 
     44 |         self.mixture_means = means
     45 |         self.mixture_scales = varz
     46 |         self.mixture_weights = weights
```

</details>

---

### `bench_134` — `pandas.DataFrame.iteritems` (pandas)
- **Sample ID**: `pandas_65` | **Line**: 9:8 | **Stratum**: `hard_negative` (`wrapper_lookalike_pyspark`)
- **Call Site**: `pser.pad(inplace=True)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: third-party PySpark/Koalas DataFrame/Series method.*

<details>
<summary>View Enclosing Code (pandas_65)</summary>

```python
      1 |     def test_pad(self):
      2 |         pser = pd.Series([np.nan, 2, 3, 4, np.nan, 6], name="x")
      3 |         psser = ps.from_pandas(pser)
      4 | 
      5 |         if LooseVersion(pd.__version__) >= LooseVersion("1.1"):
      6 |             self.assert_eq(pser.pad(), psser.pad())
      7 | 
      8 |             # Test `inplace=True`
-->   9 |             pser.pad(inplace=True)
     10 |             psser.pad(inplace=True)
     11 |             self.assert_eq(pser, psser)
     12 |         else:
     13 |             expected = ps.Series([np.nan, 2, 3, 4, 4, 6], name="x")
     14 |             self.assert_eq(expected, psser.pad())
     15 | 
     16 |             # Test `inplace=True`
     17 |             psser.pad(inplace=True)
     18 |             self.assert_eq(expected, psser)
```

</details>

---

### `bench_135` — `scipy.linalg.pinv2` (scipy)
- **Sample ID**: `scipy_1287` | **Line**: 12:19 | **Stratum**: `hard_negative` (`submodule_non_deprecated_pinv`)
- **Call Site**: `invM = scipy.linalg.pinv(self.Z)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: non-deprecated standard pinv, not pinv2.*

<details>
<summary>View Enclosing Code (scipy_1287)</summary>

```python
      1 |     def FdCalc(self, Aoptimality=True):
      2 |         """
      3 |         Compute detection power.
      4 | 
      5 |         :param Aoptimality: Kind of optimality to optimize: A- or D-optimality
      6 |         :type Aoptimality: boolean
      7 |         """
      8 |         try:
      9 |             invM = scipy.linalg.inv(self.Z)
     10 |         except scipy.linalg.LinAlgError:
     11 |             try:
-->  12 |                 invM = scipy.linalg.pinv(self.Z)
     13 |             except np.linalg.linalg.LinAlgError:
     14 |                 invM = np.nan
     15 | 
     16 |         invM = np.array(invM)
     17 |         CMC = np.matrix(self.C) * invM * np.matrix(t(self.C))
     18 |         if Aoptimality is True:
     19 |             self.Fd = float(len(self.C) / np.matrix.trace(CMC))
     20 |         else:
     21 |             self.Fd = float(np.linalg.det(CMC) ** (-1 / len(self.C)))
     22 |         self.Fd = self.Fd / self.experiment.FdMax
     23 |         return self
```

</details>

---

### `bench_136` — `scipy.interpolate.interp2d` (scipy)
- **Sample ID**: `scipy_1095` | **Line**: 15:6 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `f = scipy.interpolate.interp2d(rr,`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1095)</summary>

```python
      1 |     def gridData(self):
      2 |         nrad = 9
      3 |         npol = 7
      4 |         rgrid = np.arange(1, nrad + 1)
      5 |         pgrid = np.arange(1, npol + 1)
      6 |         rr, pp = np.meshgrid(rgrid, pgrid)
      7 |         print(pp.shape)
      8 |         rnew = np.arange(0.5, nrad + 0.51, 0.25)
      9 |         pnew = np.arange(0.5, npol + 0.51, 0.25)
     10 |         self.gdata = np.zeros((pnew.size, rnew.size, self.ftime.size))
     11 |         print('starting interpolation')
     12 |         for i in np.arange(self.ftime.size):
     13 |             if i != 0 and np.mod(i + 1, 100) == 0:
     14 |                 print('  frame {} of {}'.format(i + 1, self.ftime.size))
-->  15 |             f = scipy.interpolate.interp2d(rr,
     16 |                                            pp,
     17 |                                            self.fdata[:, :, i].squeeze(),
     18 |                                            kind='linear')
     19 |             self.gdata[:, :, i] = f(rnew, pnew)
```

</details>

---

### `bench_137` — `scipy.integrate.trapz` (scipy)
- **Sample ID**: `scipy_2012` | **Line**: 20:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `SS = scipyint.trapz(AccTemp, PeriodTemp)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_2012)</summary>

```python
      1 | def GetSSThroughSprectrum(Folder_Location, GMid, T1, T2):
      2 |     O = GetResponseSpectrumDataPoints(Folder_Location,GMid)
      3 |     Period = O.Period
      4 |     Acc = O.Sa
      5 |     Vel = O.Sv
      6 |     Disp = O.Sd
      7 | 
      8 |     PeriodTemp = []
      9 |     AccTemp = []
     10 | 
     11 |     if T1 == T2:
     12 |         return 0.0
     13 | 
     14 |     for i in range(len(Period)):
     15 |         if Period[i] <= T2 and Period[i] >= T1:
     16 |             PeriodTemp.append(Period[i])
     17 |             AccTemp.append(Acc[i])
     18 | 
     19 |     import scipy.integrate as scipyint
-->  20 |     SS = scipyint.trapz(AccTemp, PeriodTemp)
     21 |     SS = SS/(AccTemp[0]*(T2-T1))
     22 | 
     23 |     return SS
```

</details>

---

### `bench_138` — `pandas.Series.iteritems` (pandas)
- **Sample ID**: `pandas_53` | **Line**: 32:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `for i, v in x.iteritems()])`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_53)</summary>

```python
      1 |     def test_binary_ops_align(self):
      2 | 
      3 |         # test aligning binary ops
      4 | 
      5 |         # GH 6681
      6 |         index = MultiIndex.from_product([list('abc'),
      7 |                                          ['one', 'two', 'three'],
      8 |                                          [1, 2, 3]],
      9 |                                         names=['first', 'second', 'third'])
     10 | 
     11 |         df = DataFrame(np.arange(27 * 3).reshape(27, 3),
     12 |                        index=index,
     13 |                        columns=['value1', 'value2', 'value3']).sort_index()
     14 | 
     15 |         idx = pd.IndexSlice
     16 |         for op in ['add', 'sub', 'mul', 'div', 'truediv']:
     17 |             opa = getattr(operator, op, None)
     18 |             if opa is None:
     19 |                 continue
     20 | 
     21 |             x = Series([1.0, 10.0, 100.0], [1, 2, 3])
     22 |             result = getattr(df, op)(x, level='third', axis=0)
     23 | 
     24 |             expected = pd.concat([opa(df.loc[idx[:, :, i], :], v)
     25 |                                   for i, v in x.iteritems()]).sort_index()
     26 |             assert_frame_equal(result, expected)
     27 | 
     28 |             x = Series([1.0, 10.0], ['two', 'three'])
     29 |             result = getattr(df, op)(x, level='second', axis=0)
     30 | 
     31 |             expected = (pd.concat([opa(df.loc[idx[:, i], :], v)
-->  32 |                                    for i, v in x.iteritems()])
     33 |                         .reindex_like(df).sort_index())
     34 |             assert_frame_equal(result, expected)
     35 | 
     36 |         # GH9463 (alignment level of dataframe with series)
     37 | 
     38 |         midx = MultiIndex.from_product([['A', 'B'], ['a', 'b']])
     39 |         df = DataFrame(np.ones((2, 4), dtype='int64'), columns=midx)
     40 |         s = pd.Series({'a': 1, 'b': 2})
     41 | 
     42 |         df2 = df.copy()
     43 |         df2.columns.names = ['lvl0', 'lvl1']
     44 |         s2 = s.copy()
     45 |         s2.index.name = 'lvl1'
     46 | 
     47 |         # different cases of integer/string level names:
     48 |         res1 = df.mul(s, axis=1, level=1)
     49 |         res2 = df.mul(s2, axis=1, level=1)
     50 |         res3 = df2.mul(s, axis=1, level=1)
     51 |         res4 = df2.mul(s2, axis=1, level=1)
     52 |         res5 = df2.mul(s, axis=1, level='lvl1')
     53 |         res6 = df2.mul(s2, axis=1, level='lvl1')
     54 | 
     55 |         exp = DataFrame(np.array([[1, 2, 1, 2], [1, 2, 1, 2]], dtype='int64'),
     56 |                         columns=midx)
     57 | 
     58 |         for res in [res1, res2]:
     59 |             assert_frame_equal(res, exp)
     60 | 
     61 |         exp.columns.names = ['lvl0', 'lvl1']
     62 |         for res in [res3, res4, res5, res6]:
     63 |             assert_frame_equal(res, exp)
```

</details>

---

### `bench_139` — `numpy.product` (numpy)
- **Sample ID**: `numpy_68` | **Line**: 25:43 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `(''.join(s) for s in itertools.product(string.ascii_letters, repeat=2)),`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (numpy_68)</summary>

```python
      1 | def collect_values(fp, N, dtp):
      2 |     """Collect N values from stream
      3 | 
      4 |     Must be contained in exact number of lines.
      5 |     This will advance the stream forward by the number of lines found to
      6 |     contain N numeric values, and return an ndarray of type tp containing
      7 |     those.
      8 | 
      9 |     :param file fp: Stream
     10 |     :param int N: Total no. expected values
     11 |     :param dtype tp: dtype for values
     12 |     :returns: ndarray of type dtype with values found in file
     13 |     """ 
     14 |     L = []
     15 |     while len(L) < N:
     16 |         line = fp.readline()
     17 |         if line == "":
     18 |             raise EOFError("File ended prematurely")
     19 |         L.extend(ast.literal_eval(f) for f in line.strip().split())
     20 |     if len(L) != N:
     21 |         raise ValueError("Unexpected number of values.  Expected:"
     22 |             "{N:d}.  Got: {L}".format(N=N, L=len(L)))
     23 |     if numpy.dtype(dtp).isbuiltin == 0:
     24 |         flat_dtp = numpy.dtype(list(zip(
-->  25 |             (''.join(s) for s in itertools.product(string.ascii_letters, repeat=2)),
     26 |             (item for sublist in 
     27 |             [[x[1]]*(numpy.product(x[2]) if len(x)>2 else 1) for x in numpy.dtype(dtp).descr]
     28 |                     for item in sublist))))
     29 |         return numpy.array(tuple(L), dtype=flat_dtp).view(dtp)
     30 |     else:
     31 |         return numpy.array(L, dtype=dtp)
```

</details>

---

### `bench_140` — `numpy.cumproduct` (numpy)
- **Sample ID**: `numpy_3440` | **Line**: 4:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `total_scale = np.cumproduct(upsample_scales)[-1]`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (numpy_3440)</summary>

```python
      1 |     def __init__(self, feat_dims, upsample_scales, compute_dims, 
      2 |                  res_blocks, res_out_dims, pad) :
      3 |         super().__init__()
-->   4 |         total_scale = np.cumproduct(upsample_scales)[-1]
      5 |         self.indent = pad * total_scale
      6 |         self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims)
      7 |         self.resnet_stretch = Stretch2d(total_scale, 1)
      8 |         self.up_layers = nn.ModuleList()
      9 |         for scale in upsample_scales :
     10 |             k_size = (1, scale * 2 + 1)
     11 |             padding = (0, scale)
     12 |             stretch = Stretch2d(scale, 1)
     13 |             conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
     14 |             conv.weight.data.fill_(1. / k_size[1])
     15 |             self.up_layers.append(stretch)
     16 |             self.up_layers.append(conv)
```

</details>

---

### `bench_141` — `scipy.misc.factorial` (scipy)
- **Sample ID**: `scipy_1560` | **Line**: 40:18 | **Stratum**: `label_anomaly` (`external_library_mislabel`)
- **Call Site**: `factorial_N = factorial(N)`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Documented anomaly: client code imports and invokes mpmath.factorial, not scipy.misc.factorial.*

<details>
<summary>View Enclosing Code (scipy_1560)</summary>

```python
      1 | def _mpmath_kraft_burrows_nousek(N, B, CL):
      2 |     '''Upper limit on a poisson count rate
      3 | 
      4 |     The implementation is based on Kraft, Burrows and Nousek in
      5 |     `ApJ 374, 344 (1991) <http://adsabs.harvard.edu/abs/1991ApJ...374..344K>`_.
      6 |     The XMM-Newton upper limit server used the same formalism.
      7 | 
      8 |     Parameters
      9 |     ----------
     10 |     N : int
     11 |         Total observed count number
     12 |     B : float
     13 |         Background count rate (assumed to be known with negligible error
     14 |         from a large background area).
     15 |     CL : float
     16 |        Confidence level (number between 0 and 1)
     17 | 
     18 |     Returns
     19 |     -------
     20 |     S : source count limit
     21 | 
     22 |     Notes
     23 |     -----
     24 |     Requires the `mpmath <http://mpmath.org/>`_ library.  See
     25 |     `~astropy.stats.scipy_poisson_upper_limit` for an implementation
     26 |     that is based on scipy and evaluates faster, but runs only to about
     27 |     N = 100.
     28 |     '''
     29 |     from mpmath import mpf, factorial, findroot, fsum, power, exp, quad
     30 | 
     31 |     N = mpf(N)
     32 |     B = mpf(B)
     33 |     CL = mpf(CL)
     34 | 
     35 |     def eqn8(N, B):
     36 |         sumterms = [power(B, n) / factorial(n) for n in range(int(N) + 1)]
     37 |         return 1. / (exp(-B) * fsum(sumterms))
     38 | 
     39 |     eqn8_res = eqn8(N, B)
-->  40 |     factorial_N = factorial(N)
     41 | 
     42 |     def eqn7(S, N, B):
     43 |         SpB = S + B
     44 |         return eqn8_res * (exp(-SpB) * SpB**N / factorial_N)
     45 | 
     46 |     def eqn9_left(S_min, S_max, N, B):
     47 |         def eqn7NB(S):
     48 |             return eqn7(S, N, B)
     49 |         return quad(eqn7NB, [S_min, S_max])
     50 | 
     51 |     def find_s_min(S_max, N, B):
     52 |         '''
     53 |         Kraft, Burrows and Nousek suggest to integrate from N-B in both
     54 |         directions at once, so that S_min and S_max move similarly (see
     55 |         the article for details). Here, this is implemented differently:
     56 |         Treat S_max as the optimization parameters in func and then
     57 |         calculate the matching s_min that has has eqn7(S_max) =
     58 |         eqn7(S_min) here.
     59 |         '''
     60 |         y_S_max = eqn7(S_max, N, B)
     61 |         if eqn7(0, N, B) >= y_S_max:
     62 |             return 0.
     63 |         else:
     64 |             def eqn7ysmax(x):
     65 |                 return eqn7(x, N, B) - y_S_max
     66 |             return findroot(eqn7ysmax, (N - B) / 2.)
     67 | 
     68 |     def func(s):
     69 |         s_min = find_s_min(s, N, B)
     70 |         out = eqn9_left(s_min, s, N, B)
     71 |         return out - CL
     72 | 
     73 |     S_max = findroot(func, N - B, tol=1e-4)
     74 |     S_min = find_s_min(S_max, N, B)
     75 |     return float(S_min), float(S_max)
```

</details>

---

### `bench_142` — `pandas.Series.iteritems` (pandas)
- **Sample ID**: `pandas_54` | **Line**: 30:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `expected = pd.concat([ opa(df.loc[idx[:,i],:],v) for i, v in x.iteritems() ]).reindex_like(df).sortlevel()`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_54)</summary>

```python
      1 |     def test_binary_ops_align(self):
      2 | 
      3 |         # test aligning binary ops
      4 | 
      5 |         # GH 6681
      6 |         index=MultiIndex.from_product([list('abc'),
      7 |                                        ['one','two','three'],
      8 |                                        [1,2,3]],
      9 |                                       names=['first','second','third'])
     10 | 
     11 |         df = DataFrame(np.arange(27*3).reshape(27,3),
     12 |                        index=index,
     13 |                        columns=['value1','value2','value3']).sortlevel()
     14 | 
     15 |         idx = pd.IndexSlice
     16 |         for op in ['add','sub','mul','div','truediv']:
     17 |             opa = getattr(operator,op,None)
     18 |             if opa is None:
     19 |                 continue
     20 | 
     21 |             x = Series([ 1.0, 10.0, 100.0], [1,2,3])
     22 |             result = getattr(df,op)(x,level='third',axis=0)
     23 | 
     24 |             expected = pd.concat([ opa(df.loc[idx[:,:,i],:],v) for i, v in x.iteritems() ]).sortlevel()
     25 |             assert_frame_equal(result, expected)
     26 | 
     27 |             x = Series([ 1.0, 10.0], ['two','three'])
     28 |             result = getattr(df,op)(x,level='second',axis=0)
     29 | 
-->  30 |             expected = pd.concat([ opa(df.loc[idx[:,i],:],v) for i, v in x.iteritems() ]).reindex_like(df).sortlevel()
     31 |             assert_frame_equal(result, expected)
```

</details>

---

### `bench_143` — `numpy.product` (numpy)
- **Sample ID**: `numpy_17` | **Line**: 17:15 | **Stratum**: `hard_negative` (`stdlib_name_collision_product`)
- **Call Site**: `for key in itertools.product(*[range(k) for k in (shape[:where] + shape[where+1:])]):`
- **Expected Ground Truth**: **False (Benign)** | **Stage 3 Suggestion**: False (Benign)
- **Assistive Preview Rationale**: *Negative: itertools.product standard library collision.*

<details>
<summary>View Enclosing Code (numpy_17)</summary>

```python
      1 | def mv2g(**kwargs):
      2 |   '''Converts all `numpy.ndarrays` given as the keyword arguments 
      3 |   (`**kwargs`) from a vector grid of `shape=(..., Nx*Ny*Nz, ...,)` to a regular 
      4 |   grid of `shape=(..., Nx, Ny, Nz, ...,)`, and, if more than one `**kwargs` is 
      5 |   given, returns it as a dictionary.
      6 |   
      7 |   Hint: The global values for the grid dimensionality, i.e., :mod:`grid.N_`,
      8 |   are used for reshaping.
      9 |   '''
     10 |   import itertools
     11 |   return_val = {}
     12 |   for i,j in kwargs.items():
     13 |     j = numpy.asarray(j,dtype=float)
     14 |     shape = numpy.shape(j)
     15 |     where = numpy.argwhere(shape==numpy.product(N_))[0,0]
     16 |     return_val[i] = numpy.zeros(shape[:where]+tuple(N_)+shape[where+1:])
-->  17 |     for key in itertools.product(*[range(k) for k in (shape[:where] + shape[where+1:])]):
     18 |       obj = [slice(k,k+1) for k in key]
     19 |       for r in range(3):
     20 |         obj.insert(where,slice(None,None))
     21 |       return_val[i][obj] = matrix_vector2grid(j[obj[:where]+obj[where+2:]].reshape((-1,)), 
     22 |                                           **dict(zip(['Nx','Ny','Nz'],N_)))
     23 | 
     24 |   return list(return_val.values())[0] if len(return_val.values()) == 1 else return_val
```

</details>

---

### `bench_144` — `pandas.DataFrame.iteritems` (pandas)
- **Sample ID**: `pandas_52` | **Line**: 5:76 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `for (p_name, p_items), (k_name, k_items) in zip(pser.iteritems(), psser.iteritems()):`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_52)</summary>

```python
      1 |     def test_iteritems(self):
      2 |         pser = pd.Series(["A", "B", "C"])
      3 |         psser = ps.from_pandas(pser)
      4 | 
-->   5 |         for (p_name, p_items), (k_name, k_items) in zip(pser.iteritems(), psser.iteritems()):
      6 |             self.assert_eq(p_name, k_name)
      7 |             self.assert_eq(p_items, k_items)
```

</details>

---

### `bench_145` — `pandas.DataFrame.last` (pandas)
- **Sample ID**: `pandas_84` | **Line**: 13:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `df_equals(modin_df.last("20D"), pandas_df.last("20D"))`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_84)</summary>

```python
      1 | def test_last():
      2 |     modin_index = pd.date_range("2010-04-09", periods=400, freq="2D")
      3 |     pandas_index = pandas.date_range("2010-04-09", periods=400, freq="2D")
      4 |     modin_df = pd.DataFrame(
      5 |         {"A": list(range(400)), "B": list(range(400))}, index=modin_index
      6 |     )
      7 |     pandas_df = pandas.DataFrame(
      8 |         {"A": list(range(400)), "B": list(range(400))}, index=pandas_index
      9 |     )
     10 |     with pytest.warns(FutureWarning, match="last is deprecated and will be removed"):
     11 |         modin_result = modin_df.last("3D")
     12 |     df_equals(modin_result, pandas_df.last("3D"))
-->  13 |     df_equals(modin_df.last("20D"), pandas_df.last("20D"))
```

</details>

---

### `bench_146` — `scipy.integrate.trapz` (scipy)
- **Sample ID**: `scipy_1986` | **Line**: 4:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `area_guess = scipy.integrate.trapz(intensities[i:j], bands[i:j])`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1986)</summary>

```python
      1 | def _guess_area(bands, intensities, loc):
      2 |   idx = np.searchsorted(bands, loc)
      3 |   i, j = max(0, idx-4), idx+5
-->   4 |   area_guess = scipy.integrate.trapz(intensities[i:j], bands[i:j])
      5 |   return max(0, area_guess)
```

</details>

---

### `bench_147` — `scipy.interpolate.interp2d` (scipy)
- **Sample ID**: `scipy_1189` | **Line**: 35:6 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `f = scipy.interpolate.interp2d(`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_1189)</summary>

```python
      1 |     def get_obs_background_field(self, Ub, Vb, grid=None):
      2 |         """
      3 |         Contains ability to handle 1-D varying wind field. However, recommend
      4 |         only supplying scalars as arguments.
      5 |         Ub, Vb = Background U & V winds
      6 |         grid = 1D array of ranges to consider (from VAD)
      7 |         """
      8 |         iflag = False
      9 |         if hasattr(Ub, '__len__') or hasattr(Vb, '__len__'):
     10 |             rngf = (self.obs_xf**2 + self.obs_yf**2)**0.5
     11 |         # Background U
     12 |         if Ub is None:
     13 |             self.obs_Ub = 0.0
     14 |         elif hasattr(Ub, '__len__'):
     15 |             if np.ndim(Ub) == 2:
     16 |                 # Assumes that Ub grid matches analysis grid
     17 |                 f = scipy.interpolate.interp2d(
     18 |                     self.analysis_x, self.analysis_y, Ub, kind='linear')
     19 |                 x1, y1, xii, yii = self._get_sorted_coords()
     20 |                 tmpUb = f(x1, y1)
     21 |                 self.obs_Ub = tmpUb[yii, xii]
     22 |                 iflag = True
     23 |             elif grid is None:
     24 |                 self.obs_Ub = 0.0
     25 |             else:
     26 |                 self.obs_Ub = np.interp(rngf, grid, Ub)
     27 |         else:
     28 |             self.obs_Ub = Ub
     29 |         # Background V
     30 |         if Vb is None:
     31 |             self.obs_Vb = 0.0
     32 |         elif hasattr(Vb, '__len__'):
     33 |             if np.ndim(Vb) == 2:
     34 |                 # Assumes that Vb grid matches analysis grid
-->  35 |                 f = scipy.interpolate.interp2d(
     36 |                     self.analysis_x, self.analysis_y, Vb, kind='linear')
     37 |                 # Check to see if we already have sorted coords
     38 |                 if not iflag:
     39 |                     x1, y1, xii, yii = self._get_sorted_coords()
     40 |                 tmpVb = f(x1, y1)
     41 |                 self.obs_Vb = tmpVb[yii, xii]
     42 |             elif grid is None:
     43 |                 self.obs_Vb = 0.0
     44 |             else:
     45 |                 self.obs_Vb = np.interp(rngf, grid, Vb)
     46 |         else:
     47 |             self.obs_Vb = Vb
     48 |         self.compute_beta_and_m()
     49 |         # Following is for Beta as non-radar convention (0 = due East)
     50 |         self.obs_vrbf = self.obs_Ub * np.cos(self.obs_Beta) + \
     51 |             self.obs_Vb * np.sin(self.obs_Beta)
```

</details>

---

### `bench_148` — `scipy.stats.itemfreq` (scipy)
- **Sample ID**: `scipy_370` | **Line**: 18:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `frequencies = scipy.stats.itemfreq(M)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (scipy_370)</summary>

```python
      1 |     def testRPortDistribution(self):
      2 |         n_rport = 10
      3 |         nr_neurons = 20
      4 |         hf.nest.ResetKernel()
      5 |         neuron_model = 'iaf_psc_exp_multisynapse'
      6 |         neuron_dict = {'tau_syn': [0.1 + i for i in range(n_rport)]}
      7 |         self.pop1 = hf.nest.Create(neuron_model, nr_neurons, neuron_dict)
      8 |         self.pop2 = hf.nest.Create(neuron_model, nr_neurons, neuron_dict)
      9 |         syn_params = {'model': 'static_synapse'}
     10 |         syn_params['receptor_type'] = {
     11 |             'distribution': 'uniform_int', 'low': 1, 'high': n_rport}
     12 |         hf.nest.Connect(self.pop1, self.pop2, self.conn_dict, syn_params)
     13 |         M = hf.get_weighted_connectivity_matrix(
     14 |             self.pop1, self.pop2, 'receptor')
     15 |         M = hf.gather_data(M)
     16 |         if M is not None:
     17 |             M = M.flatten()
-->  18 |             frequencies = scipy.stats.itemfreq(M)
     19 |             self.assertTrue(np.array_equal(frequencies[:, 0], np.arange(
     20 |                 1, n_rport + 1)), 'Missing or invalid rports')
     21 |             chi, p = scipy.stats.chisquare(frequencies[:, 1])
     22 |             self.assertGreater(p, self.pval)
```

</details>

---

### `bench_149` — `numpy.product` (numpy)
- **Sample ID**: `numpy_133` | **Line**: 8:4 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `ntot = numpy.product([len(bin) for bin in bins])`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (numpy_133)</summary>

```python
      1 |  def __init__(self, slowmodel, ebins, parameters, filename, modelname='rebinnedmodel'):
      2 | 		params = [param for param, nbins in parameters]
      3 | 	
      4 | 		bins = [numpy.linspace(param.min, param.max, nbins) for param, nbins in parameters]
      5 | 		left = ebins[:-1]
      6 | 		right = ebins[1:]
      7 | 		width = right - left
-->   8 | 		ntot = numpy.product([len(bin) for bin in bins])
      9 | 		try:
     10 | 			alldata = numpy.load(filename)
     11 | 			data = alldata['y']
     12 | 			assert numpy.allclose(alldata['x'], ebins), 'energy binning differs -- plese delete "%s"' % filename
     13 | 			print('loaded from %s' % filename)
     14 | 		except IOError:
     15 | 			print('creating rebinnedmodel, this might take a while')
     16 | 			print('interpolation setup:')
     17 | 			print('   energies:', ebins[0], ebins[1], '...', ebins[-2], ebins[-1])
     18 | 			for (param, nbins), bin in zip(parameters, bins):
     19 | 				print('   %s: %s - %s with %d points' % (param.fullname, param.min, param.max, nbins))
     20 | 				print('        ', bin)
     21 | 		
     22 | 			data = numpy.zeros((ntot, len(ebins)-1))
     23 | 			for j, element in enumerate(tqdm(list(itertools.product(*bins)), disable=None)):
     24 | 				for i, p in enumerate(params):
     25 | 					if p.val != element[i]:
     26 | 						p.val = element[i]
     27 | 				values = [p.val for p in slowmodel.pars]
     28 | 				model = slowmodel.calc(values, left, right)
     29 | 				#assert numpy.isfinite(model).all(), ('model:', model)
     30 | 				#model = modelcum / width
     31 | 				#assert numpy.isfinite(model).all(), ('model', model)
     32 | 				data[j] = model
     33 | 				# (modelcum * width).cumsum()
     34 | 			print('model created. storing to %s' % filename)
     35 | 			numpy.savez(filename, x=ebins, y=data)
     36 | 		self.init(modelname=modelname, x=ebins, data=data, parameters=parameters)
```

</details>

---

### `bench_150` — `pandas.DataFrame.select` (pandas)
- **Sample ID**: `pandas_82` | **Line**: 4:8 | **Stratum**: `candidate_positive` (`manifest_candidate`)
- **Call Site**: `df2 = df.select(lambda indx: indx >= 1)`
- **Expected Ground Truth**: **True (Deprecated)** | **Stage 3 Suggestion**: True (Deprecated)
- **Assistive Preview Rationale**: *Directly invokes target deprecated API symbol.*

<details>
<summary>View Enclosing Code (pandas_82)</summary>

```python
      1 |     def test_set_index_bug(self):
      2 |         # GH1590
      3 |         df = DataFrame({'val': [0, 1, 2], 'key': ['a', 'b', 'c']})
-->   4 |         df2 = df.select(lambda indx: indx >= 1)
      5 |         rs = df2.set_index('key')
      6 |         xp = DataFrame({'val': [1, 2]},
      7 |                        Index(['b', 'c'], name='key'))
      8 |         assert_frame_equal(rs, xp)
```

</details>

---
