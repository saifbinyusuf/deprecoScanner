# Benchmark Receiver Audit Log (N = 150)

## 1. Formal Receiver Labeling Rule

> **Strict Receiver Rule**: Invoking the candidate target API method on a native-library receiver 
> (Pandas, NumPy, or SciPy) is a deprecated usage regardless of file or test context, whereas 
> invoking it on a third-party wrapper receiver (such as PySpark/Koalas, Modin, or Dask) is a benign lookalike.

## 2. Summary of Adjudicated Differences (v1 vs. v2)

- **Total Items Audited**: 150
- **v1 Frozen Labels**: 107 Deprecated / 43 Benign
- **v2 Adjudicated Labels**: 106 Deprecated / 44 Benign
- **Items Flipped**: 1

- **`bench_108`**: v1 = `True` $\to$ v2 = `False` (Adjudicated from True to False under strict receiver rule: Line 11 receiver 'modin_df' is a Modin third-party wrapper.)

## 3. Detailed 150-Item Audit Log

| ID | Stratum | Target API | Line | Snippet | Receiver | Category | v1 GT | v2 GT | Pred (a/b/c) | Rule Verdict |
| :--- | :--- | :--- | :---: | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `bench_001` | candidate_positive | `scipy.linalg.pinv2` | 11 | `predicted_views.append(predicted_target @ pin` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_002` | hard_negative | `scipy.misc.face` | 1 | `gray = scipy.misc.ascent()` | `<native_call>` | native_library | False | False | 0/0/1 | label_consistent_with_rule |
| `bench_003` | label_anomaly | `scipy.integrate.cumtrapz` | 40 | `ret.data = _sp_cumtrapz(arr, crd_arr.reshape(` | `<native_call>` | native_library | False | False | 0/0/1 | label_consistent_with_rule |
| `bench_004` | hard_negative | `numpy.product` | 26 | `values = list(itertools.product(*list(kwargs.` | `itertools` | standard_library | False | False | 1/0/0 | label_consistent_with_rule |
| `bench_005` | candidate_positive | `scipy.signal.hanning` | 9 | `window = scipy.signal.hanning(len(series))` | `scipy.signal` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_006` | candidate_positive | `scipy.interpolate.interp2d` | 18 | `vi = intp.interp2d(G['lon_v'][0, :], G['lat_v` | `intp` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_007` | hard_negative | `scipy.linalg.pinv2` | 14 | `sig_t_inv = scipy.linalg.pinv(sig_t, rtol=rto` | `<native_call>` | native_library | False | False | 0/0/0 | label_consistent_with_rule |
| `bench_008` | candidate_positive | `scipy.misc.logsumexp` | 11 | `res = scipy.misc.logsumexp(ps + np.log(self.a` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_009` | candidate_positive | `scipy.misc.logsumexp` | 23 | `log_denominator = logsumexp(values, dim=dim, ` | `<direct_call>` | module_import | True | True | 1/0/0 | label_consistent_with_rule |
| `bench_010` | candidate_positive | `pandas.Series.pad` | 9 | `pser.pad(inplace=True)` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_011` | candidate_positive | `scipy.misc.factorial` | 2 | `a = scipy.misc.factorial(N)` | `scipy.misc` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_012` | candidate_positive | `scipy.stats.chisqprob` | 16 | `return 0.5 * scipy.stats.chisqprob(s, 1) + 0.` | `scipy.stats` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_013` | candidate_positive | `numpy.alltrue` | 11 | `homogenous_orientations = numpy.alltrue(orien` | `numpy` | native_numpy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_014` | hard_negative | `numpy.product` | 49 | `np.repeat(np.asarray(com.values_from_object(x` | `np` | native_numpy | True | True | 1/0/1 | label_consistent_with_rule |
| `bench_015` | candidate_positive | `scipy.misc.factorial2` | 15 | `c = 1.0 * factorial2(n-3)**2 / np.sqrt(2*np.p` | `<native_call>` | native_library | True | True | 1/1/0 | label_consistent_with_rule |
| `bench_016` | candidate_positive | `pandas.DataFrame.first` | 6 | `self.assert_eq(pdf.first(DateOffset(days=1)),` | `pdf` | native_pandas | True | True | 1/0/1 | label_consistent_with_rule |
| `bench_017` | hard_negative | `scipy.misc.face` | 7 | `im = scipy.misc.imresize(scipy.misc.face(), a` | `scipy.misc` | native_scipy | False | False | 1/0/1 | label_consistent_with_rule |
| `bench_018` | candidate_positive | `pandas.DataFrame.applymap` | 6 | `assert_eq(ddf.applymap(lambda x: (x, x)), df.` | `df` | native_pandas | True | True | 1/0/1 | label_consistent_with_rule |
| `bench_019` | candidate_positive | `scipy.stats.betai` | 15 | `prob = [scipy.stats.betai(0.5*df,0.5,df/(df+t` | `scipy.stats` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_020` | candidate_positive | `pandas.DataFrame.first` | 10 | `df_equals(modin_df.first("20D"), pandas_df.fi` | `pandas_df` | native_pandas | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_021` | hard_negative | `scipy.integrate.cumtrapz` | 6 | `series = np.pi / 2 / 9.81 * cumulative_trapez` | `<native_call>` | native_library | False | False | 0/0/0 | label_consistent_with_rule |
| `bench_022` | hard_negative | `scipy.misc.face` | 22 | `return scipy.misc.imresize(D, dims)` | `<native_call>` | native_library | False | False | 0/0/1 | label_consistent_with_rule |
| `bench_023` | candidate_positive | `scipy.signal.hanning` | 18 | `ps = ps*scipy.signal.hanning(ps.shape[0])[:,N` | `scipy.signal` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_024` | hard_negative | `pandas.DataFrame.applymap` | 28 | `table = table.map(escape_value)` | `<native_call>` | native_library | False | False | 0/0/0 | label_consistent_with_rule |
| `bench_025` | hard_negative | `scipy.linalg.pinv2` | 18 | `b = scipy.dot(scipy.dot(scipy.linalg.pinv(sci` | `<native_call>` | native_library | False | False | 0/0/1 | label_consistent_with_rule |
| `bench_026` | hard_negative | `numpy.product` | 17 | `for repeats in itertools.product(*tuple(strid` | `itertools` | standard_library | False | False | 1/0/0 | label_consistent_with_rule |
| `bench_027` | candidate_positive | `numpy.alltrue` | 46 | `self.assert_(numpy.alltrue(numpy.isnan(nt3.da` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_028` | candidate_positive | `scipy.integrate.simps` | 19 | `lf = simps(X_s, x=X_t)` | `<direct_call>` | module_import | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_029` | hard_negative | `scipy.linalg.pinv2` | 37 | `blkAinv = scipy.linalg.pinv(blkA)` | `<native_call>` | native_library | False | False | 0/0/0 | label_consistent_with_rule |
| `bench_030` | candidate_positive | `scipy.misc.factorial2` | 25 | `factor_lm = 1/4*np.sqrt(spm.factorial2(2*l+1)` | `spm` | native_inferred | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_031` | candidate_positive | `scipy.interpolate.interp2d` | 4 | `fabs = scipy.interpolate.interp2d(self.lam, s` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_032` | candidate_positive | `scipy.misc.factorial` | 19 | `lKf[tmp] = np.log( scipy.misc.factorial(K[tmp` | `scipy.misc` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_033` | candidate_positive | `scipy.misc.logsumexp` | 11 | `res = scipy.misc.logsumexp(ps + np.log(self.a` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_034` | hard_negative | `pandas.DataFrame.iteritems` | 5 | `self.assert_eq(pdf.first("1D"), psdf.first("1` | `psdf/pser/kdf` | wrapper_lookalike | False | False | 1/0/1 | label_consistent_with_rule |
| `bench_035` | hard_negative | `numpy.product` | 41 | `for idx in itertools.product(*map(range, oper` | `itertools` | standard_library | False | False | 1/0/0 | label_consistent_with_rule |
| `bench_036` | hard_negative | `scipy.misc.comb` | 18 | `scipy.special.comb(sample_count, sample_array` | `scipy.special` | native_scipy | False | False | 1/0/0 | label_consistent_with_rule |
| `bench_037` | hard_negative | `scipy.integrate.cumtrapz` | 9 | `px_desired = integrate.cumulative_trapezoid(v` | `<native_call>` | native_library | False | False | 0/0/0 | label_consistent_with_rule |
| `bench_038` | candidate_positive | `scipy.stats.chisqprob` | 28 | `p = scipy.stats.chisqprob(D, df)` | `scipy.stats` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_039` | hard_negative | `scipy.misc.comb` | 24 | `choose_2p_p = scipy.special.comb(2*p, p, exac` | `scipy.special` | native_scipy | False | False | 1/0/0 | label_consistent_with_rule |
| `bench_040` | candidate_positive | `scipy.misc.factorial` | 14 | `convPml = np.sqrt(2. * factorial(mls[:,:,1]-m` | `<direct_call>` | module_import | True | True | 1/1/0 | label_consistent_with_rule |
| `bench_041` | candidate_positive | `scipy.stats.betai` | 21 | `prob = [scipy.stats.betai(0.5 * dfden, 0.5 * ` | `scipy.stats` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_042` | pipeline_miss | `scipy.special.errprint` | 4 | `c = special.errprint(b)  # returns last state` | `<native_call>` | native_library | True | True | 1/0/1 | label_consistent_with_rule |
| `bench_043` | candidate_positive | `scipy.misc.face` | 7 | `face = misc.face(gray=True)` | `misc` | native_inferred | True | True | 1/1/0 | label_consistent_with_rule |
| `bench_044` | candidate_positive | `pandas.DataFrame.swapaxes` | 4 | `df2 = df.swapaxes(ax, ax)` | `df` | native_pandas | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_045` | pipeline_miss | `scipy.stats.rvs_ratio_uniforms` | 6 | `r1 = stats.rvs_ratio_uniforms(f, umax, vmin, ` | `<native_call>` | native_library | True | True | 1/0/0 | label_consistent_with_rule |
| `bench_046` | candidate_positive | `scipy.misc.face` | 2 | `face = scipy.misc.face()` | `scipy.misc` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_047` | candidate_positive | `scipy.integrate.cumtrapz` | 9 | `pylab.semilogx(total_time[k],scipy.integrate.` | `scipy.integrate` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_048` | hard_negative | `pandas.DataFrame.iteritems` | 12 | `self.assert_eq(pdf1.transpose().sort_index(),` | `psdf/pser/kdf` | wrapper_lookalike | False | False | 0/0/0 | label_consistent_with_rule |
| `bench_049` | candidate_positive | `scipy.special.sph_yn` | 2 | `return scipy.special.sph_yn(n, z)[0][-1]` | `scipy.special` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_050` | candidate_positive | `scipy.misc.factorial` | 14 | `convPml = np.sqrt(2. * factorial(mls[:,:,1]-m` | `<direct_call>` | module_import | True | True | 1/1/0 | label_consistent_with_rule |
| `bench_051` | hard_negative | `pandas.Series.pad` | 7 | `df.ffill(inplace=True)` | `<native_call>` | native_library | False | False | 0/0/0 | label_consistent_with_rule |
| `bench_052` | candidate_positive | `scipy.special.sph_yn` | 5 | `ynb,yprb =  scipy.special.sph_yn(n,beta)` | `scipy.special` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_053` | candidate_positive | `scipy.special.sph_jn` | 3 | `for nn in range(0,n): jn,jpr = scipy.special.` | `scipy.special` | native_scipy | True | True | 1/1/0 | label_consistent_with_rule |
| `bench_054` | candidate_positive | `pandas.Series.pad` | 13 | `self.assert_eq(pdf.pad(), psdf.pad())` | `pdf` | native_pandas | True | True | 1/0/1 | label_consistent_with_rule |
| `bench_055` | hard_negative | `scipy.misc.comb` | 4 | `out[i] = 1 / (n * scipy.special.comb(n-1,i))` | `scipy.special` | native_scipy | False | False | 1/0/0 | label_consistent_with_rule |
| `bench_056` | candidate_positive | `numpy.product` | 23 | `(numpy.product(value.shape) == 1 and numpy.pr` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_057` | candidate_positive | `pandas.DataFrame.swapaxes` | 10 | `self.assert_eq(psdf.swapaxes("columns", "inde` | `pdf` | native_pandas | True | True | 1/0/1 | label_consistent_with_rule |
| `bench_058` | candidate_positive | `pandas.DataFrame.iteritems` | 27 | `for key, val in tensor.iteritems():` | `<native_call>` | native_library | False | False | 1/0/1 | label_consistent_with_rule |
| `bench_059` | candidate_positive | `scipy.stats.itemfreq` | 7 | `return scipy.stats.itemfreq(kos)` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_060` | candidate_positive | `scipy.integrate.simps` | 30 | `v_w_im = simps(f_im,dx=dt)` | `<direct_call>` | module_import | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_061` | candidate_positive | `numpy.alltrue` | 6 | `assert_(np.alltrue(b == np.array([0, 1, 2, 3,` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_062` | candidate_positive | `scipy.integrate.simps` | 26 | `sum2[j] = scipy.integrate.simps(ftemp2[:lastn` | `scipy.integrate` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_063` | candidate_positive | `pandas.io.formats.style.Styler.render` | 7 | `'level0 row0" rowspan="2">l0</th>' in s.rende` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_064` | hard_negative | `numpy.product` | 41 | `for matsize, batchdims in itertools.product([` | `itertools` | standard_library | False | False | 1/0/0 | label_consistent_with_rule |
| `bench_065` | candidate_positive | `scipy.misc.logsumexp` | 8 | `misc.logsumexp(self.model_and_run_length_log_` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_066` | hard_negative | `pandas.DataFrame.iteritems` | 5 | `for (p_name, p_items), (k_name, k_items) in z` | `psdf/pser/kdf` | wrapper_lookalike | False | False | 0/0/0 | label_consistent_with_rule |
| `bench_067` | candidate_positive | `scipy.signal.hanning` | 15 | `qfactor = np.convolve(qfactor, scipy.signal.h` | `scipy.signal` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_068` | candidate_positive | `scipy.stats.betai` | 18 | `return (F,scipy.stats.betai(0.5 * dfwn, 0.5 *` | `scipy.stats` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_069` | hard_negative | `numpy.product` | 14 | `all_possible_digits = itertools.product(*doma` | `itertools` | standard_library | False | False | 1/0/0 | label_consistent_with_rule |
| `bench_070` | candidate_positive | `scipy.special.sph_jn` | 8 | `jnf,jnfr = scipy.special.sph_jn(n,beta)` | `scipy.special` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_071` | candidate_positive | `pandas.DataFrame.first` | 9 | `df_equals(modin_result, pandas_df.first("3D")` | `pandas_df` | native_pandas | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_072` | pipeline_miss | `scipy.special.errprint` | 4 | `flag = sc.errprint(True)` | `sc` | native_inferred | True | True | 1/0/1 | label_consistent_with_rule |
| `bench_073` | candidate_positive | `pandas.Series.iteritems` | 3 | `for el, item in s.iteritems():` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_074` | candidate_positive | `numpy.cumproduct` | 17 | `self.coefs = np.cumproduct((1,) + shape_[:-1]` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_075` | candidate_positive | `pandas.DataFrame.last` | 6 | `self.assert_eq(pdf.last(DateOffset(days=1)), ` | `pdf` | native_pandas | True | True | 1/0/1 | label_consistent_with_rule |
| `bench_076` | hard_negative | `pandas.DataFrame.iteritems` | 17 | `pdf.pad(inplace=True)` | `psdf/pser/kdf` | wrapper_lookalike | False | False | 1/0/1 | label_consistent_with_rule |
| `bench_077` | hard_negative | `scipy.misc.comb` | 6 | `result += scipy.special.comb(N+n, n) * scipy.` | `scipy.special` | native_scipy | False | False | 1/0/0 | label_consistent_with_rule |
| `bench_078` | candidate_positive | `scipy.integrate.cumtrapz` | 38 | `spec_cdf = np.hstack((np.zeros(1), cumtrapz(e` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_079` | hard_negative | `pandas.Series.pad` | 17 | `expected = df.ffill()` | `<native_call>` | native_library | False | False | 0/0/0 | label_consistent_with_rule |
| `bench_080` | candidate_positive | `scipy.misc.factorial2` | 12 | `c = 1.0 * factorial2(n-3)**2 / np.sqrt(2*np.p` | `<native_call>` | native_library | True | True | 1/1/0 | label_consistent_with_rule |
| `bench_081` | pipeline_miss | `scipy.stats.rvs_ratio_uniforms` | 7 | `rvs = stats.rvs_ratio_uniforms(f, umax, vmin,` | `<native_call>` | native_library | True | True | 1/0/0 | label_consistent_with_rule |
| `bench_082` | hard_negative | `scipy.integrate.simps` | 18 | `intg2 = integrate.simpson(tmp_y, tmp_x, tmp_x` | `<native_call>` | native_library | False | False | 0/0/0 | label_consistent_with_rule |
| `bench_083` | pipeline_miss | `scipy.special.errprint` | 51 | `sv = special.errprint(sv)` | `special` | native_inferred | True | True | 1/0/1 | label_consistent_with_rule |
| `bench_084` | hard_negative | `scipy.misc.comb` | 29 | `real_pairs += scipy.special.comb(count, 2)` | `scipy.special` | native_scipy | False | False | 1/0/0 | label_consistent_with_rule |
| `bench_085` | candidate_positive | `scipy.stats.chisqprob` | 16 | `p = scipy.stats.chisqprob(chi, df)` | `scipy.stats` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_086` | candidate_positive | `scipy.special.sph_yn` | 30 | `nxt,jnintpr = scipy.special.sph_yn(n,argu)` | `scipy.special` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_087` | hard_negative | `scipy.misc.comb` | 16 | `pref = 1. / scipy.special.comb(L, L // 2 + ch` | `scipy.special` | native_scipy | False | False | 1/0/0 | label_consistent_with_rule |
| `bench_088` | candidate_positive | `pandas.DataFrame.select` | 8 | `df1 = df.select(lambda u: u[0] in ['f2', 'f3'` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_089` | candidate_positive | `scipy.stats.chisqprob` | 30 | `prob = scipy.stats.chisqprob(chi2, dof)` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_090` | pipeline_miss | `scipy.stats.rvs_ratio_uniforms` | 12 | `rvs = stats.rvs_ratio_uniforms(lambda x: np.e` | `<native_call>` | native_library | True | True | 1/0/0 | label_consistent_with_rule |
| `bench_091` | hard_negative | `pandas.DataFrame.iteritems` | 19 | `psdf.ffill(inplace=True)` | `psdf/pser/kdf` | wrapper_lookalike | False | False | 0/0/1 | label_consistent_with_rule |
| `bench_092` | candidate_positive | `scipy.misc.face` | 5 | `image_tensor = tf.constant(scipy.misc.face())` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_093` | candidate_positive | `pandas.DataFrame.iteritems` | 9 | `for (p_name, p_items), (k_name, k_items) in z` | `pdf` | native_pandas | True | True | 1/0/1 | label_consistent_with_rule |
| `bench_094` | candidate_positive | `scipy.signal.hanning` | 15 | `mX, _ = TimeFrequencyDecomposition.STFT(xn, h` | `<native_call>` | native_library | True | True | 1/1/0 | label_consistent_with_rule |
| `bench_095` | candidate_positive | `scipy.misc.comb` | 24 | `c[j] += factor * comb(j, k-a)` | `<native_call>` | native_library | True | True | 1/1/0 | label_consistent_with_rule |
| `bench_096` | candidate_positive | `scipy.special.sph_jn` | 3 | `for nn in range(0,n):  jn,jpr = scipy.special` | `scipy.special` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_097` | candidate_positive | `scipy.stats.betai` | 13 | `return (t, scipy.stats.betai(0.5*df,0.5,df/(d` | `scipy.stats` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_098` | candidate_positive | `pandas.Series.pad` | 14 | `self.assert_eq(pdf.pad(), kdf.pad())` | `pdf` | native_pandas | True | True | 1/0/1 | label_consistent_with_rule |
| `bench_099` | hard_negative | `pandas.DataFrame.iteritems` | 5 | `self.assert_eq(pdf.last("1D"), psdf.last("1D"` | `psdf/pser/kdf` | wrapper_lookalike | False | False | 1/0/1 | label_consistent_with_rule |
| `bench_100` | candidate_positive | `numpy.cumproduct` | 2 | `self.assert_deprecated(lambda: np.cumproduct(` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_101` | hard_negative | `scipy.misc.comb` | 4 | `return _spspecial.comb(n, r)` | `_spspecial` | native_inferred | False | False | 1/0/0 | label_consistent_with_rule |
| `bench_102` | candidate_positive | `pandas.DataFrame.swapaxes` | 11 | `self.assert_eq((kdf + 1).swapaxes(0, 1), (pdf` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_103` | candidate_positive | `scipy.stats.itemfreq` | 10 | `frequencies = scipy.stats.itemfreq(M)` | `scipy.stats` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_104` | candidate_positive | `scipy.integrate.trapz` | 26 | `SS = scipyint.trapz(AccTemp, PeriodTemp)` | `scipyint` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_105` | candidate_positive | `scipy.misc.factorial2` | 19 | `c = 1.0 * factorial2(n-3)**2 / np.sqrt(2*np.p` | `<native_call>` | native_library | True | True | 1/1/0 | label_consistent_with_rule |
| `bench_106` | candidate_positive | `numpy.product` | 3 | `return numpy.product(numpy.product(self.image` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_107` | hard_negative | `pandas.Series.pad` | 23 | `res = df.ffill()` | `<native_call>` | native_library | False | False | 0/0/0 | label_consistent_with_rule |
| `bench_108` | candidate_positive | `pandas.DataFrame.last` | 11 | `modin_result = modin_df.last("3D")` | `modin_df` | wrapper_modin | True | False | 1/0/1 | benign_lookalike_wrapper_receiver |
| `bench_109` | candidate_positive | `scipy.special.sph_jn` | 29 | `jnxt,jnintpr=scipy.special.sph_jn(n,argu)` | `scipy.special` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_110` | candidate_positive | `scipy.misc.comb` | 9 | `pi = ( misc.comb(n,i) * misc.comb((N-n), (m-i` | `misc` | native_inferred | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_111` | hard_negative | `numpy.product` | 29 | `for m, codes in enumerate(itertools.product(*` | `itertools` | standard_library | False | False | 1/0/0 | label_consistent_with_rule |
| `bench_112` | candidate_positive | `scipy.misc.comb` | 18 | `v -= comb(k, i) * aw[i][-1] * ck[k - i]` | `<native_call>` | native_library | True | True | 1/1/0 | label_consistent_with_rule |
| `bench_113` | candidate_positive | `pandas.io.formats.style.Styler.render` | 4 | `es.render()` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_114` | candidate_positive | `scipy.integrate.cumtrapz` | 31 | `integral = cumtrapz(rfunc, ygr, axis=1,initia` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_115` | candidate_positive | `numpy.cumproduct` | 5 | `assert_(np.all(np.cumproduct(A) == expected))` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_116` | candidate_positive | `pandas.io.formats.style.Styler.render` | 5 | `assert '<th class="col_heading level0 col0" c` | `<native_call>` | native_library | True | True | 1/1/0 | label_consistent_with_rule |
| `bench_117` | hard_negative | `scipy.integrate.cumtrapz` | 22 | `primitive = scipy.integrate.cumulative_trapez` | `<native_call>` | native_library | False | False | 0/0/0 | label_consistent_with_rule |
| `bench_118` | hard_negative | `scipy.linalg.pinv2` | 7 | `inv_r = scipy.linalg.pinv(r)` | `<native_call>` | native_library | False | False | 0/0/0 | label_consistent_with_rule |
| `bench_119` | candidate_positive | `pandas.DataFrame.select` | 32 | `expected = df.select(crit)` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_120` | candidate_positive | `pandas.DataFrame.applymap` | 27 | `result = df.applymap(str)` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_121` | hard_negative | `pandas.DataFrame.iteritems` | 9 | `for (p_name, p_items), (k_name, k_items) in z` | `psdf/pser/kdf` | wrapper_lookalike | False | False | 0/0/0 | label_consistent_with_rule |
| `bench_122` | candidate_positive | `scipy.misc.comb` | 19 | `return scipy.misc.comb(len(target), ham_dist)` | `scipy.misc` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_123` | candidate_positive | `scipy.integrate.simps` | 9 | `self.assertAllClose(scipy.integrate.simps(y=y` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_124` | candidate_positive | `scipy.misc.face` | 8 | `face = scipy.misc.face()` | `scipy.misc` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_125` | candidate_positive | `pandas.DataFrame.applymap` | 5 | `result = df.applymap(str)` | `df` | native_pandas | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_126` | pipeline_miss | `scipy.special.errprint` | 6 | `c = special.errprint(b)  # returns last state` | `<native_call>` | native_library | True | True | 1/0/1 | label_consistent_with_rule |
| `bench_127` | candidate_positive | `scipy.special.sph_yn` | 30 | `nxt,jnintpr=scipy.special.sph_yn(n,argu)` | `scipy.special` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_128` | candidate_positive | `numpy.alltrue` | 9 | `self.assert_(numpy.alltrue(a[0] == b[0]))` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_129` | candidate_positive | `scipy.stats.itemfreq` | 2 | `freq = scipy.stats.itemfreq(a)` | `scipy.stats` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_130` | hard_negative | `scipy.misc.comb` | 2 | `comb = jnp.array(scipy.special.comb(_r50[:, N` | `scipy.special` | native_scipy | False | False | 1/0/0 | label_consistent_with_rule |
| `bench_131` | candidate_positive | `scipy.integrate.trapz` | 33 | `y_hat[i] = scipy.integrate.trapz(y_py, t_list` | `scipy.integrate` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_132` | pipeline_miss | `scipy.stats.rvs_ratio_uniforms` | 11 | `r3 = stats.rvs_ratio_uniforms(f, umax, vmin, ` | `<native_call>` | native_library | True | True | 1/0/0 | label_consistent_with_rule |
| `bench_133` | candidate_positive | `scipy.integrate.cumtrapz` | 27 | `spec_cdf = np.hstack((np.zeros(1), cumtrapz(e` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_134` | hard_negative | `pandas.DataFrame.iteritems` | 9 | `pser.pad(inplace=True)` | `psdf/pser/kdf` | wrapper_lookalike | False | False | 1/0/1 | label_consistent_with_rule |
| `bench_135` | hard_negative | `scipy.linalg.pinv2` | 12 | `invM = scipy.linalg.pinv(self.Z)` | `<native_call>` | native_library | False | False | 0/0/0 | label_consistent_with_rule |
| `bench_136` | candidate_positive | `scipy.interpolate.interp2d` | 15 | `f = scipy.interpolate.interp2d(rr,` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_137` | candidate_positive | `scipy.integrate.trapz` | 20 | `SS = scipyint.trapz(AccTemp, PeriodTemp)` | `scipyint` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_138` | candidate_positive | `pandas.Series.iteritems` | 32 | `for i, v in x.iteritems()])` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_139` | candidate_positive | `numpy.product` | 25 | `(''.join(s) for s in itertools.product(string` | `itertools` | standard_library | False | False | 1/0/0 | label_consistent_with_rule |
| `bench_140` | candidate_positive | `numpy.cumproduct` | 4 | `total_scale = np.cumproduct(upsample_scales)[` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_141` | label_anomaly | `scipy.misc.factorial` | 40 | `factorial_N = factorial(N)` | `<direct_call>` | module_import | False | False | 1/0/0 | label_consistent_with_rule |
| `bench_142` | candidate_positive | `pandas.Series.iteritems` | 30 | `expected = pd.concat([ opa(df.loc[idx[:,i],:]` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_143` | hard_negative | `numpy.product` | 17 | `for key in itertools.product(*[range(k) for k` | `itertools` | standard_library | False | False | 1/0/0 | label_consistent_with_rule |
| `bench_144` | candidate_positive | `pandas.DataFrame.iteritems` | 5 | `for (p_name, p_items), (k_name, k_items) in z` | `pser` | native_pandas | True | True | 1/0/1 | label_consistent_with_rule |
| `bench_145` | candidate_positive | `pandas.DataFrame.last` | 13 | `df_equals(modin_df.last("20D"), pandas_df.las` | `pandas_df` | native_pandas | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_146` | candidate_positive | `scipy.integrate.trapz` | 4 | `area_guess = scipy.integrate.trapz(intensitie` | `scipy.integrate` | native_scipy | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_147` | candidate_positive | `scipy.interpolate.interp2d` | 35 | `f = scipy.interpolate.interp2d(` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_148` | candidate_positive | `scipy.stats.itemfreq` | 18 | `frequencies = scipy.stats.itemfreq(M)` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_149` | candidate_positive | `numpy.product` | 8 | `ntot = numpy.product([len(bin) for bin in bin` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |
| `bench_150` | candidate_positive | `pandas.DataFrame.select` | 4 | `df2 = df.select(lambda indx: indx >= 1)` | `<native_call>` | native_library | True | True | 1/1/1 | label_consistent_with_rule |