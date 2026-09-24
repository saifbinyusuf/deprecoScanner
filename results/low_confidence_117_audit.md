# Ground-Truth Audit of the 117 Low-Confidence Candidates

## 1. Summary Statistics

- **Total Scored Candidates**: 117 (all from the outdated cohort)
- **Stage 3 Verified**: 80 Confirmed Deprecated / 37 Rejected Benign (**68.38% confirmation rate**)
- **Human Call-Site Ground Truth**: 90 True Deprecations / 27 Benign Lookalikes
- **Performance**: TP = 80, FP = 0, TN = 27, FN = 10
- **Precision**: 100.00%
- **Recall**: 88.89%
- **F1 Score**: 0.9412
- **Overall Accuracy**: 91.45%

## 2. Full 117-Item Audit Table

| ID | Target API | Snippet | Category | GT | Stage 3 Pred | Conf | Correct | Rationale |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `numpy_0_lc_0_11_40` | `numpy.product` | `assert_equal(np.product(x, axis=0),` | native_library | True | True | 1.00 | ✓ | The call site explicitly invokes numpy.product (al |
| `numpy_0_lc_1_12_35` | `numpy.product` | `assert_equal(np.product(x, 0), prod` | native_library | True | True | 0.95 | ✓ | The call site explicitly invokes numpy.product via |
| `numpy_0_lc_3_20_39` | `numpy.product` | `assert_equal(np.product(x, 1), prod` | native_library | True | True | 1.00 | ✓ | The code actively invokes the deprecated numpy.pro |
| `numpy_0_lc_2_13_52` | `numpy.product` | `assert_equal(np.product(filled(xm, ` | native_library | True | True | 1.00 | ✓ | The call site explicitly invokes np.product from t |
| `numpy_3_lc_1_12_33` | `numpy.product` | `assert_(eq(np.product(x, 0), produc` | native_library | True | True | 1.00 | ✓ | The call site explicitly invokes np.product(x, 0)  |
| `numpy_3_lc_2_14_23` | `numpy.product` | `product(xm, axis=0)))` | native_library | True | True | 0.95 | ✓ | The enclosing test context explicitly compares num |
| `numpy_3_lc_3_20_37` | `numpy.product` | `assert_(eq(np.product(x, 1), produc` | native_library | True | True | 1.00 | ✓ | The call site explicitly invokes np.product(x, 1)  |
| `numpy_2_lc_0_32_40` | `numpy.product` | `for m, codes in enumerate(itertools` | standard_library | False | False | 1.00 | ✓ | The call site uses itertools.product, which is par |
| `numpy_3_lc_0_11_38` | `numpy.product` | `assert_(eq(np.product(x, axis=0), p` | native_library | True | True | 1.00 | ✓ | The call site explicitly invokes np.product using  |
| `numpy_4_lc_0_42_40` | `pandas.DataFrame.swapaxes` | `self.xdens = grid.GridData(new_list` | ndarray_receiver | False | False | 0.99 | ✓ | The receiver 'new_list' is created via numpy opera |
| `numpy_35_lc_0_6_16` | `numpy.product` | `a *= _N.product(a.shape)` | native_library | True | True | 0.95 | ✓ | The receiver '_N' is used alongside numpy fft call |
| `numpy_36_lc_0_6_16` | `numpy.product` | `a *= _N.product(a.shape)` | native_library | True | True | 0.95 | ✓ | The receiver '_N' is used alongside numpy fft call |
| `numpy_42_lc_0_11_46` | `numpy.product` | `self.assertTrue(eq(np.product(x, ax` | native_library | True | True | 0.99 | ✓ | The call site explicitly invokes numpy.product (al |
| `numpy_42_lc_1_12_41` | `numpy.product` | `self.assertTrue(eq(np.product(x, 0)` | native_library | True | True | 1.00 | ✓ | The call site explicitly invokes np.product(x, 0)  |
| `numpy_42_lc_2_14_23` | `numpy.product` | `product(xm, axis=0)))` | native_library | True | True | 0.95 | ✓ | The enclosing test context explicitly compares num |
| `numpy_61_lc_0_11_50` | `numpy.product` | `self.assertTrue (eq(numpy.product(x` | native_library | True | True | 0.95 | ✓ | The code explicitly calls numpy.product(x, axis=0) |
| `numpy_42_lc_3_20_45` | `numpy.product` | `self.assertTrue(eq(np.product(x, 1)` | native_library | True | True | 0.95 | ✓ | The call site explicitly invokes numpy.product via |
| `numpy_61_lc_3_20_49` | `numpy.product` | `self.assertTrue (eq(numpy.product(x` | native_library | True | True | 0.95 | ✓ | The call explicitly references numpy.product(x, 1) |
| `numpy_68_lc_1_25_43` | `numpy.product` | `(''.join(s) for s in itertools.prod` | standard_library | False | False | 1.00 | ✓ | The call site uses itertools.product rather than n |
| `numpy_74_lc_0_11_40` | `numpy.product` | `assert_equal(np.product(x, axis=0),` | native_library | True | True | 1.00 | ✓ | The call site explicitly invokes numpy.product (al |
| `numpy_74_lc_1_12_35` | `numpy.product` | `assert_equal(np.product(x, 0), prod` | native_library | True | True | 0.95 | ✓ | The call site explicitly invokes numpy.product via |
| `numpy_61_lc_1_12_45` | `numpy.product` | `self.assertTrue (eq(numpy.product(x` | native_library | True | True | 0.99 | ✓ | The call site explicitly invokes numpy.product(x,  |
| `numpy_61_lc_2_14_24` | `numpy.product` | `product(xm, axis=0)))` | native_library | True | True | 0.95 | ✓ | The enclosing test context explicitly compares num |
| `numpy_77_lc_0_11_47` | `numpy.product` | `self.assertTrue (eq(np.product(x, a` | native_library | True | True | 0.95 | ✓ | The code explicitly invokes np.product on a NumPy  |
| `numpy_77_lc_1_12_42` | `numpy.product` | `self.assertTrue (eq(np.product(x, 0` | native_library | True | True | 0.99 | ✓ | The call site explicitly invokes np.product(x, 0)  |
| `numpy_77_lc_2_14_24` | `numpy.product` | `product(xm, axis=0)))` | native_library | True | True | 0.95 | ✓ | The enclosing test context explicitly compares num |
| `numpy_77_lc_3_20_46` | `numpy.product` | `self.assertTrue (eq(np.product(x, 1` | native_library | True | True | 1.00 | ✓ | The code actively invokes the deprecated numpy.pro |
| `numpy_74_lc_2_13_52` | `numpy.product` | `assert_equal(np.product(filled(xm, ` | native_library | True | True | 1.00 | ✓ | The call site explicitly invokes np.product from t |
| `numpy_74_lc_3_20_39` | `numpy.product` | `assert_equal(np.product(x, 1), prod` | native_library | True | True | 1.00 | ✓ | The code actively invokes the deprecated numpy.pro |
| `numpy_133_lc_0_23_59` | `numpy.product` | `for j, element in enumerate(tqdm(li` | standard_library | False | False | 1.00 | ✓ | The call site invokes itertools.product instead of |
| `numpy_150_lc_0_27_31` | `pandas.DataFrame.iteritems` | `for key, val in tensor.iteritems():` | custom_class | False | False | 0.99 | ✓ | The method iteritems is being called on a numpy or |
| `numpy_151_lc_0_11_42` | `numpy.product` | `assert_equal(numpy.product(x,axis=0` | native_library | True | True | 0.99 | ✓ | The call site explicitly invokes numpy.product(x,  |
| `numpy_151_lc_1_12_37` | `numpy.product` | `assert_equal(numpy.product(x,0), pr` | native_library | True | True | 1.00 | ✓ | The code directly calls the deprecated numpy.produ |
| `numpy_151_lc_2_13_53` | `numpy.product` | `assert_equal(numpy.product(filled(x` | native_library | True | True | 1.00 | ✓ | The call site explicitly invokes numpy.product(fil |
| `numpy_151_lc_3_20_41` | `numpy.product` | `assert_equal(numpy.product(x,1), pr` | native_library | True | True | 1.00 | ✓ | The call explicitly invokes numpy.product(x, 1) in |
| `numpy_186_lc_0_5_49` | `numpy.product` | `for size, noncontiguous, out_int32,` | native_library | True | False | 0.99 | ✗ | The call to product(...) in the loop is Python's b |
| `numpy_216_lc_0_12_43` | `numpy.product` | `for selected_conditionals in iterto` | standard_library | False | False | 1.00 | ✓ | The call site uses itertools.product from the stan |
| `numpy_1582_lc_0_17_23` | `numpy.alltrue` | `assert npy.alltrue([isinstance(w, c` | native_library | True | True | 0.95 | ✓ | The variable npy is used as an alias for numpy, in |
| `numpy_1582_lc_1_19_23` | `numpy.alltrue` | `assert npy.alltrue([f in flist for ` | native_library | True | True | 0.95 | ✓ | The variable npy is used as an alias for numpy, ev |
| `numpy_1583_lc_0_16_15` | `numpy.alltrue` | `if npy.alltrue(self._initiator_cach` | native_library | True | True | 0.95 | ✓ | The receiver 'npy' is conventionally used as an al |
| `numpy_1576_lc_0_26_28` | `numpy.product` | `values = list(itertools.product(*li` | standard_library | False | False | 1.00 | ✓ | The call site uses standard library itertools.prod |
| `numpy_1673_lc_1_14_26` | `numpy.alltrue` | `return A*(_np.alltrue(A < self.valu` | native_library | True | True | 0.95 | ✓ | The receiver '_np' is a standard alias for NumPy,  |
| `numpy_1673_lc_0_17_22` | `numpy.alltrue` | `A *= (_np.alltrue(A < self.value, a` | native_library | True | True | 0.95 | ✓ | The receiver '_np' is a standard alias for NumPy,  |
| `numpy_1674_lc_1_14_26` | `numpy.alltrue` | `return A*(_np.alltrue(A > self.valu` | native_library | True | True | 0.95 | ✓ | The receiver '_np' is a standard alias for NumPy,  |
| `numpy_1674_lc_0_17_22` | `numpy.alltrue` | `A *= (_np.alltrue(A > self.value, a` | native_library | True | True | 0.95 | ✓ | The receiver _np is conventionally used as the ali |
| `numpy_1666_lc_0_54_24` | `numpy.alltrue` | `identical = num.alltrue(A==B)` | native_library | True | True | 0.95 | ✓ | The enclosing code checks isinstance(A, num.ndarra |
| `numpy_3466_lc_2_19_11` | `numpy.product` | `while (product(counts) * next.N < m` | native_library | True | True | 0.95 | ✓ | The code uses product() on a list of counts to cal |
| `pandas_0_lc_0_6_35` | `pandas.io.formats.style.Styler.render` | `DataFrame(columns=["a"]).style.rend` | native_library | True | True | 1.00 | ✓ | The receiver DataFrame(columns=["a"]).style direct |
| `pandas_0_lc_1_8_33` | `pandas.io.formats.style.Styler.render` | `DataFrame(index=["a"]).style.render` | native_library | True | True | 1.00 | ✓ | The receiver is directly chained from pandas DataF |
| `pandas_3_lc_0_15_29` | `pandas.io.formats.style.Styler.render` | `result = " ".join(styler.render().s` | native_library | True | True | 0.99 | ✓ | The enclosing code explicitly instantiates Styler( |
| `pandas_4_lc_0_7_60` | `pandas.io.formats.style.Styler.render` | `result = self.df.style.set_table_at` | native_library | True | True | 1.00 | ✓ | The receiver self.df is a pandas DataFrame as evid |
| `pandas_32_lc_0_7_24` | `pandas.DataFrame.swapaxes` | `self.assert_eq(psdf.swapaxes(0, 1),` | wrapper_receiver | False | False | 0.95 | ✓ | The receiver 'psdf' is initialized via ps.from_pan |
| `pandas_32_lc_1_8_24` | `pandas.DataFrame.swapaxes` | `self.assert_eq(psdf.swapaxes(1, 0),` | wrapper_receiver | False | False | 0.95 | ✓ | The receiver psdf is a PySpark pandas API (pyspark |
| `pandas_32_lc_2_9_24` | `pandas.DataFrame.swapaxes` | `self.assert_eq(psdf.swapaxes("index` | wrapper_receiver | False | False | 0.99 | ✓ | The receiver psdf is a pyspark.pandas dataframe cr |
| `pandas_32_lc_3_10_24` | `pandas.DataFrame.swapaxes` | `self.assert_eq(psdf.swapaxes("colum` | wrapper_receiver | False | False | 0.95 | ✓ | The receiver 'psdf' is a pyspark.pandas (Koalas) D |
| `pandas_32_lc_4_11_30` | `pandas.DataFrame.swapaxes` | `self.assert_eq((psdf + 1).swapaxes(` | native_library | True | True | 0.95 | ✓ | The method swapaxes is called on a pandas DataFram |
| `pandas_32_lc_5_13_51` | `pandas.DataFrame.swapaxes` | `self.assertRaises(AssertionError, l` | wrapper_receiver | False | False | 0.95 | ✓ | The receiver 'psdf' is created using 'ps.from_pand |
| `pandas_32_lc_6_14_47` | `pandas.DataFrame.swapaxes` | `self.assertRaises(ValueError, lambd` | wrapper_receiver | False | False | 0.99 | ✓ | The receiver psdf is a pyspark.pandas dataframe cr |
| `pandas_33_lc_0_7_23` | `pandas.DataFrame.swapaxes` | `self.assert_eq(kdf.swapaxes(0, 1), ` | wrapper_receiver | False | False | 0.95 | ✓ | The receiver kdf is an instance of a Koalas/PySpar |
| `pandas_33_lc_3_10_23` | `pandas.DataFrame.swapaxes` | `self.assert_eq(kdf.swapaxes("column` | wrapper_receiver | False | False | 0.95 | ✓ | The receiver kdf is an instance of databricks.koal |
| `pandas_33_lc_4_11_29` | `pandas.DataFrame.swapaxes` | `self.assert_eq((kdf + 1).swapaxes(0` | native_library | True | True | 1.00 | ✓ | The call site explicitly invokes swapaxes on a pan |
| `pandas_33_lc_5_13_50` | `pandas.DataFrame.swapaxes` | `self.assertRaises(AssertionError, l` | wrapper_receiver | False | False | 0.95 | ✓ | The receiver 'kdf' is instantiated via 'ks.from_pa |
| `pandas_33_lc_6_14_46` | `pandas.DataFrame.swapaxes` | `self.assertRaises(ValueError, lambd` | wrapper_receiver | False | False | 0.95 | ✓ | The receiver kdf is an instance of a Koalas DataFr |
| `pandas_52_lc_0_5_76` | `pandas.DataFrame.iteritems` | `for (p_name, p_items), (k_name, k_i` | wrapper_receiver | False | False | 0.95 | ✓ | The receiver psser is an instance of pyspark.panda |
| `pandas_56_lc_0_5_75` | `pandas.DataFrame.iteritems` | `for (p_name, p_items), (k_name, k_i` | native_library | True | False | 0.95 | ✗ | The receiver kser is a Koalas/pyspark.pandas Serie |
| `pandas_33_lc_1_8_23` | `pandas.DataFrame.swapaxes` | `self.assert_eq(kdf.swapaxes(1, 0), ` | wrapper_receiver | False | False | 0.99 | ✓ | The receiver kdf is an instance of a Koalas/pyspar |
| `pandas_33_lc_2_9_23` | `pandas.DataFrame.swapaxes` | `self.assert_eq(kdf.swapaxes("index"` | wrapper_receiver | False | False | 0.99 | ✓ | The receiver kdf is an instance of a Koalas/pyspar |
| `pandas_70_lc_0_9_74` | `pandas.DataFrame.iteritems` | `for (p_name, p_items), (k_name, k_i` | wrapper_receiver | False | False | 0.95 | ✓ | The receiver psdf is a PySpark pandas-on-Spark Dat |
| `pandas_71_lc_0_9_73` | `pandas.DataFrame.iteritems` | `for (p_name, p_items), (k_name, k_i` | wrapper_receiver | False | False | 1.00 | ✓ | The receiver 'kdf' is instantiated via ks.from_pan |
| `scipy_5_lc_1_20_21` | `scipy.misc.comb` | `L_ts *= misc.comb( thisN, thisCorr ` | native_library | True | True | 0.95 | ✓ | The call site uses misc.comb in an active computat |
| `scipy_5_lc_2_21_29` | `scipy.misc.comb` | `LL_ts += np.log(misc.comb( thisN, t` | native_library | True | True | 0.95 | ✓ | The call site uses misc.comb in an active loop to  |
| `scipy_11_lc_0_7_45` | `scipy.misc.comb` | `return 1 - x * np.sum(np.exp(np.log` | native_library | True | True | 0.95 | ✓ | The call to misc.comb inside a mathematical comput |
| `scipy_13_lc_0_4_23` | `scipy.misc.comb` | `cdf += sc.misc.comb(n, y) * beta_fx` | native_library | True | True | 0.95 | ✓ | The call site uses sc.misc.comb where 'sc' commonl |
| `scipy_30_lc_1_25_45` | `numpy.product` | `lambda x: sum(x) > R_p, itertools.p` | standard_library | False | False | 1.00 | ✓ | The call site uses standard library module itertoo |
| `scipy_30_lc_2_37_49` | `numpy.product` | `lambda x: sum(x) > (2*R_p), itertoo` | standard_library | False | False | 0.99 | ✓ | The call site uses standard library module itertoo |
| `scipy_30_lc_3_45_21` | `numpy.product` | `itertools.product(range(R_p+1), rep` | standard_library | False | False | 1.00 | ✓ | The call site is invoking the standard Python libr |
| `scipy_34_lc_1_9_24` | `scipy.misc.comb` | `pi = ( misc.comb(n,i) * misc.comb((` | native_library | True | True | 0.95 | ✓ | The call site uses misc.comb in active execution p |
| `scipy_34_lc_2_9_41` | `scipy.misc.comb` | `pi = ( misc.comb(n,i) * misc.comb((` | native_library | True | True | 0.95 | ✓ | The call site uses misc.comb in active execution p |
| `scipy_34_lc_0_9_69` | `scipy.misc.comb` | `pi = ( misc.comb(n,i) * misc.comb((` | native_library | True | True | 0.95 | ✓ | The call site uses misc.comb in active execution p |
| `scipy_39_lc_0_10_17` | `scipy.misc.comb` | `choices = sm.comb(n, k)` | native_library | True | True | 0.85 | ✓ | The receiver 'sm' is used to call comb alongside ' |
| `scipy_52_lc_0_9_24` | `scipy.misc.comb` | `pi = ( misc.comb(n,i) * misc.comb((` | native_library | True | True | 0.95 | ✓ | The call site explicitly invokes misc.comb within  |
| `scipy_52_lc_1_9_41` | `scipy.misc.comb` | `pi = ( misc.comb(n,i) * misc.comb((` | native_library | True | True | 0.95 | ✓ | The call site explicitly invokes misc.comb within  |
| `scipy_52_lc_3_21_24` | `scipy.misc.comb` | `pi = ( misc.comb(n,i) * misc.comb((` | native_library | True | True | 0.95 | ✓ | The call site uses misc.comb in active execution p |
| `scipy_52_lc_4_21_41` | `scipy.misc.comb` | `pi = ( misc.comb(n,i) * misc.comb((` | native_library | True | True | 0.95 | ✓ | The call site uses misc.comb in active execution p |
| `scipy_52_lc_2_21_69` | `scipy.misc.comb` | `pi = ( misc.comb(n,i) * misc.comb((` | native_library | True | True | 0.95 | ✓ | The call site uses misc.comb in active execution p |
| `scipy_532_lc_1_12_22` | `scipy.misc.logsumexp` | `summed_up = -misc.logsumexp([item f` | native_library | True | True | 0.90 | ✓ | The code uses 'misc.logsumexp' in active execution |
| `scipy_533_lc_1_12_22` | `scipy.misc.logsumexp` | `summed_up = -misc.logsumexp([item f` | native_library | True | True | 0.90 | ✓ | The code uses 'misc.logsumexp' in active execution |
| `scipy_535_lc_0_8_21` | `scipy.misc.logsumexp` | `misc.logsumexp(self.model_and_run_l` | native_library | True | True | 0.95 | ✓ | The call site explicitly invokes misc.logsumexp, w |
| `scipy_536_lc_2_25_23` | `scipy.misc.logsumexp` | `return sp.misc.logsumexp([fn2(nk, r` | native_library | True | True | 0.90 | ✓ | The variable 'sp' is commonly used as an alias for |
| `scipy_536_lc_1_27_17` | `numpy.product` | `yvalues = it.product([0, 1], repeat` | standard_library | False | False | 0.95 | ✓ | The call it.product refers to itertools.product fr |
| `scipy_542_lc_0_5_30` | `scipy.misc.logsumexp` | `ans_ne = pymbar.utils.logsumexp(a, ` | native_library | True | False | 0.95 | ✗ | The call site invokes pymbar.utils.logsumexp, whic |
| `scipy_542_lc_1_6_33` | `scipy.misc.logsumexp` | `ans_no_ne = pymbar.utils.logsumexp(` | native_library | True | False | 0.95 | ✗ | The call site invokes pymbar.utils.logsumexp, whic |
| `scipy_543_lc_0_3_23` | `scipy.misc.logsumexp` | `ans = pymbar.utils.logsumexp(a)` | native_library | True | False | 0.95 | ✗ | The call site invokes pymbar.utils.logsumexp, whic |
| `scipy_544_lc_0_6_30` | `scipy.misc.logsumexp` | `ans_ne = pymbar.utils.logsumexp(a, ` | native_library | True | False | 0.95 | ✗ | The call site invokes pymbar.utils.logsumexp, whic |
| `scipy_544_lc_1_7_33` | `scipy.misc.logsumexp` | `ans_no_ne = pymbar.utils.logsumexp(` | native_library | True | False | 0.95 | ✗ | The call site invokes pymbar.utils.logsumexp, whic |
| `scipy_549_lc_0_6_28` | `scipy.misc.logsumexp` | `alpha[i,:] = [ misc.logsumexp(alpha` | native_library | True | True | 0.90 | ✓ | The call to misc.logsumexp inside an active calcul |
| `scipy_550_lc_1_6_27` | `scipy.misc.logsumexp` | `beta[i,:] = [ misc.logsumexp(ms[i][` | native_library | True | True | 0.90 | ✓ | The call to misc.logsumexp matches the deprecated  |
| `scipy_551_lc_0_5_20` | `scipy.misc.logsumexp` | `val += p * misc.logsumexp( xalphas(` | native_library | True | True | 0.90 | ✓ | The code invokes misc.logsumexp, which corresponds |
| `scipy_552_lc_1_10_17` | `scipy.misc.logsumexp` | `logz1 = misc.logsumexp( alpha[-1] )` | native_library | True | True | 0.90 | ✓ | The call to misc.logsumexp with statistical parame |
| `scipy_552_lc_2_11_17` | `scipy.misc.logsumexp` | `logz2 = misc.logsumexp( beta[0] )` | native_library | True | True | 0.90 | ✓ | The call site uses misc.logsumexp, which matches t |
| `scipy_553_lc_1_10_17` | `scipy.misc.logsumexp` | `logz1 = misc.logsumexp( alpha[-1] )` | native_library | True | True | 0.90 | ✓ | The call to misc.logsumexp with statistical parame |
| `scipy_553_lc_2_11_17` | `scipy.misc.logsumexp` | `logz2 = misc.logsumexp( beta[0] )` | native_library | True | True | 0.90 | ✓ | The call site uses misc.logsumexp, which matches t |
| `scipy_555_lc_0_23_26` | `scipy.misc.logsumexp` | `log_denominator = logsumexp(values,` | torch_tensor | False | False | 0.85 | ✓ | The call site logsumexp(values, dim=dim, keepdim=T |
| `scipy_561_lc_1_43_22` | `scipy.misc.logsumexp` | `return sm.logsumexp(a, axis=2).T` | native_library | True | True | 0.85 | ✓ | The receiver 'sm' strongly suggests an import alia |
| `scipy_578_lc_0_6_30` | `scipy.misc.logsumexp` | `ans_ne = pymbar.utils.logsumexp(a, ` | native_library | True | False | 0.95 | ✗ | The call site invokes pymbar.utils.logsumexp, whic |
| `scipy_578_lc_1_7_33` | `scipy.misc.logsumexp` | `ans_no_ne = pymbar.utils.logsumexp(` | native_library | True | False | 0.95 | ✗ | The call site invokes pymbar.utils.logsumexp, whic |
| `scipy_579_lc_1_12_22` | `scipy.misc.logsumexp` | `summed_up = -misc.logsumexp([item f` | native_library | True | True | 0.90 | ✓ | The code uses 'misc.logsumexp' in active execution |
| `scipy_580_lc_1_12_22` | `scipy.misc.logsumexp` | `summed_up = -misc.logsumexp([item f` | native_library | True | True | 0.90 | ✓ | The code uses 'misc.logsumexp' in active execution |
| `scipy_581_lc_0_8_21` | `scipy.misc.logsumexp` | `misc.logsumexp(self.model_and_run_l` | native_library | True | True | 0.95 | ✓ | The call site explicitly invokes misc.logsumexp, w |
| `scipy_583_lc_0_28_19` | `scipy.misc.logsumexp` | `return lp-sp.misc.logsumexp(lp)` | native_library | True | True | 0.95 | ✓ | The receiver 'sp' is standard shorthand for the Sc |
| `scipy_584_lc_0_27_22` | `scipy.misc.logsumexp` | `bf = numpy.exp(sm.logsumexp(-logp1)` | native_library | True | True | 0.90 | ✓ | The receiver 'sm' is used to call logsumexp which  |
| `scipy_584_lc_1_28_25` | `scipy.misc.logsumexp` | `- (sm.logsumexp(-logp2) - numpy.log` | native_library | True | True | 0.95 | ✓ | The variable 'sm' is commonly used as an alias for |
| `scipy_585_lc_0_6_23` | `scipy.misc.logsumexp` | `return a - sp.misc.logsumexp(a)` | native_library | True | True | 0.95 | ✓ | The call sp.misc.logsumexp invokes the deprecated  |
| `scipy_586_lc_0_6_19` | `scipy.misc.logsumexp` | `return sp.misc.logsumexp(a + b)` | native_library | True | True | 0.95 | ✓ | The call sp.misc.logsumexp directly invokes the de |
| `scipy_599_lc_0_16_25` | `pandas.DataFrame.swapaxes` | `pref_prob_distrns = np.swapaxes(pre` | native_library | True | False | 1.00 | ✗ | The call uses numpy's function np.swapaxes on a nu |
| `scipy_1866_lc_1_56_16` | `numpy.product` | `x_pro = product(*x)` | native_library | True | True | 0.90 | ✓ | The code unpacks parameter lists into numpy.produc |
| `scipy_1867_lc_3_59_16` | `numpy.product` | `x_pro = product(*x)` | native_library | True | True | 0.90 | ✓ | The code unpacks parameter lists into numpy.produc |