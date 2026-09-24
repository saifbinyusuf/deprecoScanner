# Prompt v2 Evaluation on Fresh Wrapper Candidates

- **Status**: PASS
- **Benchmark Contamination Check**: PASSED (0 of 150 benchmark items present in evaluation set)
- **Cache Key Isolation Check**: PASSED (all v2 cache keys incorporate prompt_version v2.0 and differ from v1 keys)
- **Total Fresh Items Evaluated**: 13
- **Native Pandas Calls (True Deprecations)**: 6
- **Wrapper Mimic Calls (Benign Lookalikes)**: 7

## Evaluated Candidates

| ID | Sample ID | Target API | Call Site Snippet | Receiver Classification | Ground Truth Under Rule |
|---|---|---|---|---|---|
| fresh_wrapper_01 | `pandas_31` | `pandas.DataFrame.swapaxes` | `pandas_result = pandas_df.swapaxes(axis1, axi` | native_pandas | **True** |
| fresh_wrapper_02 | `pandas_31` | `pandas.DataFrame.swapaxes` | `modin_result = modin_df.swapaxes(axis1, axis2` | wrapper_lookalike | **False** |
| fresh_wrapper_03 | `pandas_56` | `pandas.Series.iteritems` | `for (p_name, p_items), (k_name, k_items) in z` | native_pandas | **True** |
| fresh_wrapper_04 | `pandas_56` | `pandas.DataFrame.iteritems` | `for (p_name, p_items), (k_name, k_items) in z` | native_pandas | **True** |
| fresh_wrapper_05 | `pandas_70` | `pandas.DataFrame.iteritems` | `for (p_name, p_items), (k_name, k_items) in z` | native_pandas | **True** |
| fresh_wrapper_06 | `pandas_70` | `pandas.DataFrame.iteritems` | `for (p_name, p_items), (k_name, k_items) in z` | native_pandas | **True** |
| fresh_wrapper_07 | `pandas_104` | `pandas.DataFrame.applymap` | `pandas_result = pandas_df.applymap(testfunc)` | native_pandas | **True** |
| fresh_wrapper_08 | `pandas_104` | `pandas.DataFrame.applymap` | `modin_df.applymap(testfunc)` | wrapper_lookalike | **False** |
| fresh_wrapper_09 | `pandas_104` | `pandas.DataFrame.applymap` | `modin_result = modin_df.applymap(testfunc)` | wrapper_lookalike | **False** |
| fresh_wrapper_10 | `pandas_116` | `pandas.DataFrame.applymap` | `pdf_result = df.applymap(lambda x: x + 1)` | wrapper_lookalike | **False** |
| fresh_wrapper_11 | `pandas_116` | `pandas.DataFrame.applymap` | `pdf_result = df.applymap(lambda x: (x, x))` | wrapper_lookalike | **False** |
| fresh_wrapper_12 | `pandas_116` | `pandas.DataFrame.applymap` | `pdf_result = df.applymap(lambda x: x + 1)` | wrapper_lookalike | **False** |
| fresh_wrapper_13 | `pandas_116` | `pandas.DataFrame.applymap` | `pdf_result = df.applymap(lambda x: (x, x))` | wrapper_lookalike | **False** |
