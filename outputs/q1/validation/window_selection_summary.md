# 窗口长度敏感性

16384 samples retained: the predeclared primary Logistic probe is within 0.02 of the best LOLO Macro-F1 and balanced accuracy; it also uses fewer windows and has lower median file-window CV than 8192. RBF-SVM remains a sensitivity probe, not the window-selection rule.

```csv
window_samples,duration_seconds,hop_samples,total_windows,mean_windows_per_file,selected_feature_count_descriptive,median_file_window_cv,model,macro_f1,balanced_accuracy,min_class_recall,runtime_seconds
8192,0.2560,4096,1718,30.6786,27,0.0843,logistic_regression,0.8749,0.9018,0.8333,8.5205
8192,0.2560,4096,1718,30.6786,27,0.0843,rbf_svm,0.7873,0.7976,0.6667,8.5205
16384,0.5120,8192,806,14.3929,28,0.0549,logistic_regression,0.8749,0.9018,0.8333,5.4330
16384,0.5120,8192,806,14.3929,28,0.0549,rbf_svm,0.7873,0.7976,0.6667,5.4330
32768,1.0240,16384,349,6.2321,29,0.0342,logistic_regression,0.8651,0.8899,0.7500,4.1837
32768,1.0240,16384,349,6.2321,29,0.0342,rbf_svm,0.8068,0.8185,0.6667,4.1837
```

所有结果均按原始 MAT 文件做 LOLO；本表不是目标域准确率。特征配置固定为 20 维 Transfer 特征，未在测试载荷上进行监督选择。
