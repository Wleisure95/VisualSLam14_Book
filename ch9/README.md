# ch9 前端工程

注意pnp优化求出来的是Tcr 或者是 Tcw，最后都需要转换为Twc。表示当前帧在世界坐标系下的姿态。

```c
SE3 Twc = pFrame->T_c_w_.inverse();

Eigen::Quaterniond q = Eigen::Quaterniond(Twc.rotation_matrix());
t<<rgb_times[i]<<" "<<Twc.matrix()(0,3)<<" "<<Twc.matrix()(1,3)<<" "<<Twc.matrix()(2,3)<<" "<<q.coeffs()[0]<<" "<<q.coeffs()[1]<<" "<<q.coeffs()[2]<<" "<<q.coeffs()[3]<<endl;
```

将位姿保存为txt文件，在当前运行环境目录下，和groundtruth.txt作对比。



## vo2

仅有帧间匹配并且只用了pnp，得到Tcr ，算出Tcw = Tcr * Trw

```c
curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;  // T_c_w = T_c_r*T_r_w 
```

## vo3

在帧间匹配pnp后加了g2o优化

## vo4

直接将当前帧和局部地图匹配，直接得到Tcw，更新维护局部地图。

```c
curr_->T_c_w_ = T_c_w_estimated_;
optimizeMap();
```

