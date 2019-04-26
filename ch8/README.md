# ch8 直接法

## 自定义安装位置的g2o库的引用 链接打包

把一堆库打包成一个变量`set( G2O_LIBS 
    g2o_core g2o_types_sba g2o_solver_csparse g2o_stuff g2o_csparse_extension )`，

方便链接时时调用。

```cmake
include_directories( "/usr/include/eigen3" 
                     "/home/leisure/all_ws/3rdparty/3rd_lib/g2o-1/include" )
link_directories("/home/leisure/all_ws/3rdparty/3rd_lib/g2o-1/lib/")

set( G2O_LIBS 
    g2o_core g2o_types_sba g2o_solver_csparse g2o_stuff g2o_csparse_extension )

add_executable( direct_sparse direct_sparse.cpp )
target_link_libraries( direct_sparse ${OpenCV_LIBS} ${G2O_LIBS} )

add_executable( direct_semidense direct_semidense.cpp )
target_link_libraries( direct_semidense ${OpenCV_LIBS} ${G2O_LIBS} )
```

## 稀疏直接法和半稠密直接法

稀疏直接法有用到opencv提取fast特征点，再从中选取梯度较大的点作为边，优化。

半稠密用除去图像四周的，中间所有点，提取梯度较大的作为边，优化，所以跟踪很多点。

深入理解，==自己推导一遍==光度误差关于位姿李代数的雅克比矩阵推导，链式法则。好像直接求偏导就好，书上把误差函数展开了，导致书上的q和李代数左扰动公式有点不一样，不是坐乘扰动，而是小量的左边叉积。delta epilong^ exp(epilong) *P。结果是对的，看怎么理解了。

最后代码里雅克比矩阵因为误差定义为 I(P2) - I(P1) ，预测值减去测量值，和书上差一个负号，书上是测量值减去预测值。

```c
class EdgeSE3ProjectDirect: public BaseUnaryEdge< 1, double, VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectDirect() {}

    EdgeSE3ProjectDirect ( Eigen::Vector3d point, float fx, float fy, float cx, float cy, cv::Mat* image )
        : x_world_ ( point ), fx_ ( fx ), fy_ ( fy ), cx_ ( cx ), cy_ ( cy ), image_ ( image )
    {}

    virtual void computeError()
    {
        const VertexSE3Expmap* v  =static_cast<const VertexSE3Expmap*> ( _vertices[0] );
        Eigen::Vector3d x_local = v->estimate().map ( x_world_ );
        float x = x_local[0]*fx_/x_local[2] + cx_;
        float y = x_local[1]*fy_/x_local[2] + cy_;
        // check x,y is in the image
        if ( x-4<0 || ( x+4 ) >image_->cols || ( y-4 ) <0 || ( y+4 ) >image_->rows )
        {
            _error ( 0,0 ) = 0.0;
            this->setLevel ( 1 );
        }
        else
        {
            _error ( 0,0 ) = getPixelValue ( x,y ) - _measurement;
        }
    }

    // plus in manifold
    virtual void linearizeOplus( )
    {
        if ( level() == 1 )
        {
            _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
            return;
        }
        VertexSE3Expmap* vtx = static_cast<VertexSE3Expmap*> ( _vertices[0] );
        Eigen::Vector3d xyz_trans = vtx->estimate().map ( x_world_ );   // q in book

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0/xyz_trans[2];
        double invz_2 = invz*invz;

        float u = x*fx_*invz + cx_;
        float v = y*fy_*invz + cy_;

        // jacobian from se3 to u,v
        // NOTE that in g2o the Lie algebra is (\omega, \epsilon), where \omega is so(3) and \epsilon the translation
        Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

        jacobian_uv_ksai ( 0,0 ) = - x*y*invz_2 *fx_;
        jacobian_uv_ksai ( 0,1 ) = ( 1+ ( x*x*invz_2 ) ) *fx_;
        jacobian_uv_ksai ( 0,2 ) = - y*invz *fx_;
        jacobian_uv_ksai ( 0,3 ) = invz *fx_;
        jacobian_uv_ksai ( 0,4 ) = 0;
        jacobian_uv_ksai ( 0,5 ) = -x*invz_2 *fx_;

        jacobian_uv_ksai ( 1,0 ) = - ( 1+y*y*invz_2 ) *fy_;
        jacobian_uv_ksai ( 1,1 ) = x*y*invz_2 *fy_;
        jacobian_uv_ksai ( 1,2 ) = x*invz *fy_;
        jacobian_uv_ksai ( 1,3 ) = 0;
        jacobian_uv_ksai ( 1,4 ) = invz *fy_;
        jacobian_uv_ksai ( 1,5 ) = -y*invz_2 *fy_;

        Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

        jacobian_pixel_uv ( 0,0 ) = ( getPixelValue ( u+1,v )-getPixelValue ( u-1,v ) ) /2;
        jacobian_pixel_uv ( 0,1 ) = ( getPixelValue ( u,v+1 )-getPixelValue ( u,v-1 ) ) /2;

        _jacobianOplusXi = jacobian_pixel_uv*jacobian_uv_ksai;
    }

    // dummy read and write functions because we don't care...
    virtual bool read ( std::istream& in ) {}
    virtual bool write ( std::ostream& out ) const {}

protected:
    // get a gray scale value from reference image (bilinear interpolated)
    inline float getPixelValue ( float x, float y )
    {
        uchar* data = & image_->data[ int ( y ) * image_->step + int ( x ) ];
        float xx = x - floor ( x );
        float yy = y - floor ( y );
        return float (
                   ( 1-xx ) * ( 1-yy ) * data[0] +
                   xx* ( 1-yy ) * data[1] +
                   ( 1-xx ) *yy*data[ image_->step ] +
                   xx*yy*data[image_->step+1]
               );
    }
public:
    Eigen::Vector3d x_world_;   // 3D point in world frame
    float cx_=0, cy_=0, fx_=0, fy_=0; // Camera intrinsics
    cv::Mat* image_=nullptr;    // reference image
};
```

还有就是其中包含的==灰度的插值计算==和==灰度的梯度计算==。



