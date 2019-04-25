# ch7 前端

## 自定义安装位置的第三方库 的cmake引用办法

​        针对自定义安装位置的g2o，没有提供.cmake之类的文件，不能像ceres一样直接`set( Ceres_DIR  "/home/leisure/all_ws/3rdparty/3rd_lib/Ceres-1.14.0/lib/cmake/Ceres")`。

​        `link_directories("/home/leisure/all_ws/3rdparty/3rd_lib/g2o-1/lib/")`可以将安装库文件的目录添加到cmakelists，链接库的时候只需要写中间的名字，不需要像ch6那样写绝对路径和库文件名字。

```cmake
link_directories("/home/leisure/all_ws/3rdparty/3rd_lib/g2o-1/lib/")

add_executable( pose_estimation_3d3d pose_estimation_3d3d.cpp )
target_link_libraries( pose_estimation_3d3d 
   ${OpenCV_LIBS}
   g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension 
   ${CSPARSE_LIBRARY}
   ${CERES_LIBRARIES} 
)
```

## 习题6

在pnp的优化中，将第一个相机的观测也考虑进来，将第一张图片的T加入作为一个顶点，并且固定住，第一张图片的观测作为边。

```c
void ba_g2o_with_first_img_T(
    const vector<Point3f> pts1_3d,
    const vector<Point2f> pts1_2d,
    const vector<Point2f> pts2_2d,
    const Mat& K,
    Mat& R, Mat& t
)
{
  typedef g2o::BlockSolver_6_3 Block;
  Block::LinearSolverType* linear_solver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>();
  Block* block_solver = new Block(linear_solver);
  g2o::OptimizationAlgorithmLevenberg* al_solver = new g2o::OptimizationAlgorithmLevenberg(block_solver);
  
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(al_solver);
  optimizer.setVerbose(true);
  
  //vertex_pose
  Eigen::Matrix3d R_mat ;
  cv2eigen(R,R_mat);
  Eigen::Vector3d t_mat ;
  cv2eigen(t,t_mat);
  
  for(int i_pose = 0;i_pose<2;i_pose++)
  {
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId(i_pose);
    pose->setFixed(i_pose==0);
    if(i_pose==1)
      pose->setEstimate(g2o::SE3Quat(R_mat,t_mat));
    else
      pose->setEstimate(g2o::SE3Quat());
    optimizer.addVertex(pose);
  }
  
  //vertex_land
  int i_land =2;
  
  for(const Point3f p:pts1_3d)
  {
    g2o::VertexSBAPointXYZ* land = new g2o::VertexSBAPointXYZ();
    land->setId(i_land++);
    land->setEstimate(Eigen::Vector3d(p.x,p.y,p.z));
    land->setMarginalized(true);
    optimizer.addVertex(land);
  }
  
  //camera
  g2o::CameraParameters* camera = new g2o::CameraParameters(K.at<double>(0,0), Eigen::Vector2d(K.at<double>(0,2),K.at<double>(1,2)),0);
  camera->setId(0);
  optimizer.addParameter(camera);
  
  //edges
  //img1
  for(int i =0;i<pts1_2d.size();i++)
  {
    g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
    edge->setId(i);
    edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i+2)));
    edge->setVertex(1,dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0)));
    edge->setMeasurement(Eigen::Vector2d(pts1_2d[i].x,pts1_2d[i].y));
    edge->setParameterId(0,0);
    edge->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edge);
  }
  //img2
  for(int i =0;i<pts2_2d.size();i++)
  {
    g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
    edge->setId(i+pts1_2d.size());
    edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i+2)));
    edge->setVertex(1,dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(1)));
    edge->setMeasurement(Eigen::Vector2d(pts2_2d[i].x,pts2_2d[i].y));
    edge->setParameterId(0,0);
    edge->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edge);
  }
  
  //cout<<"ba_g2o_with_first_img_T"<<endl;
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  
  optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.optimize(100);
  
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
  cout<<"time_used="<<time_used.count()<<endl;
  g2o::VertexSE3Expmap* T1 = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
  g2o::VertexSE3Expmap* T2 = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(1));
  
  Eigen::Isometry3d pose1 = T1->estimate();
  Eigen::Isometry3d pose2 = T2->estimate();
  cout<<"pose1="<<pose1.matrix()<<endl;
  cout<<"pose2="<<pose2.matrix()<<endl;
}
```
## 习题7

在ICP程序中将空间点也作为优化变量考虑进来，新定义链接3D点和T的边。

```c
class EdgeProjectXYZ2T: public g2o::BaseBinaryEdge<3,Eigen::Vector3d,g2o::VertexSBAPointXYZ,g2o::VertexSE3Expmap>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  EdgeProjectXYZ2T() {}
  
  virtual void computeError()
  {
    const g2o::VertexSBAPointXYZ* point = dynamic_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
    const g2o::VertexSE3Expmap* pose = dynamic_cast<const g2o::VertexSE3Expmap* >(_vertices[1]);
    _error = _measurement - pose->estimate()*point->estimate();
  }
  
  virtual void linearizeOplus()override final
  {
    
    g2o::VertexSE3Expmap* pose = dynamic_cast<g2o::VertexSE3Expmap*>(_vertices[1]);
    g2o::VertexSBAPointXYZ* point = dynamic_cast<g2o::VertexSBAPointXYZ*>(_vertices[0]);//边的第0个顶点 是点坐标，在添加边的时候设置
    Eigen::Vector3d xyz_trans = pose->estimate()*point->estimate();
    double x =xyz_trans[0];
    double y =xyz_trans[1];
    double z =xyz_trans[2];
    _jacobianOplusXi = -pose->estimate().rotation().toRotationMatrix();//关于第0个顶点也就是点坐标的导数 -R
    
    _jacobianOplusXj(0,0) = 0;           //关于第1个顶点也就是位姿的导数。
    _jacobianOplusXj(0,1) = -z;
    _jacobianOplusXj(0,2) = y;
    _jacobianOplusXj(0,3) = -1;
    _jacobianOplusXj(0,4) = 0;
    _jacobianOplusXj(0,5) = 0;

    _jacobianOplusXj(1,0) = z;
    _jacobianOplusXj(1,1) = 0;
    _jacobianOplusXj(1,2) = -x;
    _jacobianOplusXj(1,3) = 0;
    _jacobianOplusXj(1,4) = -1;
    _jacobianOplusXj(1,5) = 0;

    _jacobianOplusXj(2,0) = -y;
    _jacobianOplusXj(2,1) = x;
    _jacobianOplusXj(2,2) = 0;
    _jacobianOplusXj(2,3) = 0;
    _jacobianOplusXj(2,4) = 0;
    _jacobianOplusXj(2,5) = -1;
  }
  
  bool read(istream& in){}
  bool write(ostream& out) const{}
};
```
雅克比矩阵的位置Xi Xj和自己设置的顶点顺序有关
```c
void g2o_ba_with_img2_point(
    const vector<Point3f>& pts1,
    const vector<Point3f>& pts2,
    Mat& R, Mat& t)
{
  typedef g2o::BlockSolver_6_3 Block;
  Block::LinearSolverType* linear_solver = new g2o::LinearSolverEigen<Block::PoseMatrixType>();
  Block* block_solver = new Block(linear_solver);
  g2o::OptimizationAlgorithmLevenberg* al_solver = new g2o::OptimizationAlgorithmLevenberg(block_solver);
  //g2o::OptimizationAlgorithmGaussNewton* al_solver = new g2o::OptimizationAlgorithmGaussNewton(block_solver);
  
  g2o::SparseOptimizer optimizer;
  optimizer.setAlgorithm(al_solver);
  
  auto pose = new g2o::VertexSE3Expmap();
  pose->setId(0);
  
  Eigen::Matrix3d R_mat;
  Eigen::Vector3d t_mat;
  cv2eigen(R,R_mat);
  cv2eigen(t,t_mat);
  
  pose->setEstimate(g2o::SE3Quat(R_mat,t_mat));
  optimizer.addVertex(pose);
  
  int pointIndex=1;
  for(auto& p:pts2)
  {
    auto point = new g2o::VertexSBAPointXYZ();
    point->setId(pointIndex++);
    point->setMarginalized(true);
    point->setEstimate(Eigen::Vector3d(p.x,p.y,p.z));
    optimizer.addVertex(point);
  }
  
  vector<EdgeProjectXYZ2T*> edges;
  for(int i = 0;i<pts1.size();i++)
  {
    auto edge = new EdgeProjectXYZ2T();
    edge->setId(i);
    edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i+1)));
    //edge->setVertex(1,dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
    edge->setVertex(1,dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0)));
    edge->setMeasurement(Eigen::Vector3d(pts1[i].x,pts1[i].y,pts1[i].z));
    //edge->setInformation(Eigen::Matrix3d::Identity());
    edge->setInformation(Eigen::Matrix3d::Identity()*1e4);
    optimizer.addEdge(edge);
    edges.push_back(edge);
    
  }
  
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.setVerbose( true );
  optimizer.initializeOptimization();
  optimizer.optimize(15);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
  cout<<"optimization costs time: "<<time_used.count()<<" seconds."<<endl;

  cout<<endl<<"after optimization:"<<endl;
  cout<<"T="<<endl<<Eigen::Isometry3d( pose->estimate() ).matrix()<<endl;
  for(int i=0;i<8;i++)
  {
    cout<<"pts1="<<pts1[i]<<endl;
    cout<<"pts2="<<pts2[i]<<endl;
    cout<<"ba_pts2="<<dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i+1))->estimate().transpose()<<endl;
    cout<<"ba_T*ba_pts2="<<(dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0))->estimate()*dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(i+1))->estimate()).transpose()<<endl;
  }
}
```

## 习题10

在Ceres中实现PnP和ICP的优化

### Ceres优化Pnp 

误差函数2维,只优化R和t

```c
struct cost_function
{
  cost_function(Point3f p1,Point2f p2):_p1(p1),_p2(p2){}
  template<typename T>
  bool operator()(const T* const cere_r,const T* const cere_t,T* residual)const
  {
    T p_1[3];
    T p_2[3];
    p_1[0]=T(_p1.x);
    p_1[1]=T(_p1.y);
    p_1[2]=T(_p1.z);
    //cout<<"pts1_3d: "<<p_1[0]<<" "<<p_1[1]<<"  "<<p_1[2]<<endl;
    ceres::AngleAxisRotatePoint(cere_r,p_1,p_2);

    p_2[0]=p_2[0]+cere_t[0];
    p_2[1]=p_2[1]+cere_t[1];
    p_2[2]=p_2[2]+cere_t[2];

    const T x=p_2[0]/p_2[2];
    const T y=p_2[1]/p_2[2];
    //三维点重投影计算的像素坐标
    const T u=x*520.9+325.1;
    const T v=y*521.0+249.7;
    
   
    //观测的在图像坐标下的值
    T u1=T(_p2.x);
    T v1=T(_p2.y);
 
    residual[0]=u-u1;
    residual[1]=v-v1;
    return true;
  }
   Point3f _p1;
   Point2f _p2;
};
```
求出的是角轴,利用cv::Rodrigues转换成矩阵形式

```c
void ceres_Ba(
    const vector<Point3f> pts1_3d,
    const vector<Point2f> pts1_2d,
    const vector<Point2f> pts2_2d,
    const Mat& K,
    Mat& r, Mat& t
)
{
      
    double cere_r[3],cere_t[3];
//     cere_r[0]=r.at<double>(0,0);
//     cere_r[1]=r.at<double>(1,0);
//     cere_r[2]=r.at<double>(2,0);
    cere_r[0]=0;
    cere_r[1]=1;
    cere_r[2]=2;

//     cere_t[0]=t.at<double>(0,0);
//     cere_t[1]=t.at<double>(1,0);
//     cere_t[2]=t.at<double>(2,0);
    cere_t[0]=0;
    cere_t[1]=0;
    cere_t[2]=0;

    ceres::Problem problem;
    for(int i=0;i<pts1_3d.size();i++)
    {
      ceres::CostFunction* costfun=new ceres::AutoDiffCostFunction<cost_function,2,3,3>(new cost_function(pts1_3d[i],pts2_2d[i]));
      problem.AddResidualBlock(costfun,NULL,cere_r,cere_t);
      //注意，cere_r不能为Mat类型      输入为两个3维向量
    }
  

    ceres::Solver::Options option;
    option.linear_solver_type=ceres::DENSE_SCHUR;
    //输出迭代信息到屏幕
    option.minimizer_progress_to_stdout=true;
    //显示优化信息
    ceres::Solver::Summary summary;
    //开始求解
    ceres::Solve(option,&problem,&summary);
    //显示优化信息
    cout<<summary.BriefReport()<<endl;

    cout<<"----------------optional after--------------------"<<endl;

    Mat cam_3d = ( Mat_<double> ( 3,1 )<<cere_r[0],cere_r[1],cere_r[2]);
    Mat cam_9d;
    cv::Rodrigues ( cam_3d, cam_9d ); 
    // r为旋转向量形式，用Rodrigues公式转换为矩阵

    cout<<"cam_9d:"<<endl<<cam_9d<<endl;

    cout<<"cam_t:"<<cere_t[0]<<"  "<<cere_t[1]<<"  "<<cere_t[2]<<endl;
}
```

### Ceres优化ICP

误差函数3维，只优化R和t

```c
struct CURVE_FITTING_COST
{
    CURVE_FITTING_COST ( Point3f p1, Point3f p2 ) : _p1 ( p1 ), _p2 ( p2 ) {}
    // 残差的计算
    template <typename T>
    bool operator() (const T* const cere_r,const T* const cere_t,T* residual ) const
    {
        T p_1[3];
        T p_2[3];
        p_2[0]=T(_p2.x);
        p_2[1]=T(_p2.y);
        p_2[2]=T(_p2.z);
        ceres::AngleAxisRotatePoint(cere_r,p_2,p_1);
        p_1[0]=p_1[0]+cere_t[0];
        p_1[1]=p_1[1]+cere_t[1];
        p_1[2]=p_1[2]+cere_t[2];
	residual[0] = T(_p1.x)-p_1[0];
	residual[1] = T(_p1.y)-p_1[1];
	residual[2] = T(_p1.z)-p_1[2];
        return true;
    }
    const Point3f _p1, _p2;    //
};
```

​	        求出的也是旋转向量形式，利用eigen转换为旋转矩阵。

```c
void ceres_BA(const vector< Point3f >& pts1,const vector< Point3f >& pts2)
{   
    double cere_r[3];
    double cere_t[3];
    
    ceres::Problem problem;
    for ( int i=0; i<pts1.size(); i++ )
    {
        ceres::CostFunction* costfunction=new ceres::AutoDiffCostFunction<CURVE_FITTING_COST,3,3,3>(new CURVE_FITTING_COST(pts1[i],pts2[i]));
        problem.AddResidualBlock(costfunction,nullptr,cere_r,cere_t);
	//nullptr是核函数
    }

    // 配置求解器
    ceres::Solver::Options options;     //
    options.linear_solver_type = ceres::DENSE_SCHUR;  //
    options.minimizer_progress_to_stdout = true;   // 

    ceres::Solver::Summary summary;                // 
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve ( options, &problem, &summary );  // 
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;
    cout<<summary.BriefReport() <<endl;
    cout<<"cere_t="<<cere_t[0]<<" "<<cere_t[1]<<" "<<cere_t[2]<<endl;
    //Eigen::AngleAxisd aa(cere_r[0],cere_r[1],cere_r[2]);
    cout<<"cere_r="<<endl<<cere_r[0]<<" "<<cere_r[1]<<" "<<cere_r[2]<<endl<<endl;
    double leng=sqrt(cere_r[0]*cere_r[0]+cere_r[1]*cere_r[1]+cere_r[2]*cere_r[2]);
    Eigen::Vector3d zhou(cere_r[0]/leng,cere_r[1]/leng,cere_r[2]/leng);
    Eigen::AngleAxisd aa(leng,zhou);
    cout<<aa.matrix()<<endl;
    cout<<endl;
    
}
```
## cv::Mat 和eigen的矩阵转换

```c
include <opencv2/core/eigen.hpp>
//放在eigen后面，不然会报错
Eigen::Matrix3d R_mat;
Eigen::Vector3d t_mat;
cv2eigen(R,R_mat);
cv2eigen(t,t_mat);
```
这样利用opencv的函数转换方便。

