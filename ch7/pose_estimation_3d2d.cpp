#include <iostream>
#include <opencv2/core/core.hpp>

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

#include <opencv2/core/eigen.hpp>
//放在eigen后面，不然会报错
#include <opencv2/opencv.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
using namespace std;
using namespace cv;

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

void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );

void bundleAdjustment (
    const vector<Point3f> points_3d,
    const vector<Point2f> points_2d,
    const Mat& K,
    Mat& R, Mat& t
);

void ba_g2o_with_first_img_T(
    const vector<Point3f> pts1_3d,
    const vector<Point2f> pts1_2d,
    const vector<Point2f> pts2_2d,
    const Mat& K,
    Mat& R, Mat& t
);
void ceres_Ba(
    const vector<Point3f> pts1_3d,
    const vector<Point2f> pts1_2d,
    const vector<Point2f> pts2_2d,
    const Mat& K,
    Mat& R, Mat& t
);
int main ( int argc, char** argv )
{
    if ( argc != 5 )
    {
        cout<<"usage: pose_estimation_3d2d img1 img2 depth1 depth2"<<endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    // 建立3D点
    Mat d1 = imread ( argv[3], CV_LOAD_IMAGE_UNCHANGED );       
    // 深度图为16位无符号数，单通道图像
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    vector<Point3f> pts1_3d;
    vector<Point2f> pts1_2d;
    vector<Point2f> pts2_2d;
    for ( DMatch m:matches )
    {
        ushort d = d1.at<unsigned short> (keypoints_1[m.queryIdx].pt.y , keypoints_1[m.queryIdx].pt.x );
        if ( d == 0 )   // bad depth
            continue;
        float dd = d/5000.0;
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );
        pts1_3d.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );
	pts1_2d.push_back ( keypoints_1[m.queryIdx].pt );
        pts2_2d.push_back ( keypoints_2[m.trainIdx].pt );
    }

    cout<<"3d-2d pairs: "<<pts1_3d.size() <<endl;

    Mat r, t;
    solvePnP ( pts1_3d, pts2_2d, K, Mat(), r, t, false ); 
    // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    Mat R;
    cv::Rodrigues ( r, R ); 
    // r为旋转向量形式，用Rodrigues公式转换为矩阵

    cout<<"R="<<endl<<R<<endl;
    cout<<"t="<<endl<<t<<endl;

    cout<<"calling bundle adjustment"<<endl;

    bundleAdjustment ( pts1_3d, pts2_2d, K, R, t );
    cout<<"ba_g2o_with_first_img_T"<<endl;
    ba_g2o_with_first_img_T(pts1_3d,pts1_2d,pts2_2d,K,R,t);
    cout<<"ceres_Ba"<<endl;
    ceres_Ba(pts1_3d,pts1_2d,pts2_2d,K,r,t);
}

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
void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}

Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}

void bundleAdjustment (
    const vector< Point3f > points_3d,
    const vector< Point2f > points_2d,
    const Mat& K,
    Mat& R, Mat& t )
{
    // 初始化g2o
    typedef g2o::BlockSolver_6_3 Block;  
    // pose 维度为 6, landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); 
    // 线性方程求解器
    Block* solver_ptr = new Block ( linearSolver );     
    // 矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    Eigen::Matrix3d R_mat;
//     R_mat <<
//           R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
//                R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
//                R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
    cv2eigen(R,R_mat);
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
                            R_mat,
                            Eigen::Vector3d ( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) )
                        ) );
    optimizer.addVertex ( pose );

    int index = 1;
    for ( const Point3f p:points_3d )   // landmarks
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId ( index++ );
        point->setEstimate ( Eigen::Vector3d ( p.x, p.y, p.z ) );
        point->setMarginalized ( true ); 
	// g2o 中必须设置 marg 参见第十讲内容
        optimizer.addVertex ( point );
    }

    // parameter: camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters (
        K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0
    );
    camera->setId ( 0 );
    optimizer.addParameter ( camera );

    // edges
    int e_id=0;
    index =1;
    for ( const Point2f p:points_2d )
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId ( e_id++ );
        edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index++ ) ) );
        edge->setVertex ( 1, pose );
        edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );
        edge->setParameterId ( 0,0 );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
        optimizer.addEdge ( edge );
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose ( true );
    optimizer.initializeOptimization();
    optimizer.optimize ( 100 );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;

    cout<<endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;
}

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
