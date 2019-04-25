#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <opencv2/core/eigen.hpp>
//放在eigen后面，不然会报错
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void ceres_BA(const vector< Point3f >& pts1,const vector< Point3f >& pts2);
void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );

void pose_estimation_3d3d (
    const vector<Point3f>& pts1,
    const vector<Point3f>& pts2,
    Mat& R, Mat& t
);

void bundleAdjustment(
    const vector<Point3f>& pts1,
    const vector<Point3f>& pts2,
    Mat& R, Mat& t
);

void g2o_ba_with_img2_point(
    const vector<Point3f>& pts1,
    const vector<Point3f>& pts2,
    Mat& R, Mat& t
);

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
// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectXYZRGBDPoseOnly( const Eigen::Vector3d& point ) : _point(point) {}

    virtual void computeError()
    {
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
        // measurement is p celiangce, point is p'yucezhi jisuanzhi
        //_error = _measurement - pose->estimate().map( _point );
	_error = _measurement - pose->estimate()* _point ;
    }

    virtual void linearizeOplus()
    {
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyz_trans = pose->estimate()*_point;
	//Eigen::Vector3d xyz_trans = T.map(_point);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        _jacobianOplusXi(0,0) = 0;
        _jacobianOplusXi(0,1) = -z;
        _jacobianOplusXi(0,2) = y;
        _jacobianOplusXi(0,3) = -1;
        _jacobianOplusXi(0,4) = 0;
        _jacobianOplusXi(0,5) = 0;

        _jacobianOplusXi(1,0) = z;
        _jacobianOplusXi(1,1) = 0;
        _jacobianOplusXi(1,2) = -x;
        _jacobianOplusXi(1,3) = 0;
        _jacobianOplusXi(1,4) = -1;
        _jacobianOplusXi(1,5) = 0;

        _jacobianOplusXi(2,0) = -y;
        _jacobianOplusXi(2,1) = x;
        _jacobianOplusXi(2,2) = 0;
        _jacobianOplusXi(2,3) = 0;
        _jacobianOplusXi(2,4) = 0;
        _jacobianOplusXi(2,5) = -1;
    }

    bool read ( istream& in ) {}
    bool write ( ostream& out ) const {}
protected:
    Eigen::Vector3d _point;
};

int main ( int argc, char** argv )
{
    if ( argc != 5 )
    {
        cout<<"usage: pose_estimation_3d3d img1 img2 depth1 depth2"<<endl;
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
    Mat depth1 = imread ( argv[3], CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
    Mat depth2 = imread ( argv[4], CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    vector<Point3f> pts1, pts2;

    for ( DMatch m:matches )
    {
        ushort d1 = depth1.ptr<unsigned short> ( int ( keypoints_1[m.queryIdx].pt.y ) ) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
        ushort d2 = depth2.ptr<unsigned short> ( int ( keypoints_2[m.trainIdx].pt.y ) ) [ int ( keypoints_2[m.trainIdx].pt.x ) ];
        if ( d1==0 || d2==0 )   // bad depth
            continue;
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );
        Point2d p2 = pixel2cam ( keypoints_2[m.trainIdx].pt, K );
        float dd1 = float ( d1 ) /5000.0;
        float dd2 = float ( d2 ) /5000.0;
        pts1.push_back ( Point3f ( p1.x*dd1, p1.y*dd1, dd1 ) );
        pts2.push_back ( Point3f ( p2.x*dd2, p2.y*dd2, dd2 ) );
    }

    cout<<"3d-3d pairs: "<<pts1.size() <<endl;
    Mat R, t;
    pose_estimation_3d3d ( pts1, pts2, R, t );
    cout<<"ICP via SVD results: "<<endl;
    cout<<"R = "<<R<<endl;
    cout<<"t = "<<t<<endl;
    cout<<"R_inv = "<<R.t() <<endl;
    cout<<"t_inv = "<<-R.t() *t<<endl;
    //不是-t，注意注意

    cout<<"calling bundle adjustment"<<endl<<endl;

    bundleAdjustment( pts1, pts2, R, t );
    cout<<"ceres_start"<<endl<<endl;
    ceres_BA(pts1,pts2);
    cout<<"g2o_ba_with_img2_point"<<endl;
    g2o_ba_with_img2_point(pts1,pts2,R,t);
    // verify p1 = R*p2 + t
    for ( int i=0; i<5; i++ )
    {
        cout<<"p1 = "<<pts1[i]<<endl;
        cout<<"p2 = "<<pts2[i]<<endl;
        cout<<"(R*p2+t) = "<<
            R * (Mat_<double>(3,1)<<pts2[i].x, pts2[i].y, pts2[i].z) + t
            <<endl;
        cout<<endl;
    }
}

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
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create("BruteForce-Hamming");
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

void pose_estimation_3d3d (
    const vector<Point3f>& pts1,
    const vector<Point3f>& pts2,
    Mat& R, Mat& t
)
{
    Point3f p1, p2;     // center of mass
    int N = pts1.size();
    for ( int i=0; i<N; i++ )
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = Point3f( Vec3f(p1) /  N);
    p2 = Point3f( Vec3f(p2) / N);
    vector<Point3f>     q1 ( N ), q2 ( N ); // remove the center
    for ( int i=0; i<N; i++ )
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for ( int i=0; i<N; i++ )
    {
        W += Eigen::Vector3d ( q1[i].x, q1[i].y, q1[i].z ) * Eigen::Vector3d ( q2[i].x, q2[i].y, q2[i].z ).transpose();
    }
    cout<<"W="<<W<<endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd ( W, Eigen::ComputeFullU|Eigen::ComputeFullV );
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    if (U.determinant() * V.determinant() < 0)
	{
        for (int x = 0; x < 3; ++x)
        {
            U(x, 2) *= -1;
        }
	}
    
    cout<<"U="<<U<<endl;
    cout<<"V="<<V<<endl;

    Eigen::Matrix3d R_ = U* ( V.transpose() );
    Eigen::Vector3d t_ = Eigen::Vector3d ( p1.x, p1.y, p1.z ) - R_ * Eigen::Vector3d ( p2.x, p2.y, p2.z );

    // convert to cv::Mat
    eigen2cv(R_,R);
    eigen2cv(t_,t);
//     R = ( Mat_<double> ( 3,3 ) <<
//           R_ ( 0,0 ), R_ ( 0,1 ), R_ ( 0,2 ),
//           R_ ( 1,0 ), R_ ( 1,1 ), R_ ( 1,2 ),
//           R_ ( 2,0 ), R_ ( 2,1 ), R_ ( 2,2 )
//         );
//     t = ( Mat_<double> ( 3,1 ) << t_ ( 0,0 ), t_ ( 1,0 ), t_ ( 2,0 ) );
}

void bundleAdjustment (
    const vector< Point3f >& pts1,
    const vector< Point3f >& pts2,
    Mat& R, Mat& t )
{
    // 初始化g2o
    typedef g2o::BlockSolver_6_3 Block;  
    // pose维度为 6, landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>(); 
    // 线性方程求解器
    Block* solver_ptr = new Block( linearSolver );      
    // 矩阵块求解器
    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm( solver );

    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    pose->setId(0);
    
    Eigen::Matrix3d R_mat;
    Eigen::Vector3d t_mat;
    cv2eigen(R,R_mat);
    cv2eigen(t,t_mat);
    pose->setEstimate( g2o::SE3Quat(R_mat,t_mat) );
    optimizer.addVertex( pose );

    // edges
    int index = 0;
    vector<EdgeProjectXYZRGBDPoseOnly*> edges;
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        EdgeProjectXYZRGBDPoseOnly* edge = new EdgeProjectXYZRGBDPoseOnly(Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z) );
        edge->setId( index++ );
        edge->setVertex( 0, dynamic_cast<g2o::VertexSE3Expmap*> (pose) );
        edge->setMeasurement( Eigen::Vector3d(
            pts1[i].x, pts1[i].y, pts1[i].z) );
        edge->setInformation( Eigen::Matrix3d::Identity()*1e4 );
        optimizer.addEdge(edge);
        edges.push_back(edge);
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose( true );
    optimizer.initializeOptimization();
    optimizer.optimize(8);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<<"optimization costs time: "<<time_used.count()<<" seconds."<<endl;

    cout<<endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d( pose->estimate() ).matrix()<<endl;

}

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
