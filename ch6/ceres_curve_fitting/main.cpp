#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/LU>

using namespace std;

// 代价函数的计算模型
struct CURVE_FITTING_COST
{
    CURVE_FITTING_COST ( double x, double y ) : _x ( x ), _y ( y ) {}
    // 残差的计算
    template <typename T>
    bool operator() 
    ( const T* const abc, T* residual ) const     
        // 残差
    {
        residual[0] = T ( _y ) - ceres::exp ( abc[0]*T ( _x ) *T ( _x ) + abc[1]*T ( _x ) + abc[2] ); 
	// y-exp(ax^2+bx+c)
        return true;
    }
    const double _x, _y;    
    // x,y数据
};

double computeError(double abc[3],vector<double> x_data,vector<double> y_data )
{
    double sumError = 0;
    for(int i = 0;i<x_data.size();i++)
    {
        double ei = y_data[i] - ceres::exp(abc[0]*x_data[i]*x_data[i]+abc[1]*x_data[i]+abc[2]);
	sumError += ei;
    }
    return sumError;
}

Eigen::Vector3d LinearizeAndSolve(double abc[3],vector<double> x_data,vector<double> y_data)
{
    Eigen::Matrix3d H;
    Eigen::Vector3d b;
    H.setZero();
    b.setZero();
    
    for(int i=0;i<x_data.size();i++)
    {
	double ei = y_data[i] - ceres::exp(abc[0]*x_data[i]*x_data[i]+abc[1]*x_data[i]+abc[2]);
	Eigen::Vector3d AiT;
	AiT(0,0) = -ceres::exp(abc[0]*x_data[i]*x_data[i]+abc[1]*x_data[i]+abc[2])*x_data[i]*x_data[i];
        AiT(1,0) = -ceres::exp(abc[0]*x_data[i]*x_data[i]+abc[1]*x_data[i]+abc[2])*x_data[i];
	AiT(2,0) = -ceres::exp(abc[0]*x_data[i]*x_data[i]+abc[1]*x_data[i]+abc[2]);
        
	H += AiT*AiT.transpose();
	b += AiT*ei;
    }
    Eigen::Vector3d dabc = -H.lu().solve(b);
    return dabc;
}
int main ( int argc, char** argv )
{
    double a=1.0, b=2.0, c=1.0;         // 真实参数值
    int N=100;                          // 数据点
    double w_sigma=1.0;                 // 噪声Sigma值
    cv::RNG rng;                        // OpenCV随机数产生器
    // abc参数的估计值
    double abc[3] = {0,0,0};            

    vector<double> x_data, y_data;      // 数据
    ofstream t("data.txt");
    cout<<"generating data: "<<endl;
    for ( int i=0; i<N; i++ )
    {
        double x = i/100.0;
        x_data.push_back ( x );
        y_data.push_back (
            exp ( a*x*x + b*x + c ) + rng.gaussian ( w_sigma )
        );
	t<<x_data[i]<<" "<<y_data[i]<<endl;
        cout<<x_data[i]<<" "<<y_data[i]<<endl;
    }
    t.close();
    
    cout<<"my_GN_start: "<<endl;
    double initError = computeError(abc,x_data,y_data);
    cout<<"my_init_Error = "<<initError<<endl;
    
    int maxIteration = 200;
    double epsilon = 1e-6;
    
    chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
    for(int i = 0;i<maxIteration;i++)
    {
	cout<<"Iteration:"<<i<<endl;
	Eigen::Vector3d dabc = LinearizeAndSolve(abc,x_data,y_data);
	
	abc[0] += dabc(0);
	abc[1] += dabc(1);
	abc[2] += dabc(2);
	
	double maxError = -1;
	for(int i=0;i<3;i++)
	{
	  if(maxError < fabs(dabc(i)))
	  {
	    maxError = fabs(dabc(i));
	  }
	}
	
	if(maxError < epsilon)
	  break;
    }
    chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
    chrono::duration<double> my_time_used = chrono::duration_cast<chrono::duration<double>>( t4-t3 );
    cout<<"my_GN time cost = "<<my_time_used.count()<<" seconds. "<<endl;
    
    double finalError = computeError(abc,x_data,y_data);
    cout<<"my_finalError="<<finalError<<endl;
    cout<<"my_abc = ";
    for(auto a:abc)
    {
	cout<<a<<" ";
    }
    cout<<endl;
    cout<<"my_GN_end"<<endl;
    
    abc[0]=0;
    abc[1]=0;
    abc[2]=0;
    // 构建最小二乘问题
    ceres::Problem problem;
    for ( int i=0; i<N; i++ )
    {
      ceres::CostFunction* cost_fun = 
          new ceres::AutoDiffCostFunction<CURVE_FITTING_COST,1,3>
               (new CURVE_FITTING_COST(x_data[i],y_data[i]));
	       
      problem.AddResidualBlock(cost_fun,nullptr,abc);
    }

    // 配置求解器
    ceres::Solver::Options options;     
    // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_QR;  
    // 增量方程如何求解
    options.minimizer_progress_to_stdout = true;   
    // 输出到cout

    ceres::Solver::Summary summary;                
    // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve ( options, &problem, &summary );  
    // 开始优化
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;

    // 输出结果
    cout<<summary.BriefReport() <<endl;
    cout<<"estimated a,b,c = ";
    for ( auto a:abc ) cout<<a<<" ";
    cout<<endl;

    return 0;
}

