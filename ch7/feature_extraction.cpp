#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main ( int argc, char** argv )
{
    if ( argc != 3 )
    {
        cout<<"usage: feature_extraction img1 img2"<<endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    //-- 初始化
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // Ptr<FeatureDetector> detector = FeatureDetector::create(detector_name);
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create(descriptor_name);
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    Mat outimg1;
    drawKeypoints( img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imshow("ORB",outimg1);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> matches;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, matches );
    
    cout<<matches[0].queryIdx<<endl;
    cout<<matches[0].trainIdx<<endl;
    cout<<matches[0].imgIdx<<endl;
    cout<<matches[0].distance<<endl;
    
    cout<<matches[1].queryIdx<<endl;
    cout<<matches[1].trainIdx<<endl;
    cout<<matches[1].imgIdx<<endl;
    cout<<matches[1].distance<<endl;
    
    cout<<matches[200].queryIdx<<endl;//样本图像的第200个描述子的下标，还是200
    cout<<matches[200].trainIdx<<endl;//第200个匹配的，匹配图像的描述子下标
    cout<<matches[200].imgIdx<<endl;//当样本为多张图像时有用，为样本图像的下标
    
    cout<<matches[200].distance<<endl;
    
    cout<<"1:"<<keypoints_1.size()<<endl;
    cout<<keypoints_1[0].pt<<endl;
    cout<<keypoints_1[0].size<<endl;
    cout<<keypoints_1[0].angle<<endl;
    
    cout<<keypoints_1[1].pt<<endl;
    cout<<keypoints_1[1].size<<endl;
    cout<<keypoints_1[1].angle<<endl;
    
    cout<<keypoints_1[2].pt<<endl;
    cout<<keypoints_1[2].size<<endl;
    //该点的直径大小
    cout<<keypoints_1[2].angle<<endl;
    cout<<"2:"<<keypoints_2.size()<<endl;

    cout<<"match:"<<matches.size()<<endl;
    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }
    
    // 仅供娱乐的写法
    min_dist = min_element( matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;
    max_dist = max_element( matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector< DMatch > good_matches;
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            good_matches.push_back ( matches[i] );
        }
    }
    cout<<"good:"<<good_matches.size()<<endl;
    cout<<descriptors_1.size()<<endl;//500个描述子，每个都是32位2进制数
    cout<<descriptors_1.rows<<endl;

    //-- 第五步:绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, matches, img_match );
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch );
    imshow ( "all_matches", img_match );
    imshow ( "good_matches", img_goodmatch );
    waitKey(0);

    return 0;
}
