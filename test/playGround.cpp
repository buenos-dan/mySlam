//
// Created by buenos on 2021/4/21.
//

// evo_ape kitti /home/buenos/buenos/data_odometry_gray/groundtruth/poses/00.txt output.txt -va --plot --plot_mode xz

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main(){
    cv::Mat a;// = cv::Mat::ones(4,3, CV_64F);
    cv::Mat b = cv::Mat::ones(4,3, CV_64F);
    a.push_back(b);
    cout << a << endl;
    return 0;
}

