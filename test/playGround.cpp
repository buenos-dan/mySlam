//
// Created by buenos on 2021/4/21.
//
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

