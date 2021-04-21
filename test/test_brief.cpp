//
// Created by buenos on 2021/4/20.
//

#define OPENCV_TRAITS_ENABLE_DEPRECATED

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

void extractBrief(Mat imgLeft, Mat imgRgiht) {
    vector<KeyPoint> kpsLeft, kpsRight;
    cv::Mat descLeft, descRight;

    Ptr<GFTTDetector> gftt_ = cv::GFTTDetector::create(500, 0.01, 30);
    gftt_->detect(imgLeft, kpsLeft);
    gftt_->detect(imgRgiht, kpsRight);
    cout << "keyPoints num: " << kpsLeft.size() << endl;

    Ptr<xfeatures2d::BriefDescriptorExtractor> brief = xfeatures2d::BriefDescriptorExtractor::create(32, false);
    brief->compute(imgLeft, kpsLeft, descLeft);
    brief->compute(imgRgiht, kpsRight, descRight);
    cout << "keyPoints num: " << kpsLeft.size() << endl;        // 计算完描述子后会丢失一部分特征点。
    cout << "descriptors num: " << descLeft.size << endl;

    brief->compute(imgLeft, kpsLeft, descLeft);
    brief->compute(imgRgiht, kpsRight, descRight);
    cout << "keyPoints num: " << kpsLeft.size() << endl;        // 计算完描述子后会丢失一部分特征点。
    cout << "descriptors num: " << descLeft.size << endl;
    cout << "check consistence." << endl;


    vector<DMatch> matches;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    matcher ->match(descLeft, descRight, matches);

    Mat imgOut;
    drawMatches(imgLeft, kpsLeft, imgRgiht, kpsRight, matches, imgOut);

    imshow("brief", imgOut);
    waitKey(0);

}



int main(int argc, char **argv) {
    string basePath = "/home/buenos/buenos/data_odometry_gray/dataset/sequences/";

    Mat imgLeft, imgRight;
    if(argc != 2) {
        imgLeft = imread(basePath + "00/image_0/000000.png", CV_LOAD_IMAGE_GRAYSCALE);
        imgRight = imread(basePath + "00/image_1/000000.png", CV_LOAD_IMAGE_GRAYSCALE);
        printf("usage: OrbVsOptflow <img1> <img2>\n");
    } else {
        imgLeft = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
        imgRight = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    }

    assert(imgLeft.data != nullptr && imgRight.data != nullptr);

    extractBrief(imgLeft, imgRight);

}

