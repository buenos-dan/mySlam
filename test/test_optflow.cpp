//
// Created by buenos on 2021/4/20.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
using namespace cv;
using namespace std;

void stereoMatchOptflow(Mat imgLeft, Mat imgRight) {
    vector<KeyPoint> kpsLeft, kpsRight;
    vector<Point2f> ptsLeft, ptsRight;
    Mat descsLeft;

    vector<uchar> status;
    Mat error;

    Ptr<GFTTDetector> gftt_ = cv::GFTTDetector::create(1000, 0.01, 20);
    gftt_->detect(imgLeft, kpsLeft);

    for(auto &kp : kpsLeft) {
        ptsLeft.push_back(kp.pt);
    }

    cv::calcOpticalFlowPyrLK(
            imgLeft, imgRight, ptsLeft, ptsRight,
            status, error, cv::Size(11, 11), 3);
    for(auto &kpPt : ptsRight) {
        KeyPoint curKp;
        curKp.pt = kpPt;
        kpsRight.push_back(curKp);
    }

    vector<DMatch> matches;
    for(int i = 0; i < status.size(); i++) {
        if(status[i]) {
            DMatch curMatch;
            curMatch.queryIdx = i;
            curMatch.trainIdx = i;
            matches.push_back(curMatch);
        }
    }
    printf("optflow: %d matches\n", matches.size());

    Mat imgOut;
    drawMatches(imgLeft, kpsLeft, imgRight, kpsRight, matches, imgOut);
    imshow("optflow", imgOut);
    cvWaitKey(0);
}

int main(int argc, char **argv) {
    string basePath = "/home/buenos/buenos/data_odometry_gray/dataset/sequences/";

    Mat imgLeft, imgRight;
    if(argc != 2) {
        imgLeft = imread(basePath + "00/image_0/000000.png", CV_LOAD_IMAGE_GRAYSCALE);
        imgRight = imread(basePath + "00/image_1/000000.png", CV_LOAD_IMAGE_GRAYSCALE);
        cout << "usage: test_optflow <img1> <img2>\n" << endl;
    } else {
        imgLeft = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
        imgRight = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    }

    assert(imgLeft.data != nullptr && imgRight.data != nullptr);

    stereoMatchOptflow(imgLeft, imgRight);
}