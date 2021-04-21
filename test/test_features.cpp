//
// Created by buenos on 2021/4/21.
//

/* result
 * gftt: 368 12.35ms 9.86ms
 * shi-tomasi: 500 10.67ms
 * fast: 2027 0.63ms
 * optflow: 368->337 1.28ms
 * brief: 200->162 0.96ms
 * match: 162 0.22ms
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <chrono>

using namespace cv;
using namespace std;

/* ******************************
 * aux functions
 * *******************************/

// ret = v1 - v2;
vector<KeyPoint> intersection(vector<KeyPoint>& v1, vector<KeyPoint>& v2){
    vector<KeyPoint> result;
    for(int i = 0; i < v1.size(); i++){
        int flag = 0;
        for(int j =0; j < v2.size(); j++){
            if(v1[i].pt.x == v2[j].pt.x && v1[i].pt.y == v2[j].pt.y ) flag = 1;
        }
        if(flag == 0) result.push_back(v1[i]);
    }
    return result;
}


/* ******************************
 * test functions
 * *******************************/
void detectShiTomasi(Mat imgLeft, Mat imgRgiht){
    vector<KeyPoint> kpsLeft;
    vector<Point2f> ptsLeft;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    goodFeaturesToTrack(imgLeft, ptsLeft, 500, 0.01, 10, Mat());
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    cout << "shi-tomasi num: " << ptsLeft.size() << endl;
    cout << "time: " << elapsed_seconds.count() * 1000 << " ms" << endl;
    for(int i = 0; i < (int)ptsLeft.size(); i++)
    {
        cv::KeyPoint key;
        key.pt = ptsLeft[i];
        kpsLeft.push_back(key);
    }

    drawKeypoints(imgLeft, kpsLeft, imgLeft);

    imshow("shi-tomasi", imgLeft);
    waitKey(0);
}

void detectFast(Mat imgLeft, Mat imgRgiht){
    vector<KeyPoint> kpsLeft;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    FAST(imgLeft, kpsLeft, 40, true);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    cout << "fast num: " << kpsLeft.size() << endl;
    cout << "time: " << elapsed_seconds.count() * 1000 << " ms" << endl;

    drawKeypoints(imgLeft, kpsLeft, imgLeft);

    imshow("fast", imgLeft);
    waitKey(0);
}

void extractBrief(Mat imgLeft, Mat imgRgiht) {
    vector<KeyPoint> kpsLeft, kpsRight;
    Mat descLeft, descRight;

    // set mask
    int boarderSize = 48 + 4;
    cout << "imgLeft size: " << imgLeft.size() << endl;
    Mat mask = Mat::zeros(imgLeft.size(), CV_8UC1);
    rectangle(mask, {boarderSize,boarderSize}, {imgLeft.cols - boarderSize, imgLeft.rows - boarderSize}, 255, CV_FILLED);



    Ptr<GFTTDetector> gftt_ = cv::GFTTDetector::create(500, 0.01, 30);
    gftt_->detect(imgLeft, kpsLeft, mask);
    gftt_->detect(imgRgiht, kpsRight, mask);
    cout << "keyPoints num: " << kpsLeft.size() << endl;

    vector<KeyPoint> kpsTmp = kpsLeft;

    Ptr<xfeatures2d::BriefDescriptorExtractor> brief = xfeatures2d::BriefDescriptorExtractor::create(32, false);
    {
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        {
            brief->compute(imgLeft, kpsLeft, descLeft);
        }
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        cout << "brief num: " << kpsLeft.size() << endl;
        cout << "time: " << elapsed_seconds.count() * 1000 << " ms" << endl;
    }


    Mat test;
    vector<KeyPoint> inter = intersection(kpsTmp, kpsLeft);
    drawKeypoints(imgLeft, inter, test, {255,0,0});
    for(int i = 0; i < inter.size(); i++) cout << inter[i].pt << " "; cout << endl;
    cout << kpsTmp.size() << endl;
    cout << kpsLeft.size() << endl;
    cout << inter.size() << endl;
    cout << endl;
    imshow("test", test);
//    imshow("after", after);
    waitKey(0);

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
    {
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        {
            matcher ->match(descLeft, descRight, matches);
        }
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        cout << "match num: " << descLeft.rows << endl;
        cout << "time: " << elapsed_seconds.count() * 1000 << " ms" << endl;
    }


    Mat imgOut;
    drawMatches(imgLeft, kpsLeft, imgRgiht, kpsRight, matches, imgOut);

    imshow("brief", imgOut);
    waitKey(0);

}

void stereoMatchOptflow(Mat imgLeft, Mat imgRight) {
    vector<KeyPoint> kpsLeft, kpsRight;
    vector<Point2f> ptsLeft, ptsRight;
    Mat descsLeft;

    vector<uchar> status;
    Mat error;

    Ptr<GFTTDetector> gftt_ = cv::GFTTDetector::create(1000, 0.01, 20);

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    {
        gftt_->detect(imgLeft, kpsLeft);
    }
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    cout << "gftt num: " << kpsLeft.size() << endl;
    cout << "time: " << elapsed_seconds.count() * 1000 << " ms" << endl;


    for(auto &kp : kpsLeft) {
        ptsLeft.push_back(kp.pt);
    }

    {
        std::chrono::time_point<std::chrono::system_clock> start, end;
        start = std::chrono::system_clock::now();
        {
            cv::calcOpticalFlowPyrLK(
                    imgLeft, imgRight, ptsLeft, ptsRight,
                    status, error, cv::Size(11, 11), 3);
        }
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        cout << "optflow num: " << ptsLeft.size() << endl;
        cout << "time: " << elapsed_seconds.count() * 1000 << " ms" << endl;
    }


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
        cout << "usage: test_features <img1> <img2>\n" << endl;
    } else {
        imgLeft = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
        imgRight = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    }

    assert(imgLeft.data != nullptr && imgRight.data != nullptr);

    extractBrief(imgLeft, imgRight);
}
