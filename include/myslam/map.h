#pragma once
#ifndef MAP_H
#define MAP_H

#include <opencv2/xfeatures2d.hpp>
#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"
#include "DBoW3/DBoW3.h"

namespace myslam {

/**
 * @brief 地图
 * 和地图的交互：前端调用InsertKeyframe和InsertMapPoint插入新帧和地图点，后端维护地图的结构，判定outlier/剔除等等
 */
class Map {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Map> Ptr;
    typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;
    typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;

    std::mutex loop_mutex_;
    std::queue<std::pair<unsigned long, unsigned long>> loopQueue_;

    std::unordered_map<unsigned long, SE3> framePoses;     // just for calculate rmse

public:
    Map();

    /// 保存帧位姿
    void setFramePose(unsigned long id, SE3 twc){
        if(framePoses.count(id) > 0) framePoses.at(id) = twc;
        else framePoses.insert({id, twc});
    }

    /// 增加一个关键帧
    void InsertKeyFrame(Frame::Ptr frame);
    /// 增加一个地图顶点
    void InsertMapPoint(MapPoint::Ptr map_point);

    /// 获取所有地图点
    LandmarksType GetAllMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return landmarks_;
    }
    /// 获取所有关键帧
    KeyframesType GetAllKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return keyframes_;
    }

    /// 获取激活地图点
    LandmarksType GetActiveMapPoints() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_landmarks_;
    }

    /// 获取激活关键帧
    KeyframesType GetActiveKeyFrames() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_keyframes_;
    }

    /// 获取指定区间的关键帧
    KeyframesType GetRegionKeyFrames(long startFrameID, long endFrameID);

    /// 获取指定区间的地图点
    LandmarksType GetRegionMapPoints(long startFrameID, long endFrameID);

   private:
    // 将旧的关键帧置为不活跃状态
    void RemoveOldKeyframe();

    // ...
    void DetectLoopAndCorrectMappoint(Frame::Ptr frame);

    // 计算KeyFrame的特征点的描述子
    cv::Mat ExtractKFDescriptors();

    std::mutex data_mutex_;
    LandmarksType landmarks_;         // all landmarks
    LandmarksType active_landmarks_;  // active landmarks
    KeyframesType keyframes_;         // all key-frames
    KeyframesType active_keyframes_;  // active key-frames
    LandmarksType region_landmarks_;  // region landmarks
    KeyframesType region_keyframes_;         // region key-frames

    Frame::Ptr current_frame_ = nullptr;

    // settings
    int num_active_keyframes_ = 7;  // 激活的关键帧数量
    bool loop_flag_ = false; // 是否启用回环检测
    unsigned long last_loop_index_ = 0; // 上次检测到回环的keyframeId

    // DBoW2
    DBoW3::Database db_;
    DBoW3::Vocabulary* voc_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
};
}  // namespace myslam

#endif  // MAP_H
