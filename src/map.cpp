/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "myslam/map.h"
#include "myslam/feature.h"
#include "myslam/config.h"

namespace myslam {

    Map::Map(){
        matcher_ = cv::DescriptorMatcher::create("BruteForce-Hamming");
        loop_flag_ = Config::Get<int>("loop_flag");
        if(loop_flag_){
            LOG(INFO) << "loading voc ...";
            voc_ = new DBoW3::Vocabulary(Config::Get<std::string>("voc_file"));
            db_.setVocabulary(*voc_, false, 0);
            assert(!voc_->empty());
            LOG(INFO) << "load voc finished!";
        }
    }

void Map::InsertKeyFrame(Frame::Ptr frame) {
    current_frame_ = frame;

    if(loop_flag_) DetectLoopAndCorrectMappoint(frame);

    if (keyframes_.find(frame->keyframe_id_) == keyframes_.end()) {
        keyframes_.insert(make_pair(frame->keyframe_id_, frame));
        active_keyframes_.insert(make_pair(frame->keyframe_id_, frame));
    } else {
        // almost never jump into here...
        keyframes_[frame->keyframe_id_] = frame;
        active_keyframes_[frame->keyframe_id_] = frame;
    }

    if (active_keyframes_.size() > num_active_keyframes_) {
        RemoveOldKeyframe();
    }
}

void Map::InsertMapPoint(MapPoint::Ptr map_point) {
    if (landmarks_.find(map_point->id_) == landmarks_.end()) {
        landmarks_.insert(make_pair(map_point->id_, map_point));
        active_landmarks_.insert(make_pair(map_point->id_, map_point));
    } else {
        landmarks_[map_point->id_] = map_point;
        active_landmarks_[map_point->id_] = map_point;
    }
}

void Map::RemoveOldKeyframe() {
    if (current_frame_ == nullptr) return;
    // 寻找与当前帧最近与最远的两个关键帧
    double max_dis = 0, min_dis = 9999;
    double max_kf_id = 0, min_kf_id = 0;
    auto Twc = current_frame_->Pose().inverse();
    for (auto& kf : active_keyframes_) {
        if (kf.second == current_frame_) continue;
        auto dis = (kf.second->Pose() * Twc).log().norm();
        if (dis > max_dis) {
            max_dis = dis;
            max_kf_id = kf.first;
        }
        if (dis < min_dis) {
            min_dis = dis;
            min_kf_id = kf.first;
        }
    }

    const double min_dis_th = 0.2;  // 最近阈值
    Frame::Ptr frame_to_remove = nullptr;
    if (min_dis < min_dis_th) {
        // 如果存在很近的帧，优先删掉最近的
        frame_to_remove = keyframes_.at(min_kf_id);
    } else {
        // 删掉最远的
        frame_to_remove = keyframes_.at(max_kf_id);
    }

    LOG(INFO) << "remove keyframe " << frame_to_remove->keyframe_id_;
    // remove keyframe and landmark observation
    active_keyframes_.erase(frame_to_remove->keyframe_id_);
    for (auto feat : frame_to_remove->features_left_) {
        auto mp = feat->map_point_.lock();
        if (mp) {
            mp->RemoveObservation(feat);
        }
    }
    for (auto feat : frame_to_remove->features_right_) {
        if (feat == nullptr) continue;
        auto mp = feat->map_point_.lock();
        if (mp) {
            mp->RemoveObservation(feat);
        }
    }

    CleanMap();
}

void Map::CleanMap() {
    int cnt_landmark_removed = 0;
    for (auto iter = active_landmarks_.begin();
         iter != active_landmarks_.end();) {
        if (iter->second->observed_times_ == 0) {
            iter = active_landmarks_.erase(iter);
            cnt_landmark_removed++;
        } else {
            ++iter;
        }
    }
    LOG(INFO) << "Removed " << cnt_landmark_removed << " active landmarks";
}

    void Map::DetectLoopAndCorrectMappoint(Frame::Ptr frame){
        if(frame->keyframe_id_ < last_loop_index_ + 50){
            db_.add(frame->descriptors_left_);
            return;
        }
        DBoW3::QueryResults ret;
        db_.query(frame->descriptors_left_, ret, 4, frame->keyframe_id_ - 50);
        db_.add(frame->descriptors_left_);


        if(!ret.empty() && ret[0].Score > 0.05){
            unsigned long loopIndex = LONG_MAX;
            for(int i = 1; i < ret.size(); i++){
                if(ret[i].Score > 0.015 && ret[i].Id < loopIndex) loopIndex = ret[i].Id;
            }
            if(loopIndex == LONG_MAX) return;

            { // debug
//                for(int i =0; i < ret.size();i++) LOG(INFO) << "loop score: " << ret[i].Score;
//                for (int i = 0; i < ret.size(); i++) {
//                    LOG(INFO) << "detect loop, current frame: " << frame->keyframe_id_ << " loop frame: " << ret[i].Id;
//                    cv::imshow("loop" + i, keyframes_.at(ret[i].Id)->left_img_);
//                }
//                cv::imshow("cur img", frame->left_img_);
//                cv::waitKey(0);
            }

            Frame::Ptr loopFrame = keyframes_[loopIndex];
            cv::Mat descCur = frame->descriptors_left_, descLoop = loopFrame->descriptors_left_;
            std::vector<cv::DMatch> matches;
            matcher_ ->match(descLoop, descCur, matches);

            // correct map_points.
            for(cv::DMatch match: matches){
                Feature::Ptr loopFeat = loopFrame->features_left_[match.queryIdx];
                if(!loopFeat->map_point_.expired()){
                    frame->features_left_[match.trainIdx]->map_point_ = loopFeat ->map_point_;
                }
            }
            // add into queue to BA.
            {
                std::unique_lock<std::mutex> lck(loop_mutex_);
                loopQueue_.push({loopIndex, frame->keyframe_id_});
            }

            last_loop_index_ = frame->keyframe_id_;
            LOG(INFO) << "detect loop, current frame: " << frame->keyframe_id_ << " loop frame: " << loopIndex;

        }

    }

    Map::KeyframesType Map::GetRegionKeyFrames(long startFrameID, long endFrameID){
        std::unique_lock<std::mutex> lck(data_mutex_);
        region_keyframes_.clear();
        for(unsigned long i = startFrameID; i <= endFrameID; i++){
            auto item = keyframes_.find(i);
            if(item != keyframes_.end()) region_keyframes_.insert(*item);
            else throw "invalid region, loop BA failed!";
        }
        return region_keyframes_;
    }

    Map::LandmarksType Map::GetRegionMapPoints(long startFrameID, long endFrameID){
        std::unique_lock<std::mutex> lck(data_mutex_);
        region_landmarks_.clear();
        for(int i = startFrameID; i <= endFrameID; i++){
            auto kf = keyframes_.find(i);
            if(kf != keyframes_.end()){
               for(auto& feat: kf->second->features_left_){
                   if(!feat->map_point_.expired()){
                       auto mp = feat -> map_point_.lock();
                       region_landmarks_.insert({mp->id_, mp});
                   }
               }
            }
        }
        return region_landmarks_;
    }

}  // namespace myslam
