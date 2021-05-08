//
// Created by gaoxiang on 19-5-4.
//
#include "myslam/visual_odometry.h"
#include <chrono>
#include "myslam/config.h"
#include <fstream>


namespace myslam {

VisualOdometry::VisualOdometry(std::string &param_config, std::string &path_config)
    : param_config_path_(param_config), path_config_path_(path_config) {}

bool VisualOdometry::Init() {
    // read from config file
    if (Config::SetParameterFile(param_config_path_) == false || Config::SetPathFile(path_config_path_) == false) {
        return false;
    }

    dataset_ = Dataset::Ptr(new Dataset(
                Config::GetPath<std::string>("dataset_dir"),
                Config::GetParam<int>("max_image_index")));
    CHECK_EQ(dataset_->Init(), true);

    // create components and links
    frontend_ = Frontend::Ptr(new Frontend);
    backend_ = Backend::Ptr(new Backend);
    map_ = Map::Ptr(new Map);

    frontend_->SetBackend(backend_);
    frontend_->SetMap(map_);
    frontend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));

    backend_->SetMap(map_);
    backend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));


    show_viewer_ = Config::GetParam<int>("show_viewer");
    if(show_viewer_){
        viewer_ = Viewer::Ptr(new Viewer);
        frontend_->SetViewer(viewer_);
        viewer_->SetMap(map_);
    }

    return true;
}

void VisualOdometry::Run() {
    while (1) {
        if (Step() == false) {
            break;
        }
    }

    backend_->Stop();
    viewer_->Close();

    // cal RMSE
    LOG(INFO) << "RMSE: " << CalSeqError();

    LOG(INFO) << "VO exit";
}

bool VisualOdometry::Step() {
    Frame::Ptr new_frame = dataset_->NextFrame();
    if (new_frame == nullptr) return false;

    auto t1 = std::chrono::steady_clock::now();
    bool success = frontend_->AddFrame(new_frame);
    auto t2 = std::chrono::steady_clock::now();
    auto time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    LOG(INFO) << "VO cost time: " << time_used.count() << " seconds.";
    return success;
}

double VisualOdometry::CalSeqError(){
    double rmse = 0;

    Viewer::TrajectoryType gtPoses = loadPoses(Config::GetPath<std::string>("ground_truth_file"));
    if(gtPoses.empty()){
        LOG(ERROR) << "load ground truth err, CalSeqError exit.";
        return rmse;
    }

    Map::KeyframesType kfs = map_->GetAllKeyFrames();
    for(auto& kf: kfs){
        SE3 gtPoseInv = gtPoses.at(kf.second->id_);
        SE3 pose = kf.second->Pose();
        double error = (gtPoseInv * pose).log().norm();
        rmse += error * error;
    }

    rmse = rmse / double(kfs.size());
    rmse = sqrt(rmse);
    return rmse;
}

Viewer::TrajectoryType VisualOdometry::loadPoses(const std::string file_name) {
    Viewer::TrajectoryType gtPoses;
    std::ifstream fin(file_name);
    if(!fin) return gtPoses;

    unsigned long cnt = 0;
    while(!fin.eof()) {
        float a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34;
        fin >> a11 >> a12 >> a13 >> a14 >> a21 >> a22 >> a23 >> a24
            >> a31 >> a32 >> a33 >> a34;
        Sophus::Matrix4f p;
        p << a11, a12, a13, a14, a21, a22, a23, a24, a31, a32, a33, a34, 0, 0 ,0, 1;
        Sophus::SE3f pose_gt(p);
        gtPoses.insert({cnt++, pose_gt.cast<double>()});
    }
    fin.close();
    return gtPoses;
}

}  // namespace myslam
