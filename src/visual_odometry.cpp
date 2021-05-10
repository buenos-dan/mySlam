//
// Created by gaoxiang on 19-5-4.
//
#include "myslam/visual_odometry.h"
#include <chrono>
#include "myslam/config.h"
#include <fstream>
#include <iomanip>


namespace myslam {

VisualOdometry::VisualOdometry(std::string &config_path)
    : config_file_path_(config_path) {}

bool VisualOdometry::Init() {
    // read from config file
    if (Config::SetParameterFile(config_file_path_) == false) {
        return false;
    }

    dataset_ = Dataset::Ptr(new Dataset(
                Config::Get<std::string>("dataset_dir"),
                Config::Get<int>("max_image_index")));
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


    show_viewer_ = Config::Get<int>("show_viewer");
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

    // save trajectory
    bool save_trajectory = Config::Get<int>("save_trajectory");
    if(save_trajectory){
        std::string trajectoryFile = Config::Get<std::string>("output_trajectory_file");
        SaveTrajectory(trajectoryFile);
    }


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

void VisualOdometry::SaveTrajectory(const std::string &filename){
    std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;


    for(int i = 0; i < map_->framePoses.size(); i++)
    {
        Eigen::Matrix<double, 4, 4> pose = map_->framePoses.at(i).inverse().matrix();
        f << std::setprecision(9) << pose(0,0) << " " << pose(0,1)  << " " << pose(0,2) << " " << pose(0, 3)
                              << " " << pose(1,0) << " " << pose(1,1)  << " " << pose(1,2) << " "  << pose(1,3)
                              << " " << pose(2,0) << " " << pose(2,1)  << " " << pose(2,2) << " "  << pose(2,3)
                              << std::endl;
    }
    f.close();
    LOG(INFO) << "trajectory saved!";
}

}  // namespace myslam
