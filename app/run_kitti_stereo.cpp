//
// Created by gaoxiang on 19-5-4.
//

#include <gflags/gflags.h>
#include "myslam/visual_odometry.h"

DEFINE_string(param_config_file, "./config/default.yaml", "parameter config file path");
DEFINE_string(path_config_file, "./config/path.yaml", "dataset, voc, gt path config file path");

int main(int argc, char **argv) {
    FLAGS_log_dir = "./logs";
    google::InitGoogleLogging(argv[0]);

    google::ParseCommandLineFlags(&argc, &argv, true);

    myslam::VisualOdometry::Ptr vo(
        new myslam::VisualOdometry(FLAGS_param_config_file, FLAGS_path_config_file));
    assert(vo->Init() == true);
    vo->Run();

    return 0;
}
