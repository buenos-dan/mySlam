#include "myslam/config.h"

namespace myslam {
bool Config::SetParameterFile(const std::string &filename) {
    if (config_ == nullptr)
        config_ = std::shared_ptr<Config>(new Config);
    config_->param_file_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);
    if (config_->param_file_.isOpened() == false) {
        LOG(ERROR) << "parameter file " << filename << " does not exist.";
        config_->param_file_.release();
        return false;
    }
    return true;
}

bool Config::SetPathFile(const std::string &filename) {
    if (config_ == nullptr)
        config_ = std::shared_ptr<Config>(new Config);
    config_->path_file_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);
    if (config_->path_file_.isOpened() == false) {
        LOG(ERROR) << "path file " << filename << " does not exist.";
        config_->path_file_.release();
        return false;
    }
    return true;
}

Config::~Config() {
    if (param_file_.isOpened())
        param_file_.release();
}

std::shared_ptr<Config> Config::config_ = nullptr;

}
