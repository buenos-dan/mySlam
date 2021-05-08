#pragma once
#ifndef MYSLAM_CONFIG_H
#define MYSLAM_CONFIG_H

#include "myslam/common_include.h"

namespace myslam {

/**
 * 配置类，使用SetParameterFile确定配置文件
 * 然后用GetParam得到对应值
 * 单例模式
 */
class Config {
   private:
    static std::shared_ptr<Config> config_;
    cv::FileStorage param_file_;
    cv::FileStorage path_file_;

    Config() {}  // private constructor makes a singleton
   public:
    ~Config();  // close the file when deconstructing

    // set a new config file
    static bool SetParameterFile(const std::string &filename);
    static bool SetPathFile(const std::string &filename);

    // access the parameter values
    template <typename T>
    static T GetParam(const std::string &key) {
        return T(Config::config_->param_file_[key]);
    }

    template <typename T>
    static T GetPath(const std::string &key) {
        return T(Config::config_->path_file_[key]);
    }
};
}  // namespace myslam

#endif  // MYSLAM_CONFIG_H
