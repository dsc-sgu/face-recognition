#include <opencv2/opencv.hpp>

#ifndef ISGET_IMG_RESULT
#define ISGET_IMG_RESULT
class IsgetImgResult {
    public:
        bool isImg;
        cv::Mat img;
        IsgetImgResult(bool isImg, cv::Mat img) {
            this->isImg = isImg;
            this->img = img;
        }
};
#endif

#ifndef ISGET_IMG
#define ISGET_IMG
IsgetImgResult isgetImg(cv::String source) {
    cv::Mat potentialImg = cv::imread(source);
    if (!potentialImg.empty()) {
        return IsgetImgResult(true, potentialImg);
    } else {
        return IsgetImgResult(false, potentialImg);
    }
}
#endif