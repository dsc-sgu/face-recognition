#ifndef IMGPROC
#define IMGPROC

#include <opencv2/opencv.hpp>

cv::Mat coverImg(cv::Mat &img, cv::Size trgSize) {
    cv::Mat imgROI; img.copyTo(imgROI);
    if (imgROI.size[1] > trgSize.width)
        cv::resize(imgROI, imgROI,
            {trgSize.width, trgSize.height * imgROI.size[0] / imgROI.size[1]}
        );
    if (imgROI.size[0] > trgSize.height)
        cv::resize(imgROI, imgROI,
            {trgSize.width * imgROI.size[1] / imgROI.size[0], trgSize.height}
        );
    cv::Mat canvas = cv::Mat::zeros(trgSize, CV_8UC3);
    cv::Mat canvasROI = canvas(
        cv::Rect(
            (trgSize.width - imgROI.size[1]) / 2, (trgSize.height - imgROI.size[0]) / 2,
            imgROI.size[1], imgROI.size[0]
        )
    );
    imgROI.copyTo(canvasROI);
    return canvas;
}

#endif