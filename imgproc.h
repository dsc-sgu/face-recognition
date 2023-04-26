#include <opencv2/opencv.hpp>

#ifndef COVER_IMG_META
#define COVER_IMG_META
class CoverImgMeta {
    public:
        short horizGap, vertGap;
        CoverImgMeta(
            short horizGap,
            short vertGap
        ) {
            this->horizGap = horizGap;
            this->vertGap = vertGap;
        }
};
#endif

#ifndef COVER_IMG
#define COVER_IMG
CoverImgMeta coverImg(cv::Mat &img, cv::Size trgSize) {
    cv::Mat imgCov; img.copyTo(imgCov);
    float horizScale = imgCov.size[1] / imgCov.size[0], vertScale = imgCov.size[0] / imgCov.size[1];
    float horizGap = trgSize.width - imgCov.size[1];
    if (horizGap != imgCov.size[1]) {
        cv::resize(imgCov, imgCov,
            {imgCov.size[1] + (short)(horizGap * horizScale), imgCov.size[0] + (short)(horizGap * vertScale)}
        );
        horizGap = 0;
    }
    float vertGap = trgSize.height - imgCov.size[0];
    if (vertGap < 0) {
        cv::resize(imgCov, imgCov,
            {imgCov.size[1] + (short)(vertGap * horizScale), imgCov.size[0] + (short)(vertGap * vertScale)}
        );
        horizGap += std::abs(vertGap) * horizScale;
        vertGap = 0;
    }
    cv::Mat canvas = cv::Mat::zeros(trgSize, CV_8UC3);
    cv::Mat canvasROI = canvas(cv::Rect(horizGap / 2, vertGap / 2, imgCov.size[1], imgCov.size[0]));
    imgCov.copyTo(canvasROI);
    canvas.copyTo(img);
    return CoverImgMeta(horizGap, vertGap);
}
#endif