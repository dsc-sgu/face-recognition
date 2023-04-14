#include <opencv2/opencv.hpp>

class IsgetImageResult {
    public:
        bool isImage;
        cv::Mat image;
        IsgetImageResult(bool isImage, cv::Mat image) {
            this->isImage = isImage;
            this->image = image;
        }
};

IsgetImageResult isgetImage(cv::String source) {
    cv::Mat potentialImage = cv::imread(source);
    if (!potentialImage.empty()) {
        return IsgetImageResult(true, potentialImage);
    } else {
        return IsgetImageResult(false, potentialImage);
    }
}

void detectOnImage(cv::Mat image, cv::String output, int imageMaxWidth) {
    
    // ...

    if (output.length() > 0) {
        cv::imwrite(output, image);
    }

    if (image.size[0] > imageMaxWidth) {
        cv::resize(image, image, cv::Size(imageMaxWidth, image.size[1] - (image.size[0] - imageMaxWidth)));
    }
    cv::imshow("Raspli Detector", image);
    cv::waitKey();
}

int main(int argc, char *argv[]) {
    // ./main.o [<source> [<output>]]]

    cv::String source = "";
    if (argc >= 2) {
        source = argv[1];
    }
    cv::String output = "";
    if (argc >= 3) {
        output = argv[2];
    }

    IsgetImageResult isgetImageResult = isgetImage(source);
    if (isgetImageResult.isImage) {
        detectOnImage(isgetImageResult.image, output, 640);
    }

    return 0;
}