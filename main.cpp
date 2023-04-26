#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include "checkget.h"
#include "imgproc.h"
#include "detect.h"

using namespace std;

cv::Mat detect(
    torch::jit::script::Module &model,
    cv::Mat img,
    int imgMaxWidth
) {

    if (img.size[1] > imgMaxWidth) {
        cv::resize(img, img, {imgMaxWidth, img.size[0] - (img.size[1] - imgMaxWidth)});
    }

    CoverImgMeta coverImgMeta = coverImg(img, {640,640});
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::normalize(img, img, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
    vector<torch::jit::IValue> inputs = {
        torch::from_blob(
            img.data,
            {640,640,3},
            torch::kFloat32
        ).permute({2,0,1}).unsqueeze(0)
    };
    at::Tensor outputs = model.forward(inputs).toTensor();
    vector<Box> boxes = getBoxes(outputs, coverImgMeta);
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    hightBoxes(img, boxes);
    
    return img;
}

void detectOnImg(
    torch::jit::script::Module &model,
    cv::Mat &img,
    cv::String &output,
    int imgMaxWidth
) {
    cv::Mat imgHighted = detect(model, img, imgMaxWidth);
    if (output.length() > 0) {
        cv::imwrite(output, imgHighted);
    }
    cv::imshow("Raspli Detector", imgHighted);
    cv::waitKey();
}

int main(int argc, char *argv[]) {
    // ./main.o <model> <source> [<output>]

    cv::String argmodel = "";
    if (argc >= 2) {
        argmodel = argv[1];
    }
    cv::String argsource = "";
    if (argc >= 3) {
        argsource = argv[2];
    }
    cv::String argoutput = "";
    if (argc >= 4) {
        argoutput = argv[3];
    }

    torch::jit::script::Module model = torch::jit::load(argmodel);

    IsgetImgResult isgetImgResult = isgetImg(argsource);
    if (isgetImgResult.isImg) {
        detectOnImg(model, isgetImgResult.img, argoutput, 640);
    } else {
        // detectOnVideo(model, argoutput, 640);
    }

    return 0;
}