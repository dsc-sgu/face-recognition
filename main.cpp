// Raspli (^-^), GNU AGPL-3.0 license

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include "imgproc.h"
#include "detect.h"

using namespace std;

cv::Mat detect(
    torch::jit::script::Module &model,
    cv::Mat img,
    int imgMaxWidth
) {
    if (img.size[1] > imgMaxWidth)
        cv::resize(img, img, {imgMaxWidth, imgMaxWidth * img.size[0] / img.size[1]});
    
    cv::Mat imgCov = coverImg(img, {640,640});

    cv::Mat imgNorm; imgCov.copyTo(imgNorm);
    cv::cvtColor(imgNorm, imgNorm, cv::COLOR_BGR2RGB);
    cv::normalize(imgNorm, imgNorm, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
    vector<torch::jit::IValue> inputs = {
        torch::from_blob(
            imgNorm.data,
            {640,640,3},
            torch::kFloat32
        ).permute({2,0,1}).unsqueeze(0)
    };
    at::Tensor outputs = model.forward(inputs).toTensor();
    vector<Box> boxes = getBoxes(outputs);
    
    highlightBoxes(imgCov, boxes);
    return imgCov;
}

void detectOnImg(
    torch::jit::script::Module &model,
    cv::String &source,
    cv::String &output,
    int imgMaxWidth
) {
    cv::Mat img = cv::imread(source);
    img = detect(model, img, imgMaxWidth); // detection highlight
    
    // result save
    if (cv::haveImageWriter(output))
        cv::imwrite(output, img);

    cv::imshow("Raspli Detector", img);
    cv::waitKey();
}

void detectOnVid(
    torch::jit::script::Module &model,
    cv::String &source,
    cv::String &output,
    cv::String &vidcodec,
    int imgMaxWidth
) {
    cv::VideoCapture cap = (source != "0") ? cv::VideoCapture(source) : cv::VideoCapture(0);

    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::Size frameSize = {
        (int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
        (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT)
    };

    int rec_fourcc;
    cv::VideoWriter rec;
    if (output.length() > 0) {
        rec_fourcc = cv::VideoWriter::fourcc(vidcodec[0], vidcodec[1], vidcodec[2], vidcodec[3]);
        rec.open(output, rec_fourcc, fps, frameSize);
    }

    cv::Mat frame;
    while (cap.isOpened()) {
        cap >> frame;
        frame = detect(model, frame, imgMaxWidth); // detection highlight
        if (frame.empty() || cv::waitKey(1) == 'c')
            break;
        if (rec.isOpened())
            rec.write(frame);
        cv::imshow("Raspli Detector", frame);
    }

    cap.release();
    if (rec.isOpened())
        rec.release();
    cv::destroyAllWindows();
}

int main(int argc, char *argv[]) {
    // ./main.o <model> <source> [<output> [<vidcodec>]]
    // if <source> equals "0" -> will be used to detect on webcam video

    cv::String argmodel = "";
    if (argc >= 2)
        argmodel = argv[1];
    cv::String argsource = "";
    if (argc >= 3)
        argsource = argv[2];
    cv::String argoutput = "";
    if (argc >= 4)
        argoutput = argv[3];
    cv::String argvidcodec = "";
    if (argc >= 5)
        argvidcodec = argv[4];

    torch::jit::script::Module model = torch::jit::load(argmodel);

    if (cv::haveImageReader(argsource))
        detectOnImg(model, argsource, argoutput, 640);
    else
        detectOnVid(model, argsource, argoutput, argvidcodec, 640);

    return 0;
}