#include <opencv2/opencv.hpp>
#include <torch/script.h>

using namespace std;

class IsgetImageResult {
    public:
        bool isImage;
        cv::Mat image;
        IsgetImageResult(bool isImage, cv::Mat image) {
            this->isImage = isImage;
            this->image = image;
        }
};

class Box {
    public:
        int rx1, ry1, rx2, ry2;
        float conf;
        Box(int rx1, int ry1, int rx2, int ry2, float conf) {
            this->rx1 = rx1;
            this->ry1 = ry1;
            this->rx2 = rx2;
            this->ry2 = ry2;
            this->conf = conf;
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

float iou(Box fb, Box sb) {
    float inter = max(min(fb.rx2, sb.rx2)-min(fb.rx1, sb.rx1), 0) * max(min(fb.ry2, sb.ry2)-min(fb.ry1, sb.ry1), 0);
    float union_ = (fb.rx2-fb.rx1)*(fb.ry2-fb.ry1) + (sb.rx2-sb.rx1)*(sb.ry2-sb.ry1) - inter;
    return inter / union_;
}

vector<Box> nms(vector<Box> boxes, float iouThres) {
    vector<Box> boxesSup;
    for (Box box: boxes) {
        bool valid = true;
        for (Box boxSup: boxesSup) {
            if (iou(box, boxSup) > iouThres) {
                valid = false;
                break;
            }
        }
        if (valid == true) {
            boxesSup.push_back(box);
        }
    }
    return boxesSup;
}

vector<Box> getBoxes(
    torch::jit::script::Module model,
    cv::Mat image,
    float confThres = 0.25,
    float iouThres = 0.15
) {
    cv::resize(image, image, {640,640});
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX, CV_32F);

    vector<torch::jit::IValue> inputs;
    at::Tensor imageTensor = torch::from_blob(
        image.data,
        {640,640,3},
        torch::kFloat32
    ).permute({2,0,1}).unsqueeze(0);
    inputs.push_back(imageTensor);
    
    at::Tensor outputs = model.forward(inputs).toTensor();

    vector<Box> candidates;
    for (int ibatch = 0; ibatch < outputs.sizes()[0]; ibatch++) {
        for (int ibox = 0; ibox < outputs.sizes()[2]; ibox++) {
            float conf = outputs[ibatch][4][ibox].item<float>();
            if (conf >= confThres) {
                int rcx = outputs[ibatch][0][ibox].item<int>(),
                    rcy = outputs[ibatch][1][ibox].item<int>(),
                     rw = outputs[ibatch][2][ibox].item<int>(),
                     rh = outputs[ibatch][3][ibox].item<int>();
                
                int rx1 = rcx - rw / 2,
                    ry1 = rcy - rh / 2,
                    rx2 = rcx + rw / 2,
                    ry2 = rcy + rh / 2;

                candidates.push_back(Box(rx1, ry1, rx2, ry2, conf));
            }
        }
    }

    sort(candidates.begin(), candidates.end(), [](Box b1, Box b2){return b1.conf > b2.conf;});
    vector<Box> boxes = nms(candidates, iouThres);

    return boxes;
}

cv::Mat highlightBoxes(cv::Mat image, vector<Box> boxes) {
    for (Box box: boxes) {
        cv::rectangle(image, {box.rx1, box.ry1}, {box.rx2, box.ry2}, cv::Scalar(0,192,0), 2);
        string conf_str = to_string(box.conf);
        conf_str = conf_str.substr(0, conf_str.find('.') + 2 + 1);
        int fontScale = 2;
        cv::rectangle(
            image,
            {box.rx1, box.ry1 - fontScale * 12},
            {box.rx1 + conf_str.length() * fontScale * 9, box.ry1},
            cv::Scalar(0,192,0),
            -1
        );
        cv::putText(image, conf_str, {box.rx1, box.ry1}, cv::FONT_HERSHEY_PLAIN, fontScale, cv::Scalar(255,255,255), 2);
    }
    return image;
}

void detectOnImage(
    torch::jit::script::Module model,
    cv::Mat image,
    cv::String output,
    int imageMaxWidth
) {
    if (image.size[0] > imageMaxWidth) {
        cv::resize(image, image, cv::Size(imageMaxWidth, image.size[1] - (image.size[0] - imageMaxWidth)));
    }

    vector<Box> boxes = getBoxes(model, image);
    image = highlightBoxes(image, boxes);

    if (output.length() > 0) {
        cv::imwrite(output, image);
    }
    cv::imshow("Raspli Detector", image);
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

    torch::jit::script::Module model;
    model = torch::jit::load(argmodel);

    IsgetImageResult isgetImageResult = isgetImage(argsource);
    if (isgetImageResult.isImage) {
        detectOnImage(model, isgetImageResult.image, argoutput, 640);
    }

    return 0;
}