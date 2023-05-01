// Raspli (^-^), GNU AGPL-3.0 license

#ifndef DETECT
#define DETECT

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include "imgproc.h"

using namespace std;

class Box {
    public:
        unsigned short x1, y1, x2, y2;
        float conf;
        Box(
            unsigned short x1,
            unsigned short y1,
            unsigned short x2, 
            unsigned short y2,
            float conf
        ) {
            this->x1 = x1;
            this->y1 = y1;
            this->x2 = x2;
            this->y2 = y2;
            this->conf = conf;
        }
};

float iou(Box &fb, Box &sb) {
    float inter = max(min(fb.x2, sb.x2) - min(fb.x1, sb.x1), 0) * max(min(fb.y2, sb.y2) - min(fb.y1, sb.y1), 0);
    float union_ = (fb.x2-fb.x1)*(fb.y2-fb.y1) + (sb.x2-sb.x1)*(sb.y2-sb.y1) - inter;
    return inter / union_;
}

vector<Box> nms(vector<Box> &boxes, float iouThres) {
    vector<Box> supBoxes;
    for (Box box: boxes) {
        bool valid = true;
        for (Box supBox: supBoxes) {
            if (iou(box, supBox) > iouThres) {
                valid = false;
                break;
            }
        }
        if (valid == true) {
            supBoxes.push_back(box);
        }
    }
    return supBoxes;
}

vector<Box> getBoxes(
    at::Tensor &outputs,
    float confThres = 0.25,
    float iouThres = 0.15
) {
    vector<Box> candidates;
    for (unsigned short ibatch = 0; ibatch < outputs.sizes()[0]; ibatch++) {
        for (unsigned short ibox = 0; ibox < outputs.sizes()[2]; ibox++) {
            float conf = outputs[ibatch][4][ibox].item<float>();
            if (conf >= confThres) {
                unsigned short
                    cx = outputs[ibatch][0][ibox].item<int>(),
                    cy = outputs[ibatch][1][ibox].item<int>(),
                    w = outputs[ibatch][2][ibox].item<int>(),
                    h = outputs[ibatch][3][ibox].item<int>();
                unsigned short
                    x1 = cx - w / 2,
                    y1 = cy - h / 2,
                    x2 = cx + w / 2,
                    y2 = cy + h / 2;
                candidates.push_back(Box(x1,y1,x2,y2,conf));
            }
        }
    }
    sort(candidates.begin(), candidates.end(), [](Box b1, Box b2){return b1.conf > b2.conf;});
    vector<Box> boxes = nms(candidates, iouThres);
    return boxes;
}

void highlightBoxes(cv::Mat &img, vector<Box> &boxes) {
    
    cv::Scalar rectColor(0,192,0);
    unsigned short fontScale = 2, confPrecis = 2;

    for (Box box: boxes) {
        string text = to_string(box.conf);//.substr(0, text.find('.') + confPrecis + 1);
        cv::rectangle(img, {box.x1,box.y1}, {box.x2,box.y2}, rectColor, 2);
        cv::rectangle(
            img,
            {box.x1, box.y1 - fontScale * 12},
            {box.x1 + (unsigned short)text.length() * fontScale * 9, box.y1},
            rectColor,
            -1
        );
        cv::putText(img, text, {box.x1,box.y1}, cv::FONT_HERSHEY_PLAIN, fontScale, {255,255,255}, 2);
    }
}

#endif