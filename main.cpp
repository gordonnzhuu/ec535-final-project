#include <opencv2/opencv.hpp>
using namespace cv;

int main(){
    // 1) Load
    Mat img = imread("tung.jpg");
    if(img.empty()) return -1;

    // 2) Pre‑process: gray + blur
    Mat gray, blurred;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(5,5), 1.5);

    // 3) Edge detection
    Mat edges;
    Canny(blurred, edges, 50, 150);

    // 4) Find contours
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 5) Draw contours onto a copy of the original
    Mat contourImg = img.clone();
    RNG rng(12345);
    for(size_t i = 0; i < contours.size(); i++){
        Scalar color(rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256));
        drawContours(contourImg, contours, static_cast<int>(i), color, 2, LINE_8, hierarchy, 0);
    }

    Mat contourMask = Mat::zeros(img.size(), CV_8UC3);
    RNG rng2(54321);
    for(size_t i = 0; i < contours.size(); i++){
        Scalar color(rng2.uniform(0,256), rng2.uniform(0,256), rng2.uniform(0,256));
        drawContours(contourMask, contours, static_cast<int>(i), color, 2, LINE_8, hierarchy, 0);
    }

    // 6) Show results in resizable windows
    namedWindow("Edges", WINDOW_NORMAL);
    resizeWindow("Edges", 800, 600);
    imshow("Edges", edges);

    namedWindow("Contours", WINDOW_NORMAL);
    resizeWindow("Contours", 800, 600);
    imshow("Contours", contourImg);

    namedWindow("Contour Mask", WINDOW_NORMAL);
    resizeWindow("Contour Mask", 800, 600);
    imshow("Contour Mask", contourMask);    

    waitKey(0);
    return 0;
}

