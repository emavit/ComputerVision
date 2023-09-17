
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"

#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <map>
#include <thread>
#include <chrono>
using namespace std;
using namespace cv;

enum class Personal_Flags {
    SIFT = 1,
    ORB = 2,
    OTHER = 4,
    //FLAG4 = 8
};

vector<Mat>  loadImage(String, String, String);

void clear_terminal();

