#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <limits>
#include <vector>
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

void clear_terminal();

bool userBinaryDecision (string, string, string);

int askThreshold(string, string, int);

Ptr<Feature2D> createFeatureDetector(bool, int, int);

vector<Mat>  loadImage( string, string, string, bool);

void saveImageToFolder(Mat, string, string, string);

tuple<Mat, vector<KeyPoint>> KeypointsFeatureExtractor (Ptr<Feature2D>, Mat, bool, int);

vector<DMatch> matchRefine (Mat, Mat, float, Personal_Flags, int);

Mat flippingMatcher (Ptr<Feature2D>, Mat, Mat,  float, int);

Mat homograpySTD (vector<KeyPoint>, vector<KeyPoint>, vector<DMatch>);

Mat homograpyMAN (vector<KeyPoint>, vector<KeyPoint>, vector<DMatch>);

Mat mergeNoShape (Mat, Mat);

vector<vector<Mat>> multiplePatchMatcher (vector<Mat>, vector<Mat>, int, int, float, int);

Mat showImageDifferences(Mat, Mat);

string getLastFolder(const std::string&);

Mat templateMatching(Mat, Mat);

