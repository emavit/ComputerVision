//#include "function.h"
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


/***************************************************************
                                                CLEAR TERMINAL
    Funtion to clean the terminal. It works on all operating systems' terminals.
****************************************************************/
void clear_terminal() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

/***************************************************************
                                                USER CHOICES
    Function that allows the user to chose among two possible values. We 
    assume that the funtion returns true if the first options is selected, false
    if the secod one is selected.

    The second function return an int value inserted by the user (we want this 
    value to be bigger than a certain threshold)
****************************************************************/
bool userBinaryDecision (string title, string option1, string option2){
    cout << title  <<endl;
    cout << "1)" << option1  <<endl;
    cout << "2)" << option2  <<endl;
    cout << "\nChoice: ";

    // Initialize variables
    bool selected = true;
    bool decision;
    string choice;
    vector<Mat> patches;

    // Loop until a valid choice is selected
    while(selected == true) {
        cin >> choice;
        if(choice == "1"){
            cout << "\n\n";
            decision = true;
            selected = false;
        }else if(choice == "2"){
            cout << "\n\n";
            decision = false;
            selected = false;        
        }else{
            cout << "\034[0;33mInvalid Choice\033[0m"<<endl;
            //cout << "Invalid choice.\n";
            cout << "\nChoice: ";
        }
    }
    // Return the selected choice   
    //cout << "\n";
    return decision;

}

int askThreshold(string title, string subtitle, float threshold){

    string input;
    int userInput;
    bool valid = false;
    while(!valid){
        cout << title <<endl;
        cout << subtitle <<endl;
        //cout<<endl<<"\033[1;33mHow many Octave Layers do you want SIFT to use? [the higher the value the higher the computational time]\033[0m"<<endl;
        //cout<<"Octave Layers [int>0]: ";
        cin >> input;
        
        // Create a stringstream from the input
        stringstream ss(input);
        
        // Read the value from the stringstream
        ss >> input;
        
        // Check if the stringstream extraction was successful
        if (ss.fail()) {
            // Extraction failed
            cout<<"\033[0;31mInvalid Choice\033[0m"<<endl;
        }
        else{
            if(userInput < threshold){
                cout<<"\033[0;31mInvalid Choice\033[0m"<<endl;
            }
            //else{
            //   cout <<"\033[1;32mYour choice: " +to_string(userInput)+" was correctly registered\033[0m"<<endl<<endl;
            //    valid = true; 
            //}
            
        }
    }    
    // Return the integer value
    return userInput;
}


/**********************************************************
                                            ANALYZER 
    Creates a <Feature2D> object using SIFT or ORB 
**********************************************************/

Ptr<Feature2D> createFeatureDetector(bool type) {
    if (type) {
        int octavelayers= 5;
        int features = 0;
        return SIFT::create(features, octavelayers);
    } else 
        return ORB::create(10000000);
}


/***************************************************************
                                                LOAD IMAGES
    Function to load a certain group of images having the same prefix with only 
    final number changing. We are not supposed to know the number of images.
    Futhermore the routine is able to load images with different extensions.
****************************************************************/

vector<Mat>  loadImage( string dir, string patchName, string title = "", bool print = false) {
    // Initialize vector of images
    vector<Mat> images;
    // Possible extensions
    vector<string> exts = {".jpg", ".png", ".jpeg"};

 for (int i = 0; ; i++) {
        // Initialize patch file path
        string patchPath = "";

        // Loop over possible extensions
        for (string ext : exts) {
            // Construct patch file path with extension
            patchPath = dir + patchName + to_string(i) + ext;

            // Check if file exists with current extension
            if (ifstream(patchPath)) {
                break;
            } else {
                // Clear path if file not found
                patchPath = "";
            }
        }

        // Check if file exists
        if (patchPath == "") {
            break;
        }

        // Read image file
        Mat patch = imread(patchPath, IMREAD_COLOR);

        // Add image to the vector
        images.push_back(patch);
    }    

    // Display number of images read (only if required)
    if (print)
        cout << title << images.size() << endl;
    return images;
}

/***************************************************************
                                                SAVE IMAGES
    Function to save an image file inside a directory with a given name.
****************************************************************/

void saveImageToFolder(Mat image, string imagePath, string directory, string filename)
{ 
    // Create the directory if it doesn't exist
    filesystem::create_directory(imagePath + "/" + directory);
    
    // Construct the full path to save the image
    string savePath = imagePath + "/" + directory + "/" + filename + ".jpg";
    
    // Save the image to the specified directory
    imwrite(savePath, image);
}

/***************************************************************
                                            SIFT and ORB
    Function to return descriptors and keypoints of a given image in a tuple.
    The function takes as input a <Feature2D> object (namely SIFT or ORB),
    the image to analyze and a boolean indicating whether to draw the results.
****************************************************************/

tuple<Mat, vector<KeyPoint>> KeypointsFeatureExtractor (Ptr<Feature2D> recognizer, Mat image, bool draw, int flag){

    // Output variables
    Mat descriptors;
    vector<KeyPoint> keypoints;

    // Detect and compute keypoints and descriptors
    recognizer->detectAndCompute(image, noArray(), keypoints, descriptors);

    // DEBUG
    /*
        cout<<typeid(*recognizer).name()<<endl;

    string tp = typeid(*recognizer).name();

    if(tp == "N2cv9SIFT_ImplE")
        cout<<"sift"<<endl;
    */

    if (draw){
        Mat output;
        string title = "";
        if(flag == 1){
            title = "SIFT - Keypoints image";
        }else{
            title = "ORB - Keypoints image";
        }
        drawKeypoints(image, keypoints, output, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        namedWindow(title, WINDOW_AUTOSIZE);
        imshow(title, output);
        waitKey(0);
    }  
    return  make_tuple(descriptors,  keypoints); 

}

/***************************************************************
                                                MATCHES
    Function that receives as input two Mat objects containing the descriptors
    and a float containign the distance ratio to filter the matches. The function
    returns a vector of DMatch objects containing the matches.
****************************************************************/

vector<DMatch> matchRefine (Mat descriptorsImage, Mat descriptorsTemp, float distanceRatio, Personal_Flags flag){

    // Output vector
    vector<DMatch> matches;

    // Define the matcher. Notice that is case of ORB we use Hamming distance, in case of SIFT we use L2
    BFMatcher matcher(NORM_L2);

    if(int(flag)==1){
        BFMatcher matcher(NORM_L2);
    }
    else if (int(flag)==2)
    {
         BFMatcher matcher(NORM_HAMMING);
    }
    else{
        throw runtime_error("No Norm Selected");
    }
    
    // Auxiliary vector to store the matches
    vector<DMatch> unfilteredMatches;
    vector<vector<DMatch>> normMatches;

    // Find matches between the patch and the descriptors passed as input
    // Find 2 matches orderd by distance (for each descriptor)

    matcher.knnMatch(descriptorsTemp, descriptorsImage, normMatches, 2);
    
    // Filter the matches basing on distance ratio
    // Consider only matches with distance <= distanceRatio*secondDistance
    for (size_t j = 0; j < normMatches.size(); ++j) {
        if (normMatches[j][0].distance < distanceRatio * normMatches[j][1].distance) {
            unfilteredMatches.push_back(normMatches[j][0]); 
        }            
    }

    // Filter the matches basing on distance
    // Find minumin distance
    float minDistance = FLT_MAX;
    for (size_t j = 0; j < unfilteredMatches.size(); ++j) {
        if (unfilteredMatches[j].distance < minDistance) {
            minDistance = unfilteredMatches[j].distance;
        }
    }

    // Consider only matches with distance <= 3*minDistance
    for (size_t j = 0; j < unfilteredMatches.size(); ++j) {
        if (unfilteredMatches[j].distance <= 3 * minDistance) {
            matches.push_back(unfilteredMatches[j]);
        }
    }

    // DEBUG ONLY
    /*
     cout << "After filtering based on distanceRatio:" << unfilteredMatches.size() << endl;
    cout << "After filtering based on minDistance: " << matches.size() << endl; 
    */
      
    return matches;
}



/***************************************************************
                                            FLIPPING
    Function that takes as input a <Feature2D> object (namely SIFT or ORB), 
    a patch and a vector of descriptors that is used to find the best orientation
    for the patch (best means  the maximum number of matches with 
    desciptorsImage). The function returns the patch in its best orientation.
****************************************************************/

Mat flippingMatcher (Ptr<Feature2D> sift, Mat patch, Mat descriptorsImage,  float distanceRatio){
    
    // Variable to consider the best number of matches find between
    int numMatchesBest = -1;

    // Vector to store the output image
    Mat imageOut;

    // Loop over the 4 possible flipping of the patch, including the original one
    // Because of the 'flip' option, we need to start from -1
    for (int i=-1; i<=2; i++){

        // Auxiliary variables
        vector<KeyPoint> kpTemp; 
        Mat descTemp;
        vector<DMatch> matches;
        Mat flippedPatch;

        // Flip the image. The first 3 iterations are done with different flipping, the fourth one with the original image
        if (i<=1){
            flip(patch, flippedPatch, i);
        }
        else{
            flippedPatch = patch.clone();
        }

        // Detect and compute keypoints and descriptors of the flipped patch
        tie (descTemp, kpTemp) = KeypointsFeatureExtractor(sift, flippedPatch, false,1);
        // Find matches between the patch and the descriptors passed as input
        matches = matchRefine(descriptorsImage, descTemp, distanceRatio, Personal_Flags::SIFT);

        // Update the best patch at each iteration
        if(int(matches.size())>numMatchesBest){
            numMatchesBest = int(matches.size());
            imageOut = flippedPatch.clone();
        }
    }
    return imageOut;
}




/***************************************************************
                                            HOMOGRAPHY
    In this section we define the functions to find the homography matrix.
    We implemented two different functions: one that uses the library functions
    and one that manually implements the RANSAC algorithm (BONUS).
****************************************************************/

// Implementation of the homography function using the library functions of OpenCV
// In both cases the function returns a Mat object in the format CV_32F to be compatible with WarpPerspective
Mat homograpySTD (vector<KeyPoint> kpImage, vector<KeyPoint> kpTemp, vector<DMatch> matches){

    // Define the homography matrix as a 3x3 null matrix
    Mat H = cv::Mat::zeros(3, 3, CV_32F);

    // Find the corresponding points in the image and in the kpTemp using the matches vector
    vector<Point2f> patchPoints, imagePoints;
    for (int i = 0; i < matches.size(); ++i) {
        imagePoints.push_back(kpImage[matches[i].trainIdx].pt);
        patchPoints.push_back(kpTemp[matches[i].queryIdx].pt);
    }

    //DEBUG
    //cout << "Patch points" << patchPoints.size() << endl;
    //cout << "Image points" << imagePoints.size() << endl;

    // Set parameters of findHomography. Nootice that maxIters is set to 2000 to avoid infinite loop
    double ransacReprojThreshold = 3.0;
    double ransacConfidence = 0.99;
    int maxIters = 2000;

    // Find the homography matrix using the RANSAC algorithm implemented in OpenCV
    H = findHomography(patchPoints, imagePoints, RANSAC, ransacReprojThreshold, noArray(), maxIters, ransacConfidence); 

    // Convert the homography matrix to the correct output format
    H.convertTo(H, CV_32F);

    return H;
}


// Manual implementation of the RANSAC algorithm to find the homography matrix
Mat homograpyMAN (vector<KeyPoint> kpImage, vector<KeyPoint> kpTemp, vector<DMatch> matches){

    // Define the affine matrix as an empty matrix with the correct output format
    Mat affineMatrix(2, 3, CV_32F, Scalar(0));

    // Find the corresponding points in the image and in the kpTemp using the matches vector
    vector<Point2f> patchPoints, imagePoints;
    for (int i = 0; i < matches.size(); ++i) {
        imagePoints.push_back(kpImage[matches[i].trainIdx].pt);
        patchPoints.push_back(kpTemp[matches[i].queryIdx].pt);
    }

    // Set parameters for the RANSAC algorithm
    int maxIters = 100;
    int distThreshold = 3;

    // In order to find the best matrix for homography, we implement the following iterative algorithm
    // Initialization
    int bestCorrespondencies = 0;    

    //  Iterate to find the best matrix for homography
    for (int i = 0; i < maxIters ; i++) {
        
        vector<Point2f> patchSubset(3);
        vector<Point2f> imageSubset(3);

        // Select 3 random points inside the patch and the image at each iteration
        // The homography matrix will be based on these points
        for (int j = 0; j < 3; j++) {
            int idx = rand()%(patchPoints.size() - 1);
            patchSubset[j] = patchPoints[idx];
            imageSubset[j] = imagePoints[idx];
        }

        // Determines the values of the matrix A and b as seen during lectures
        Mat A(6, 6, CV_32F, Scalar(0));
        Mat b(6, 1, CV_32F, Scalar(0));
        for (int j=0; j<3; j++){
            int idx = 2*j;
            A.at<float>(idx, 0) = patchSubset[j].x;
            A.at<float>(idx, 1) = patchSubset[j].y;
            A.at<float>(idx, 4) = 1;
            A.at<float>(idx+1, 2) = patchSubset[j].x;
            A.at<float>(idx+1, 3) = patchSubset[j].y;
            A.at<float>(idx+1, 5) = 1;
            b.at<float>(idx) = imageSubset[j].x;
            b.at<float>(idx+1) = imageSubset[j].y;
        }

        // Soolve the system Ax = b using the least squares
        Mat affineLS;
        solve(A, b, affineLS, DECOMP_SVD);

        // Correctly sotore LS solution (1x6) inside an affine temporary matrix (2x3)
        Mat affineTemp(2, 3, CV_32F, Scalar(0));
        affineTemp.at<float>(0,0) = affineLS.at<float>(0);
        affineTemp.at<float>(0,1) = affineLS.at<float>(1);
        affineTemp.at<float>(0,2) = affineLS.at<float>(4);

        affineTemp.at<float>(1,0) = affineLS.at<float>(2);
        affineTemp.at<float>(1,1) = affineLS.at<float>(3);
        affineTemp.at<float>(1,2) = affineLS.at<float>(5);
       
        // Find the number of correspondencies we can achieve through the affine transformation we just found
        int correspondencies = 0;
        for (int j = 0; j < patchPoints.size(); j++) {

            // Project the point using the affine transformation
            Point2f projectedPoint;
            Mat sourcePoint = (Mat_<float>(3, 1) << patchPoints[j].x, patchPoints[j].y, 1.0);         
            Mat projectedTemp = affineTemp * sourcePoint;
            projectedPoint.x = projectedTemp.at<float>(0, 0);
            projectedPoint.y = projectedTemp.at<float>(1, 0);

            // Evaluate the distance between the projected point and the original point using L2 norm 
            double distance = norm(projectedPoint - imagePoints[j]);

            // If the distance is smaller than the threshold, we consider the point as a valide correspondency
            if (distance < distThreshold) {
                correspondencies++;
            }   
        }

        // If the number of correspondencies is better than the previous best, we update the best correspondencies value and the affine matrix
        if (correspondencies > bestCorrespondencies) {
            bestCorrespondencies = correspondencies;
            affineMatrix = affineTemp;
        }
    }

    return affineMatrix;

}

/***************************************************************
                                            MERGE AND FILTER
    Function that receives the image and the patch as input and merges them
****************************************************************/

Mat mergeNoShape (Mat image, Mat patch){

    // Create a mask, containing only non-black points of the patch
    Mat mask = patch > 0;

    // Convert the mask to grayscale
    Mat maskTemp;
    mask.convertTo(maskTemp, CV_32FC3, 1.0 / 255.0);

    // Create a smaller mask
    Mat maskReduced;
    int shapingDim = 2; 
    Mat element = getStructuringElement(MORPH_RECT, Size(2 * shapingDim + 1, 2 * shapingDim + 1), Point(shapingDim, shapingDim));
    // Remove possible noise in the mask and erodes it
    morphologyEx(maskTemp, maskReduced, MORPH_CLOSE,  element);
    // Remove the black edges that results from making the mask rectangular and smaller than the original patch
    morphologyEx(maskReduced, maskReduced, MORPH_RECT,  element);

    // Check the correct mask type CV_32FC3
    if (maskReduced.type() != CV_32FC3) {
        maskReduced.convertTo(maskReduced, CV_32FC3);
    }

    // Resize the mask to the image size
    resize(maskReduced, maskReduced, image.size(), 0, 0, INTER_CUBIC);

    // Convert the mask to grayscale
    cvtColor(maskReduced, maskReduced, COLOR_BGR2GRAY);

    // Split the image and the patch into their channels
    vector<Mat> imageChannels, patchChannels;
    split(image, imageChannels);
    split(patch, patchChannels);

    // Perform alpha blending applying the feathered mask to each channel, then merge them all togheter
    vector<Mat> blendedImageChannels(3);

    for (int j = 0; j < 3; ++j) {

        // Convert the channels to desired types
        Mat patchTempCh, imageTempCh;
        patchChannels[j].convertTo(patchTempCh, CV_32FC1);
        imageChannels[j].convertTo(imageTempCh, CV_32FC1);

        // Apply the mask
        blendedImageChannels[j] = imageTempCh.mul(1.0 - maskReduced) + patchTempCh.mul(maskReduced);
        blendedImageChannels[j].convertTo(blendedImageChannels[j], CV_8UC1);
    }

    // Merge the channels back into the final image
    // Noitce that we are modifying the original image
    merge(blendedImageChannels, image);

    return image;
}


/***************************************************************
                                        MULTIPLE IMAGES
    Function that receives multiple images and multiples patches as input. 
    Morevoer, it receives the number of octaves, the threshold for the number of 
    matches and the distance ratio. With these parameters, it creates a SIFT
    obejct that, toghter with some filtering,  can be used to assoiciate each patch
    with its corresponding image. This function return a 2D vector of patches.
    By reading the output veoctr, we can find for each row all the patches 
    assiciated with the corresponding image (ordered as the input). 
****************************************************************/

vector<vector<Mat>> multiplePatchMatcher (vector<Mat> images, vector<Mat>patches, int octavelayers, int matchesThreshold, float distanceRatio){
    // Auxiliary varibale to store patches. We use a 2D vector in order to mainatin the order of the images
    vector<vector<Mat>> outPatches;

    // Create SIFT object 
    Ptr<SIFT> sift = SIFT::create(0, octavelayers);

    // This for cycle iterates over all images passed as input. For each image it tries to 
    // find the corresponding patches among all the ones passed as input.
    for(int i=0; i<images.size(); i++){

        cout<<"\nWorking on source image "+to_string(i) + "..."<<endl;

        // Compute keypoints and descriptors
        vector<KeyPoint> kpImage;
        Mat descriptorsImage;
        tie (descriptorsImage, kpImage) = KeypointsFeatureExtractor(sift, images[i], false,1);

        // Vecotor containing all the patches associated with the image we are considering
        vector<Mat> macthingPatches;

        // Iterate over all the patches
        for (int j = 0; j < patches.size(); j++)
        {
            cout<<"Comparing with patch "+to_string(j) <<endl;
            vector<KeyPoint> kpTemp; 
            Mat descTemp;
            vector<DMatch> matches;

            // Find best flipping versoion of the patch
            patches[j] = flippingMatcher (sift, patches[j], descriptorsImage, distanceRatio);

            // Detect and compute SIFT features of the pach        
            tie (descTemp, kpTemp) = KeypointsFeatureExtractor(sift, patches[j], false,1);

            // Evaluate the matches between the image and the patch
            matches = matchRefine(descriptorsImage, descTemp, distanceRatio, Personal_Flags::SIFT);

            // Check the sparsity of the matches. This parameter is really important. In fact, by tuning the sparsity we can deal with
            // really different corrupted images. Tune it for good results!!!

            int sparseMatchCount = 0;
            for (const auto& match : matches) {
                if (match.distance > matchesThreshold) {                    
                    sparseMatchCount++;
                }
                //cout<<"match:"+to_string(match.distance)<<endl;
            }
            double sparsityRatio = static_cast<double>(sparseMatchCount) / static_cast<double>(matches.size());    
            //cout<<"sparsity: "+to_string(sparsityRatio)<<endl;

            // Add the match to the vector of patches associated with the image if the sparsity is low enough
            if(sparsityRatio<0.2){
                macthingPatches.push_back(patches[j]);
            }
        }
        // Add the patches to the corresponding image
        outPatches.push_back(macthingPatches);
    }

    return outPatches;
}

/***************************************************************
                                    HIGHLIGHT DIFFERENCES
    Function that receives two images as input and returns a binary image that 
    shows the difference between them.
****************************************************************/

Mat showImageDifferences(Mat image1, Mat image2)
{

    // Check if the images were loaded successfully
    if (image1.empty() || image2.empty()) {
        std::cerr << "Failed to load images." << std::endl;
    }

    // Check if the images have the same size
    if (image1.size() != image2.size()) {
        std::cerr << "Images have different sizes." << std::endl;
    }

    // Compute the absolute difference between the images
    Mat diffImage;
    absdiff(image1, image2, diffImage);

    // Convert the difference image to grayscale
    cvtColor(diffImage, diffImage, cv::COLOR_BGR2GRAY);

    // Threshold the difference image to highlight the changes
    threshold(diffImage, diffImage, 30, 255, cv::THRESH_BINARY);


    imshow("Differences", diffImage);

    return diffImage;
}

string getLastFolder(const std::string& path)
{
    filesystem::path fsPath(path);

    // Check if the path exists
    if (!filesystem::exists(fsPath)) {
        cerr << "Path does not exist." <<endl;
        return "";
    }

    // Check if the path is a directory
    if (!filesystem::is_directory(fsPath)) {
        cerr << "Path is not a directory." << endl;
        return "";
    }

    // Get the last folder from the path
    string lastFolder = fsPath.filename().string();

    return lastFolder;
}