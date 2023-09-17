 /**********************************************************
                            COMPUTER VISION - FINAL PROJECT 
                                    Academic Year 2022/2023
                                    Emanuele Vitetta 2082149
***********************************************************/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>
#include <fstream>
#include <random>
#include <limits>
#include <vector>
#include <fstream>
#include <map>
#include <thread>
#include <chrono>

//#include "function.h"
#include "function_user.cpp"

using namespace std;
using namespace cv;



int main(int argc, char** argv)
{

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    // Clear terminal
    clear_terminal();

    // Check arguments and evemtually print correct usage
    if (argc != 2) {
        cout << "Usage: ./main <image path> " << endl;
        cout << "The image has to be in the same folder of the patches" << endl;
        return -1;
    }

     String path = argv[1];

    /**********************************************************
                                        LOAD  PATCHES 
        We have to decide first if we want to use only one image or multiple 
        images (optional part). This can be decided by the user via terminal. 
        First of all we load all the possible patches.
    **********************************************************/

    // Load the patches
    vector<Mat> patches_base = loadImage(path, "/patch_", "Base patches: ",false);
    vector<Mat> patches_affine = loadImage(path, "/patch_t_", "Affine Tranformed patches: ",false);
    vector<Mat> patches;

    /**********************************************************
                                            USER CHOICE
        In this code section the user can decide the setup of the program.
    **********************************************************/
    cout << "------------------------------------------ IMAGE FIXER ------------------------------------------\n"<<endl;
    cout << "This program is able to reconstruct a corrupted image using a set of patches. The patches are" << endl;
    cout << "extracted from the original image. You have to decide first what technique you want to apply.\n" << endl;

    // Decide what technique to use to reconstruct images 
    bool reconstructorType = userBinaryDecision("What tecnique do you want to use to reconstruct the image?", "Feature extraction", "Template matching");

    // Use template matching
    if(!reconstructorType){
        cout << "-------------------------------------- TEMPLATE MATCHING -------------------------------------\n"<<endl;
        cout << "You are going to reconstruct the image using template matching. This part only work with" << endl;
        cout << "base patches. Moreover you are going to see intermediate results.\n" << endl;

        String fd = getLastFolder(path);
        Mat realImage = imread(path +"/" + fd + ".jpg");
        Mat sourceImage = imread(path + "/image_to_complete.jpg");
        namedWindow("Corrupted image", WINDOW_NORMAL);
        imshow("Corrupted image", sourceImage);

        // This section only works with base patches
        patches = patches_base;

        // Image on which patches wil be applied
        Mat sourceCopy = sourceImage.clone();

        // Iterate template matching for all patches
        for(int i=0; i<patches.size(); i++){

            // Overwrite the image with the one with the overlapped patch
            sourceCopy = templateMatching(sourceCopy, patches[i]);

            // Save results
            string title = "Template Matching - Image blenede with patch up to " + to_string(i);
            saveImageToFolder(sourceCopy, path, "results", title);
            cout<<title<<endl;      

            // Show results
            namedWindow(title, WINDOW_AUTOSIZE);
            imshow(title, sourceCopy);
            waitKey(0);

            // Show the difference with the original image when arriving at the last patch
            if(i == patches.size() -1 )
            {
      
                cout << "\nShowing the differences between the blended image and the real one...\n";

                // Show differences
                Mat diff = showImageDifferences(sourceCopy, realImage);
                string sub = "Template Matching - Differences";
                saveImageToFolder(diff, path, "results", sub);
                waitKey(0);
            }            
        } 

        cout<<"\n";

        // Exit the program 
        return -1;
    }

    cout << "-------------------------------------- FEATURE EXTRACTION --------------------------------------\n"<<endl;
    cout << "You are required to make some choices before starting the program. Notice that you can fix" << endl;
    cout << "one or multiple images, deciding if you want to use SIFT or ORB (with parametres choice)\n\n"<<endl;

    // Work with single corrupted image or with multiple courrupted images (BONUUS)
    bool mixedSource = userBinaryDecision("Do you want to use single or muliple corrupted images?", "Single [easy]", "Multiple [hard]");

    // Decide if to print intermediate results (useful for debugging)
    bool drawResults = userBinaryDecision("Do you want to print intermediate results?", "Yes", "No");

    // Patch selection
    bool pactSelect = userBinaryDecision("Select which set of patches you want to use:", "Base patches [easy]", "Affine patches [hard]");

    if(pactSelect == true){
        patches = patches_base;
    }else{
        patches = patches_affine;
    }
    int num_pathces = patches.size();

    // Decide if to use SIFT or ORB (BONUS)
    int flag;
    int features = 0;
    int octavelayers = 0;
    int minDis = 3;
    int minMatchRANSAC = 3;
    bool imageAnalyzer = userBinaryDecision("Do you want to use SIFT or ORB for marching?", "SIFT", "ORB");
    bool imageAnalyzerParameters = userBinaryDecision("Do you want to use manual or automatic parameters", "Automatic", "Manual");

    if(imageAnalyzerParameters){
        // Default values
        if(imageAnalyzer){
            flag = 1;
            features = 0;
            octavelayers = 5;
         }else{
            flag = 2;
            features = 100000;
        }   
    }else{
        // Manual values
        if(imageAnalyzer){
            flag = 1;
            features = askThreshold("How many features do you want to extract? [0 = all possible features are extracted]", "Features: ", -1);
            octavelayers = askThreshold("How many octave layers do you want to use?", "Octave layers: ", -1);
            minDis = askThreshold("What threshold you want to use to refine mathches? [integrer value, usally set to 3]", "Threshold: ", 1);
            minMatchRANSAC  = askThreshold("How many mathches do you want to have at least to apply RANSAC? [minum required is 3, higher values guarantee greater robustness]", "Minimum number of matches: ", 3);
         }else{
            flag = 2;
            features = askThreshold("How many features do you want to extract? [chose a great value]", "Features: ", -1);
            minDis = askThreshold("What threshold you want to use to refine mathches? [integrer value, usally set to 3]", "Threshold: ", 1);
            minMatchRANSAC = askThreshold("How many mathches do you want to have at least to apply RANSAC? [minum required is 3, higher values guarantee greater robustness]", "Minimum number of matches: ", 3);
        }   
    }   

   

    cout << "\n\033[1;32mAnalysis parameters\033[0m\nNumber of features = " << features << "\nOctave layers = " << octavelayers;
    cout << " [Only for SIFT]\nRefine threshold = "<< minDis << "\nNumber of matches for RANSAC = "<< minMatchRANSAC<< "\n" << endl;
    Ptr<Feature2D> recognizer = createFeatureDetector(imageAnalyzer, octavelayers, features);

    // Decide if use automatic or manual RANSAC (BONUS)
    bool homography = userBinaryDecision("Do you want to use manual or automatic RANSAC?", "Automatic", "Manual");


    cout << "Loaded " << num_pathces << " patches." << endl;
    cout << "----------------------------------------------------\n\n" << endl;
   

    // Single image
    if(mixedSource)
    {

        //vector<Mat> src = loadImage(path, "/image_to_complete", "Corrupted image: ",true);
        //Mat sourceImage = src[0];
        Mat sourceImage = imread(path + "/image_to_complete.jpg");
        namedWindow("Corrupted image", WINDOW_NORMAL);
        imshow("Corrupted image", sourceImage);
            
        
        /**********************************************************
                                                    SIFT and ORB
            For each image (original and patches) evaluate the SIFT/ORB descriptors 
            and keypoints. Then evaluate the matches between the original image
            and each patch. Then refine matches using a suitable threshold.
        **********************************************************/       

        // Detect and compute features of the corrupted image
        vector<KeyPoint> keypointsImage;
        Mat descriptorsImage;
        tie (descriptorsImage, keypointsImage) = KeypointsFeatureExtractor(recognizer, sourceImage, drawResults, flag);

        if (drawResults) {
            Mat output;
            string title = "";
            if(flag == 1){
                title = "SIFT - Keypoints image";
            }else{
                title = "ORB - Keypoints image";
            }
            drawKeypoints(sourceImage, keypointsImage, output, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            saveImageToFolder(output, path, "results", title);
        }  

        //vectors containg KP descriptors and Matches
        vector<vector<KeyPoint>> keypointsPatches;
        vector<Mat> descriptorsPatches;
        vector<vector<DMatch>> matches;

        // Distance ratio to refine good matches
        const float distanceRatio = 0.8f;
        
        // This for cycle iterates through all the patches. For each of them performs features and descriptors extraction
        // The algorithm also replaces the patche with the corresponding best flipped version (fixes affine transformation)
        // The finds best matches between the image and the patch
        if(drawResults){
            cout << "Extracting features and descriptors from patches...\n";
            cout << "\n\n\033[1;32mMATCHES\033[0m" << endl;
        }

        for (int i = 0; i < patches.size(); i++) {
            
            // Create auxiliary vectors
            vector<KeyPoint> kpTemp; 
            Mat descTemp;

            // Replace the original patch with the best flipped version. 
            patches[i] = flippingMatcher (recognizer, patches[i], descriptorsImage, distanceRatio, minDis);

            // Detect and compute features of the patch (using 'recognizer')   
            tie (descTemp, kpTemp) = KeypointsFeatureExtractor(recognizer, patches[i], drawResults, flag);

            // Add extracted values to the corresponding vectors
            keypointsPatches.push_back(kpTemp);
            descriptorsPatches.push_back(descTemp); 
            
            // Match descriptors of the image and the patch (pay attention to the last FLAG - determine what kind of error to use)
            if(flag == 1){
                matches.push_back(matchRefine(descriptorsImage, descriptorsPatches[i], distanceRatio, Personal_Flags::SIFT, minDis)); 
            }else{
                matches.push_back(matchRefine(descriptorsImage, descriptorsPatches[i], distanceRatio, Personal_Flags::ORB, minDis));
            }
            
            
            // Show matching results
            if(drawResults){
                Mat usefulMatches;

                string title;
                if(imageAnalyzer){
                    title  = "SIFT - Good Matches " + to_string(i);
                }else{
                    title  = "ORB - Good Matches " + to_string(i);
                }  

                drawMatches(patches[i], keypointsPatches[i], sourceImage, keypointsImage, matches[i], usefulMatches);
                namedWindow(title, WINDOW_NORMAL);
                imshow(title, usefulMatches);
                cout << "Patch " << i << " has " << matches[i].size() << " good matches." << endl;
                saveImageToFolder(usefulMatches, path, "results", title);

                waitKey(0); 
            }
        }

        /**********************************************************
                                                        RANSAC 
            Compute the homography matrix using RANSAC algorithm, which
            is implemented both using OpenCV routines and manually. Use this
            matrix to reconstruct the image by blending patches.
        **********************************************************/

        //create a copy of the image to attache the patches
        Mat sourceCopy = sourceImage.clone();

        // This for cycle iterates through all the patches. For each of them it finds the transformation to merge the patch with the corrupted image
        for (int i = 0; i < patches.size(); i++) {  

            if(i == 0)
                cout << "\n\033[1;32mMERGE\033[0m" << endl;

            // Check if we have enough matches to find the transformation
            if(matches[i].size()>=minMatchRANSAC){
                // Find the transformation
                Mat warpedPatch;
                if  (homography) {
                    // Automatic homography
                    Mat H =homograpySTD (keypointsImage, keypointsPatches[i], matches[i]);
                    // Compute the warped patch
                    warpPerspective(patches[i], warpedPatch, H, sourceImage.size());     
                    cout << "Merging with patch " << to_string(i) << endl;        
                }
                else {
                    // Manual homography
                    Mat A = homograpyMAN (keypointsImage, keypointsPatches[i], matches[i]);                
                    // Compute the warped patch   
                    warpAffine(patches[i], warpedPatch, A, sourceImage.size());
                    cout << "Merging with patch " << to_string(i) << "."<< endl; 
                }

                // Merge the patch with the corrupted image
                sourceCopy = mergeNoShape (sourceCopy, warpedPatch);
            }
            else{
                // If we do not have enough matches, we cannot find the transformation, hence print a warning
                
                cout<<"\033[0;31mWe don't have enough matches for patch "+ to_string(i) << "\033[0m" <<endl;
            }



            // Show the blended image and save results
            string title;
            if(homography){
                title = "RANSAC OpenCv & " + title;
            }else{
                title = "RANSAC Manual & " + title;
            }
            if(imageAnalyzer){
                title  = title + "SIFT - Blended Image with patch up to " + to_string(i);
            }else{
                title  = title + "ORB - Blended Image with patch up to " + to_string(i);
            }   
            namedWindow(title,  WINDOW_NORMAL);
            imshow(title, sourceCopy);
            saveImageToFolder(sourceCopy, path, "results", title);

            if(i == patches.size() -1 )
            {
                // Show the differences between the blended image and the real one
                cout << "\nShowing the differences between the blended image and the real one...\n";
                String fd = getLastFolder(path);
                Mat realImage = imread(path +"/" + fd + ".jpg");
                Mat diff = showImageDifferences(sourceCopy, realImage);
                string sub = "Feature Extraction - Differences";
                if(homography)
                    sub = "RANSAC OpenCv & " + sub;
                else
                    sub = "RANSAC Manual & " + sub;
                saveImageToFolder(diff, path, "results", sub);
                waitKey(0);
            }            
            waitKey(0);
        }
    } else{
        /**********************************************************
                                        MULTIPLE IMAGES & PATCHES
            This part of the code is used to associate mixed patches to mixed 
            images. The code structure is excatly the same as the one used for
            single image patches. The only difference is the presence of a function
            that associates patches to the corresponding image.
        **********************************************************/

        // Multiple patches are analyzed only using SIFT (it is more robust than ORB)
        // Define SIFT parameters and allso the value matchesThreshold. It sets the distance threshold for two non accettable matches
        int octavelayers= 3;
        const float distanceRatio = 0.8f;
        int matchesThreshold = 100;

        // Load corrupted images. This time we can have more than one image
        cout<<"Loading corrupted images..."<<endl;
        vector<Mat> sourceImages = loadImage(path, "/image_to_complete_", "Images: ", false);
        cout << "Loaded " << sourceImages.size() << " images...\n" << endl;

        // Assign each patch to the corresponding image
        cout<<"Assigning patches to the correct image..."<<endl;
        vector<vector<Mat>> orderedPatches = multiplePatchMatcher(sourceImages, patches, octavelayers, matchesThreshold, distanceRatio, minDis);


        cout<<"\nDone!\n"<<endl;
        cout<<"Starting the reconstruction process...\n"<<endl;
        for(int t=0; t<sourceImages.size(); t++){

            Mat sourceImage=sourceImages[t];
            vector<Mat> sourcePatches= orderedPatches[t];

            // Detect and compute SIFT features of the corrupted image
            vector<KeyPoint> keypointsImage;
            Mat descriptorsImage;
            bool drawResults = 0;
            tie (descriptorsImage, keypointsImage) = KeypointsFeatureExtractor(recognizer, sourceImage, drawResults, flag);


            // Auxiliary functions
            vector<vector<KeyPoint>> keypointsPatches;
            vector<Mat> descriptorsPatches;
            vector<vector<DMatch>> matches;
            //const float distanceRatio = 0.8f;
            
            // Iterate through all the patch files and perform SIFT featue extraction (with flipping) and matching.
            for (int i = 0; i < sourcePatches.size(); i++) {
                
                vector<KeyPoint> kpTemp; 
                Mat descTemp;

                // Find best flipping
                sourcePatches[i] = flippingMatcher (recognizer, sourcePatches[i], descriptorsImage, distanceRatio, minDis);

                // Detect and compute SIFT features of the pach        
                tie (descTemp, kpTemp) = KeypointsFeatureExtractor(recognizer, sourcePatches[i], drawResults, flag);
                keypointsPatches.push_back(kpTemp);
                descriptorsPatches.push_back(descTemp); 
                
                // Match SIFT descriptors of the image and the patch
                if(flag == 1){
                    matches.push_back(matchRefine(descriptorsImage, descriptorsPatches[i], distanceRatio, Personal_Flags::SIFT, minDis)); 
                }else{
                    matches.push_back(matchRefine(descriptorsImage, descriptorsPatches[i], distanceRatio, Personal_Flags::ORB, minDis));
                }  
                
                // Show matching results and save them
                Mat usefulMatches;
                
                drawMatches(sourcePatches[i], keypointsPatches[i], sourceImage, keypointsImage, matches[i], usefulMatches);
                string title = "Good matches for image " + to_string(i) +" with patch " + to_string(t);
                namedWindow(title, WINDOW_NORMAL);
                imshow(title, usefulMatches);
                saveImageToFolder(usefulMatches,path, "results", title);
                waitKey(0); 
            }

            //create a copy of the image to attache the patches
            Mat sourceCopy = sourceImage.clone();

            
            for (int i = 0; i < sourcePatches.size(); i++) {  
        
                //checking if we have enough matches to find a transformation and then merge the image
                if(matches[i].size()>=minMatchRANSAC){
                    bool homography = true;
                    
                    Mat warpedPatch;
                    if  (homography) {
                        Mat H =homograpySTD (keypointsImage, keypointsPatches[i], matches[i]);
                        //using warpPerspective since we have an homography
                        warpPerspective(sourcePatches[i], warpedPatch, H, sourceImage.size());       
                        
                    }
                    else {
                        Mat A = homograpyMAN (keypointsImage, keypointsPatches[i], matches[i]);                
                        //using warpAffine since we have an affine transfomation 
                        warpAffine(sourcePatches[i], warpedPatch, A, sourceImage.size());
                    }


                    sourceCopy = mergeNoShape (sourceCopy, warpedPatch);

                }
                else{
                    cerr<<"We don't have enough matches for patch "+ to_string(i)<<endl;
                }

                string title = "Image " + to_string(t) + " blended with patch up to "   +    to_string(i);
                namedWindow(title, WINDOW_NORMAL);
                imshow(title, sourceCopy);
                saveImageToFolder(sourceCopy,path, "results", title);                
                waitKey(0);



        }
        
        }

    }

    cout<<"\n";
    // Wait for key press to advance
    waitKey(0);
}
