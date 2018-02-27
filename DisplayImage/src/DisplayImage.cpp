#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Mat removeSmallRegions(Mat img, int neighbors){
	const int connectivity = neighbors;
	Mat labelImage, stats, centroids;
	int nLabels = connectedComponentsWithStats(img, labelImage, stats, centroids, connectivity, CV_32S);

    Mat mask(labelImage.size(), CV_8UC1, Scalar(0));
    Mat surfSup=stats.col(4)>150;

    for (int i = 1; i < nLabels; i++){
        if (surfSup.at<uchar>(i, 0)){
            mask = mask | (labelImage==i);
        }
    }
    Mat r(img.size(), CV_8UC1, Scalar(0));
    img.copyTo(r,mask);
    //imshow("Result", r);

	return r;
}

Mat imRotate(const Mat source, double angle) {
    cv::Mat dst;
    // Special Cases
    if (std::fmod(angle, 360.0) == 0.0)
        dst = source;
    else{
        cv::Point2f center(source.cols / 2.0F, source.rows / 2.0F);
        cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
        // determine bounding rectangle
        cv::Rect bbox = cv::RotatedRect(center, source.size(), angle).boundingRect();
        // adjust transformation matrix
        rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
        rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;
        cv::warpAffine(source, dst, rot, bbox.size(), cv::INTER_LINEAR);

    }
    return dst;
}


Mat rectify(Mat imgNeg){
// Use Sobel filter to find horizontal gradient
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y,imgSobel;
	Sobel( imgNeg, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );
	Sobel( imgNeg, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, imgSobel );
	// Find the angle via the hough and houghpeaks
	vector<Vec2f> lines;
	Canny(imgSobel, imgSobel, 50, 200, 3);

	HoughLines(imgSobel, lines, 1, CV_PI/180, 100, 0, 0 );

	unordered_map<int, int> frequencyCount;

	for( size_t i = 0; i < lines.size(); i++ )
	{
	  float theta = lines[i][1];
	  int roundedTheta= cvRound(theta*180/CV_PI);
	  frequencyCount[roundedTheta]++;
	}
	unsigned currentMax = 0;
	unsigned arg_max = 0;
	for(auto it = frequencyCount.cbegin(); it != frequencyCount.cend(); ++it ) {
	    if (it ->second > currentMax) {
	        arg_max = it->first;
	        currentMax = it->second;
	    }
	}

	for (auto& t : frequencyCount)
	    std::cout << t.first << " "
	              << t.second << "\n";
	printf("the angle is %d", currentMax);
	fflush( stdout );




	unsigned angle = arg_max;
	printf("the angle is %d", angle);
	fflush( stdout );

	//Rotate image
	//Mat imgRotate = imRotate(imgNeg,angle);
	Mat imgGradient = imRotate(imgSobel, angle);
	  namedWindow( "Display Image", WINDOW_AUTOSIZE );
	  imshow( "Display Image", imgGradient );
	  waitKey(0);
	return imgGradient;
}



int main( int argc, char** argv )
{
  Mat initialFrame = imread("images/input.jpg", CV_LOAD_IMAGE_COLOR);
  resize(initialFrame, initialFrame, Size(initialFrame.cols/4, initialFrame.rows/4));


  Mat initialFrameGray;
  cvtColor(initialFrame, initialFrameGray, cv::COLOR_RGB2GRAY);

  Mat initialFrameBin;
  threshold ( initialFrameGray, initialFrameBin, 0, 255, THRESH_BINARY | THRESH_OTSU );

  initialFrameBin = removeSmallRegions(initialFrameBin,4);


  Mat initialFrameBinNeg;
  bitwise_not(initialFrameBin, initialFrameBinNeg);
  Mat initialFrameGrad = rectify(initialFrameBin);

  namedWindow( "Display Image", WINDOW_AUTOSIZE );
  imshow( "Display Image", initialFrameGrad );
  waitKey(0);
  return 0;
}
