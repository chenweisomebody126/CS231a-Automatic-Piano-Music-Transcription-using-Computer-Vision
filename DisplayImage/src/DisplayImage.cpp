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
//https://stackoverflow.com/questions/38715363/how-to-implement-imrotate-of-matlab-in-opencv
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

Mat findAngleViaHoughPeaks(Mat img){
//	int height = img.rows;
//	int width = img.cols;
//	int img_diagonal = ceil(sqrt(height*height + width*width));
//	int rhos[2*img_diagonal];
//	for (int i =0; i< sizeof(rhos); i++){
//		rhos[i] = i-img_diagonal;
//	}
//	double thetas[180];
//	for (int i=0; i< sizeof(thetas); i++){
//		thetas[0] = ((i-90) * M_PI) / 180;
//	}
//	Mat H(sizeof(rhos), sizeof(thetas), CV_32S, Scalar(0));
//    Mat nonZeroCoordinates;
//    findNonZero(img, nonZeroCoordinates);
//    for (int i = 0; i < nonZeroCoordinates.total(); i++ ) {
//    		Point pnt = nonZeroCoordinates.at<Point>(i);
//    		for (int j=0; j<sizeof(thetas); j++){
//    			int rho = int((pnt.x * cos(thetas[j]) + pnt.y * sin(thetas[j])) + img_diagonal);
//    			H[rho, j] += 1;
//    		}
//        //cout << "Zero#" << i << ": " << nonZeroCoordinates.at<Point>(i).x << ", " << nonZeroCoordinates.at<Point>(i).y << endl;
//    }
//    //hough_peaks
//    double minVal, maxVal;
//    cv::minMaxLoc(H, &minVal, &maxVal);
//    double threshold = 0.3*maxVal;
//    int H1[H.total()];
//    for (int i=0; i<H.rows; i++){
//    		for (int j=0; j<H.cols; j++){
//        		if(H[i, j]< threshold){
//        			H1[i*H.rows+j] = 0;
//        		} else{
//        			H1[i*H.rows+j] = H[i, j];
//        		}
//    		}
//    }
//    int num_peaks = 20;
//    kth_smallest_idx = quick_select(H1, 0, sizeof(H1)-1, sizeof(H1)-num_peaks);
//
//
//    for (int ith_peak = 0; ith_peak< num_peaks; ith_peak++){
//    		minMaxIdx()
//    }
//    indices =  np.argpartition(H.flatten(), -2)[-num_peaks:]
//    return np.vstack(np.unravel_index(indices, H.shape)).T

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

//  vector<Vec2f> lines;
//  Canny(initialFrameBin, initialFrameBin, 50, 200, 3);
//  HoughLines(initialFrameBin, lines, 1, CV_PI/180, 100, 0, 0 );
//
//  for( size_t i = 0; i < lines.size(); i++ )
//  {
//     float rho = lines[i][0], theta = lines[i][1];
//     Point pt1, pt2;
//     double a = cos(theta), b = sin(theta);
//     double x0 = a*rho, y0 = b*rho;
//     pt1.x = cvRound(x0 + 1000*(-b));
//     pt1.y = cvRound(y0 + 1000*(a));
//     pt2.x = cvRound(x0 - 1000*(-b));
//     pt2.y = cvRound(y0 - 1000*(a));
//     line( initialFrameBin, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
//  }
//  imshow("detected lines", initialFrameBin);
//  waitKey(0);


  Mat initialFrameBinNeg;
  bitwise_not(initialFrameBin, initialFrameBinNeg);
  Mat initialFrameGrad = rectify(initialFrameBin);

  namedWindow( "Display Image", WINDOW_AUTOSIZE );
  imshow( "Display Image", initialFrameGrad );
  waitKey(0);
  return 0;
}
