#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	// Read image from file
	//string imgDir = "C:\\Users\\mimtiaz\\visualStudio17Projects\\getMeHired\\computerVision\\standard-test-images-for-Image-Processing\\standard_test_images\\lena.bmp";
	string imgDir = "lena.bmp";
	Mat img = imread(imgDir, IMREAD_GRAYSCALE);		//Reading one plane grayscale image
	
	//checking reading image
	if (img.empty()) {
		printf("ERROR reading image!\n");
	}

	/*
	There will be two output after DFT: Real and complex. So we need containers that can hold these outputs
	There are two ways to setup these containers containers for DFT.
	*/

	//-------------------------

	/*
	Process 1 follwoing:::
	Complex plane to contain the DFT coefficients {[0]-Real,[1]-Img}
	*/
	Mat planes[] = { Mat_<float>(img), Mat::zeros(img.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);

	dft(complexI, complexI);	// Applying DFT

	/*
	Process 2 follwoing:::
	Complex plane to contain the DFT coefficients {[0]-Real,[1]-Img}
	*/
	//Mat dftInput, dftImage1;
	//img.convertTo(dftInput, CV_32F);
	//dft(dftInput, dftImage1, DFT_COMPLEX_OUTPUT);    // Applying DFT


	// Reconstructing original imae from the DFT coefficients
	Mat invDFT, invDFTcvt;
	idft(complexI, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT for process 1
	//idft(dftImage1, invDFT, DFT_SCALE | DFT_REAL_OUTPUT); // Applying IDFT for process 2
	invDFT.convertTo(invDFTcvt, CV_8U);
	imshow("Output", invDFTcvt);

	//show the image
	imshow("Original Image", img);

	// Wait until user press some key
	waitKey(0);

	//Releasing allocated mamory
	img.release();
	planes->release();
	complexI.release();
	invDFT.release();
	invDFTcvt.release();

	return 0;
}




