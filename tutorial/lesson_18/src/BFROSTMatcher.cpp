#include <string>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "GPUMatching.h"
#include "BFROSTMatcher.h"


using namespace std;
unsigned int texLeft=0,texGrayLeft=0;
unsigned int texRight=0,texGrayRight=0;


CBFROSTMatcher::CBFROSTMatcher()
{
	showInputCUDA = false;
	showComputedKeyPoints = false;
	showROIForDescriptors = false;

	this->img1height_left = 1;
	this->img1width_left = 1;
	h_out_keypoints_left = (bool *)malloc(sizeof(bool) * img1width_left*img1height_left);
	h_out_keypointsXindexes_left = (int *)malloc(sizeof(int) * img1width_left*img1height_left);
	h_out_keypointsYindexes_left = (int *)malloc(sizeof(int) * img1width_left*img1height_left);
	h_in_image_left = (unsigned char *)malloc(sizeof(unsigned char) * img1width_left*img1height_left);
	h_integral_image_left = (unsigned int *)malloc(sizeof(unsigned int) * img1width_left*img1height_left);

	this->img1height_right = 1;
	this->img1width_right = 1;
	h_out_keypoints_right = (bool *)malloc(sizeof(bool) * img1width_right*img1height_right);
	h_out_keypointsXindexes_right = (int *)malloc(sizeof(int) * img1width_right*img1height_right);
	h_out_keypointsYindexes_right = (int *)malloc(sizeof(int) * img1width_right*img1height_right);
	h_in_image_right = (unsigned char *)malloc(sizeof(unsigned char) * img1width_right*img1height_right);
	h_integral_image_right = (unsigned int *)malloc(sizeof(unsigned int) * img1width_right*img1height_right);

	this->thresholdAmountOfContiguousPoints = 9;
    this->thresholdIntensity = 20;
	this->scale = 1;
	queryPoint = 0;
	angle = 0;
	this->descriptorThreshold = 25;
	showMatchingResult = false;
}

CBFROSTMatcher::~CBFROSTMatcher()
{
	delete [] h_out_keypoints_left;
	delete [] h_in_image_left;
	delete [] h_integral_image_left;

	delete [] h_out_keypoints_right;
	delete [] h_in_image_right;
	delete [] h_integral_image_right;
}

void CBFROSTMatcher::paintGL()
{
	glOrtho (0,img1width_left,0,img1height_left, -100.0f, 100.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

	glColor3f(1,1,1);

	glEnable(GL_TEXTURE_2D);
	if(!showInputCUDA)
	{
		glBindTexture(GL_TEXTURE_2D,texLeft);
	}else
	{
		glBindTexture(GL_TEXTURE_2D,texGrayLeft);
	}

		glBegin(GL_QUADS);
			glTexCoord2f(0,1);
			glVertex3f(0,0,-10.0);
			glTexCoord2f(1,1);
			glVertex3f(img1width_left/2,0,-10.0);
			glTexCoord2f(1,0);
			glVertex3f(img1width_left/2,img1height_left,-10.0);
			glTexCoord2f(0,0);
			glVertex3f(0,img1height_left,-10.0);
		glEnd();

//////////////////////////RIGHT VIEW//////////////
		
		glEnable(GL_TEXTURE_2D);
		if(!showInputCUDA)
		{
			glBindTexture(GL_TEXTURE_2D,texRight);
		}else
		{
			glBindTexture(GL_TEXTURE_2D,texGrayRight);
		}

		glBegin(GL_QUADS);
			glTexCoord2f(0,1);
			glVertex3f(img1width_left/2,0,-10.0);
			glTexCoord2f(1,1);
			glVertex3f(img1width_left,0,-10.0);
			glTexCoord2f(1,0);
			glVertex3f(img1width_left,img1height_left,-10.0);
			glTexCoord2f(0,0);
			glVertex3f(img1width_left/2,img1height_left,-10.0);
		glEnd();
		
		if(this->showComputedKeyPoints)
		{
			glDisable(GL_TEXTURE_2D);
			glColor3f(1,0,0);
			glPointSize(3);
			glBegin(GL_POINTS);
			for(int i = 0 ; i < img1width_left*img1height_left; i++)
			{
				if(h_out_keypoints_left[i])
				{
					glVertex3f(h_out_keypointsXindexes_left[i]/2, img1height_left-h_out_keypointsYindexes_left[i], -9.0f);
				}
			}
			glEnd();

			glBegin(GL_POINTS);
			for(int i = 0 ; i < img1width_right*img1height_right; i++)
			{
				if(h_out_keypoints_right[i])
				{
					glVertex3f(img1width_left/2 + h_out_keypointsXindexes_right[i]/2, img1height_right-h_out_keypointsYindexes_right[i], -9.0f);
				}
			}
			glEnd();
		}

		if(this->showROIForDescriptors == true)
		{
			
			glDisable(GL_TEXTURE_2D);
			glPointSize(3);
			glColor3f(0.0f, 1.0f, 0.0f);
			glBegin(GL_POINTS);
			for(int i = 0 ; i < img1width_left*img1height_left; i++)
			{
				if(h_out_keypoints_left[i])
				{
					for(int ii = 0 ; ii < 64; ii++)
					{
						glVertex3f(this->X[ii] + h_out_keypointsXindexes_left[i]/2, this->Y[ii]+img1height_left-h_out_keypointsYindexes_left[i], -1.0f);
					}
				}
			}
			glEnd();

			glPointSize(3);
			glColor3f(0.0f, 1.0f, 0.0f);
			glBegin(GL_POINTS);
			for(int i = 0 ; i < img1width_right*img1height_right; i++)
			{
				if(h_out_keypoints_right[i])
				{
					for(int ii = 0 ; ii < 64; ii++)
					{
						glVertex3f(img1width_left/2 + this->X[ii] + h_out_keypointsXindexes_right[i]/2, this->Y[ii]+img1height_right-h_out_keypointsYindexes_right[i], -1.0f);
					}
				}
			}
			glEnd();



			glColor3f(1.0f, 0.0f, 0.0f);
			
			for(int i = 0 ; i < img1width_left*img1height_left; i++)
			{
				if(h_out_keypoints_left[i])
				{
					for(int ii = 0 ; ii < 64; ii ++)
					{
						glBegin(GL_LINE_STRIP);
							glVertex3f((this->X[ii] + h_out_keypointsXindexes_left[i]/2 - this->Z[ii]), (this->Y[ii]+img1height_left-h_out_keypointsYindexes_left[i] - this->Z[ii]) , 0);
							glVertex3f((this->X[ii] + h_out_keypointsXindexes_left[i]/2 + this->Z[ii]), (this->Y[ii]+img1height_left-h_out_keypointsYindexes_left[i] - this->Z[ii]) , 0);
							glVertex3f((this->X[ii] + h_out_keypointsXindexes_left[i]/2 + this->Z[ii]), (this->Y[ii]+img1height_left-h_out_keypointsYindexes_left[i] + this->Z[ii]) , 0);
							glVertex3f((this->X[ii] + h_out_keypointsXindexes_left[i]/2 - this->Z[ii]), (this->Y[ii]+img1height_left-h_out_keypointsYindexes_left[i] + this->Z[ii]) , 0);
							glVertex3f((this->X[ii] + h_out_keypointsXindexes_left[i]/2 - this->Z[ii]), (this->Y[ii]+img1height_left-h_out_keypointsYindexes_left[i] - this->Z[ii]) , 0);
						glEnd();
					}
				}
			}

			for(int i = 0 ; i < img1width_right*img1height_right; i++)
			{
				if(h_out_keypoints_right[i])
				{
					for(int ii = 0 ; ii < 64; ii ++)
					{
						glBegin(GL_LINE_STRIP);
							glVertex3f((img1width_left/2 + this->X[ii] + h_out_keypointsXindexes_right[i]/2 - this->Z[ii]), (this->Y[ii]+img1height_right-h_out_keypointsYindexes_right[i] - this->Z[ii]) , 0);
							glVertex3f((img1width_left/2 + this->X[ii] + h_out_keypointsXindexes_right[i]/2 + this->Z[ii]), (this->Y[ii]+img1height_right-h_out_keypointsYindexes_right[i] - this->Z[ii]) , 0);
							glVertex3f((img1width_left/2 + this->X[ii] + h_out_keypointsXindexes_right[i]/2 + this->Z[ii]), (this->Y[ii]+img1height_right-h_out_keypointsYindexes_right[i] + this->Z[ii]) , 0);
							glVertex3f((img1width_left/2 + this->X[ii] + h_out_keypointsXindexes_right[i]/2 - this->Z[ii]), (this->Y[ii]+img1height_right-h_out_keypointsYindexes_right[i] + this->Z[ii]) , 0);
							glVertex3f((img1width_left/2 + this->X[ii] + h_out_keypointsXindexes_right[i]/2 - this->Z[ii]), (this->Y[ii]+img1height_right-h_out_keypointsYindexes_right[i] - this->Z[ii]) , 0);
						glEnd();
					}
				}
			}
		}
		
		if(showMatchingResult)
		{
			glDisable(GL_TEXTURE_2D);
			glColor3f(0,1,0);
			glBegin(GL_LINES);
			for(int i = 0 ; i < (int)vPair.size(); i++)
			{
				glVertex3f(h_out_keypointsXindexes_left[vPair[i].indexLeft]/2, img1height_left-h_out_keypointsYindexes_left[vPair[i].indexLeft], 0);
				glVertex3f(img1width_left/2 + h_out_keypointsXindexes_right[vPair[i].indexRight]/2,img1height_left-h_out_keypointsYindexes_right[vPair[i].indexRight], 0);
			}
			glEnd();
		}

}

bool CBFROSTMatcher::loadPhotos(std::string filename1, std::string filename2)
{
	cv::Mat imageLeft;
	cv::Mat imageRight;
	cv::Mat imageLeftGray;
	cv::Mat imageRightGray;


	imageLeft = cv::imread( filename1, 1 );
	imageRight = cv::imread( filename2, 1 );

	img1width_left=imageLeft.size().width;
	img1height_left=imageLeft.size().height;

	unsigned char *pixels;
	pixels=new unsigned char [img1width_left*img1height_left*3];

	memcpy(pixels, imageLeft.data, sizeof(unsigned char)* img1width_left*img1height_left*3);

	if(texLeft)
	glDeleteTextures(1,&texLeft);

	glGenTextures(1,&texLeft);
	glBindTexture(GL_TEXTURE_2D, texLeft);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,img1width_left,img1height_left,0,GL_BGR, GL_UNSIGNED_BYTE,pixels);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,GL_REPEAT);

	unsigned char *graypixels;
	graypixels=new unsigned char [img1width_left*img1height_left*3];

	for(int i = 0 ; i< img1height_left; i++)
		for(int j = 0 ; j < img1width_left; j++)
		{
			int index = j+i*img1width_left;
			unsigned char a = pixels[index*3];
			unsigned char b = pixels[index*3+ 1];
			unsigned char c = pixels[index*3+ 2];
			unsigned char d = (unsigned char)floor(float(a+b+c)/3.0f);
			graypixels[index*3] = d;
			graypixels[index*3+1] = d;
			graypixels[index*3+2] = d;
		}

	if(texGrayLeft)
		glDeleteTextures(1,&texGrayLeft);

	glGenTextures(1,&texGrayLeft);
	glBindTexture(GL_TEXTURE_2D, texGrayLeft);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,img1width_left,img1height_left,0,GL_RGB, GL_UNSIGNED_BYTE,graypixels);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,GL_REPEAT);

	free(h_out_keypoints_left); h_out_keypoints_left = 0;
	free(h_in_image_left); h_in_image_left = 0;
	free(h_out_keypointsXindexes_left); h_out_keypointsXindexes_left = 0;
	free(h_out_keypointsYindexes_left); h_out_keypointsYindexes_left = 0;


	h_out_keypoints_left = (bool *)malloc(sizeof(bool) * img1width_left*img1height_left);
	h_out_keypointsXindexes_left = (int *)malloc(sizeof(int) * img1width_left*img1height_left);
	h_out_keypointsYindexes_left = (int *)malloc(sizeof(int) * img1width_left*img1height_left);
	h_in_image_left = (unsigned char *)malloc(sizeof(unsigned char) * img1width_left*img1height_left);
	h_integral_image_left = (unsigned int *)malloc(sizeof(unsigned int) * img1width_left*img1height_left);

	for(int i = 0; i < img1width_left*img1height_left; i++)
	{
		h_out_keypoints_left[i] = false;
		h_in_image_left[i] = graypixels[i * 3];
		h_integral_image_left[i] = 0;
	}

	for(int i = 0; i< img1height_left; i++)
		for(int j = 0; j < img1width_left; j++)
		{
			int index = j+i*img1width_left;
			this->h_out_keypointsXindexes_left[index] = j;
			this->h_out_keypointsYindexes_left[index] = i;
		}

	delete[] graypixels;
	delete[] pixels;

	this->img1width_right=imageRight.size().width;
	this->img1height_right=imageRight.size().height;

	printf("Size of loaded image: width:%d height:%d\n", this->img1width_right, this->img1height_right);

	pixels=new unsigned char [img1width_right*img1height_right*3];

	memcpy(pixels, imageRight.data, sizeof(unsigned char)* img1width_right*img1height_right*3);

	if(texRight)
		glDeleteTextures(1,&texRight);

	glGenTextures(1,&texRight);
	glBindTexture(GL_TEXTURE_2D, texRight);

	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,img1width_right,img1height_right,0,GL_BGR, GL_UNSIGNED_BYTE,pixels);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,GL_REPEAT);

	graypixels=new unsigned char [img1width_right*img1height_right*3];

	for(int i = 0 ; i< img1height_right; i++)
		for(int j = 0 ; j < img1width_right; j++)
		{
			int index = j+i*img1width_right;
			unsigned char a = pixels[index*3];
			unsigned char b = pixels[index*3+ 1];
			unsigned char c = pixels[index*3+ 2];
			unsigned char d = (unsigned char)floor(float(a+b+c)/3.0f);

			graypixels[index*3] = d;
			graypixels[index*3+1] = d;
			graypixels[index*3+2] = d;
		}

	if(texGrayLeft)
		glDeleteTextures(1,&texGrayRight);

	glGenTextures(1,&texGrayRight);
	glBindTexture(GL_TEXTURE_2D, texGrayRight);

	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,img1width_right,img1height_right,0,GL_RGB, GL_UNSIGNED_BYTE,graypixels);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,GL_REPEAT);

	free(h_out_keypoints_right); h_out_keypoints_right = 0;
	free(h_in_image_right); h_in_image_right = 0;
	free(h_out_keypointsXindexes_right); h_out_keypointsXindexes_right = 0;
	free(h_out_keypointsYindexes_right); h_out_keypointsYindexes_right = 0;

	h_out_keypoints_right = (bool *)malloc(sizeof(bool) * img1width_right*img1height_right);
	h_out_keypointsXindexes_right = (int *)malloc(sizeof(int) * img1width_right*img1height_right);
	h_out_keypointsYindexes_right = (int *)malloc(sizeof(int) * img1width_right*img1height_right);
	h_in_image_right = (unsigned char *)malloc(sizeof(unsigned char) * img1width_right*img1height_right);
	h_integral_image_right = (unsigned int *)malloc(sizeof(unsigned int) * img1width_right*img1height_right);

	for(int i = 0; i < img1width_right*img1height_right; i++)
	{
		h_out_keypoints_right[i] = false;
		h_in_image_right[i] = graypixels[i * 3];
		h_integral_image_right[i] = 0;
	}

	for(int i = 0; i< img1height_right; i++)
		for(int j = 0; j < img1width_right; j++)
		{
			int index = j+i*img1width_right;
			this->h_out_keypointsXindexes_right[index] = j;
			this->h_out_keypointsYindexes_right[index] = i;
		}

	delete[] graypixels;
	delete[] pixels;

	return true;
}

void CBFROSTMatcher::computeKeypointsWithRotation()
{
	CGPUMatching gpuMatching;
	gpuMatching.SetDebugLevel(0);
	if(gpuMatching.Init(0, img1width_left,img1height_left,this->thresholdAmountOfContiguousPoints))
	{
		int *h_keypointsIndexX1 = (int *)malloc(sizeof(int)*this->img1width_left* this->img1height_left);
		int *h_keypointsIndexY1 = (int *)malloc(sizeof(int)*this->img1width_left* this->img1height_left);
		int *h_keypointsIndexX2 = (int *)malloc(sizeof(int)*this->img1width_left* this->img1height_left);
		int *h_keypointsIndexY2 = (int *)malloc(sizeof(int)*this->img1width_left* this->img1height_left);

		char *vdesctriptor1 = (char *)malloc(sizeof(char)* this->img1width_left* this->img1height_left* 32);
		bool *isvdesctriptor1 = (bool *)malloc(sizeof(bool)* this->img1width_left* this->img1height_left);

		char *vdesctriptor2 = (char *)malloc(sizeof(char)* this->img1width_left* this->img1height_left* 32);
		bool *isvdesctriptor2 = (bool *)malloc(sizeof(bool)* this->img1width_left* this->img1height_left);
		
		gpuMatching.ComputeSamplePatternGPU(this->X, this->Y, this->Z, this->n, this->scale);

		int amountofkeypoints1 = gpuMatching.ComputeKeyPointsRot(h_out_keypoints_left, h_in_image_left, img1width_left, img1height_left, this->thresholdIntensity);
		gpuMatching.ComputeIntegralImageGPU(this->h_integral_image_left, this->h_in_image_left, this->img1width_left, this->img1height_left,5);
		gpuMatching.ComputeDescriptorGPURot(vdesctriptor1, isvdesctriptor1, this->img1width_left, this->img1height_left, amountofkeypoints1, this->scale);

		gpuMatching.getIndexesOfKeypoints(h_keypointsIndexX1, h_keypointsIndexY1, this->img1width_left, this->img1height_left);
				
		int amountofkeypoints2 = gpuMatching.ComputeKeyPointsRot(h_out_keypoints_right, h_in_image_right, img1width_right, img1height_right, this->thresholdIntensity);
		gpuMatching.ComputeIntegralImageGPU(this->h_integral_image_right, this->h_in_image_right, this->img1width_right, this->img1height_right,5);
		gpuMatching.ComputeDescriptorGPURot(vdesctriptor2, isvdesctriptor2,this->img1width_left, this->img1height_left, amountofkeypoints2, this->scale);
		gpuMatching.getIndexesOfKeypoints(h_keypointsIndexX2, h_keypointsIndexY2, this->img1width_left, this->img1height_left);

		/////////////////maching hamming////////////
		for(int i =0 ;i < amountofkeypoints1; i++)
		{
			if(isvdesctriptor1[i])
			{
				int dist_min = 10000;
				int tempIndex = -1;
				int ind1 = h_keypointsIndexX1[i] + h_keypointsIndexY1[i]*this->img1width_left;
				int ind2 ;

				for(int j = 0; j < amountofkeypoints2; j++)
				{
					if(isvdesctriptor2[j])
					{
						bool descriptor1[256];
						bool descriptor2[256];

						for(int k = 0 ; k < 32 ; k++)
						{
							char wynik1 = vdesctriptor1[k + i*32];
							char wynik2 = vdesctriptor2[k + j*32];
							bool descriptor1b[8];
							bool descriptor2b[8];
							for(int l = 0; l < 8 ; l++)
							{
								descriptor1b[l] = wynik1 & (1 << l);
								descriptor2b[l] = wynik2 & (1 << l);
								descriptor1[k * 8 + l] = descriptor1b[l];
								descriptor2[k * 8 + l] = descriptor2b[l];
							}
						}

						int distance = gpuMatching.distanceHamming(descriptor1, descriptor2);
						if(distance < dist_min)
						{
							tempIndex=j;
							dist_min = distance;
							ind2 = h_keypointsIndexX2[j] + h_keypointsIndexY2[j]*this->img1width_right;
						}
					}
				}
				if(tempIndex!=-1 && dist_min <= this->descriptorThreshold)
				{
					pair_t p;
					
					p.indexLeft = ind1;
					p.indexRight = ind2;
					vPair.push_back(p);
				}
			}
		}

		free(h_keypointsIndexX1);
		free(h_keypointsIndexY1);
		free(h_keypointsIndexX2);
		free(h_keypointsIndexY2);
		
		free(vdesctriptor1);
		free(isvdesctriptor1);
		free(vdesctriptor2);
		free(isvdesctriptor2);

		gpuMatching.Free();

		showComputedKeyPoints = true;
		showMatchingResult = true;
	}
}

