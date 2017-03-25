#ifndef __CBFROSTMatcher__
#define __CBFROSTMatcher__

#include <GL/freeglut.h>
#include <string>
#include <vector>

using namespace std;

typedef struct pair
{
	int indexLeft;
	int indexRight;
}pair_t;

typedef struct temp
{
	bool descriptor[256];
	int index;
}temp_t;

class CBFROSTMatcher
{
public:
	CBFROSTMatcher();
	~CBFROSTMatcher();

	bool loadPhotos(std::string filename1, std::string filename2);
	void computeKeypointsWithRotation();
	void paintGL();

	bool showInputCUDA;
	bool showComputedKeyPoints;
	bool showROIForDescriptors;

	bool *h_out_keypoints_left;
	int *h_out_keypointsXindexes_left;
	int *h_out_keypointsYindexes_left;
	unsigned char *h_in_image_left;
	unsigned int *h_integral_image_left;
	int img1width_left;
	int img1height_left;

	bool *h_out_keypoints_right;
	int *h_out_keypointsXindexes_right;
	int *h_out_keypointsYindexes_right;
	unsigned char *h_in_image_right;
	unsigned int *h_integral_image_right;
	int img1width_right;
	int img1height_right;

	int thresholdAmountOfContiguousPoints;
	int thresholdIntensity;

	float X[64], Y[64], Z[64], n[64], scale;
	int queryPoint;
	int angle;

	vector<pair_t> vPair;

	int descriptorThreshold;
	bool showMatchingResult;
};


#endif
