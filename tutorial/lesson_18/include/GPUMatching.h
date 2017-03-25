#ifndef __GPUMATCHING_H__
#define __GPUMATCHING_H__

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>


class CGPUMatching
{
public:
	CGPUMatching(void);
	~CGPUMatching(void);

	private:
		unsigned char *d_image;
		unsigned int *d_integralImage;

		bool *d_keypoints;
		int *d_keypointsIndexX;
		int *d_keypointsIndexY;
		int *d_keypointsRotation;
		bool *d_lookup_table;
		int *d_rotation_lookup;
		bool *d_isdescriptor;
		bool *d_vdescriptor;
		char *d_vdescriptorCHAR;

		float *d_tempS;

		float *d_X;
		float *d_Y;
		float *d_Z;
		float *d_n;

		bool h_lookuptable[65536];
		int h_lookuptable_rot[65536];
		int d_width;
		int d_height;
		int d_debugLevel;
		int d_cudaDevice;
		cudaDeviceProp deviceProp;
		int thresholdAmountOfContiguousPoints;

		int get_rotation(int i);
		int get_rot_len(int a, int b);
		int get_rot_index(int rot_mask);

		bool checkiffeature(int a0, int a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8, int a9, int a10, int a11, int a12, int a13, int a14, int a15, int _debugLevel);
		void initLookupTable();
		void initLookupTableRot();

	public:
		bool Init(int _cudaDevice, int _d_width, int _d_height, int _thresholdAmountOfContiguousPoints);
		void Free();
		void SetDebugLevel(int _debugLevel);
		int ComputeKeyPointsRot(bool *_h_out_keypoints, unsigned char *_h_in_image, int _h_width, int _h_height, int _thresholdIntensity);
		bool ComputeIntegralImageGPU(unsigned int *_h_out_integralImage, unsigned char *_h_in_image, int _h_width, int _h_height, int _debugMode);
		bool ComputeSamplePattern(float *_X, float *_Y, float *_Z, float *_n, float _scale);
		bool ComputeSamplePatternGPU(float *_X, float *_Y, float *_Z, float *_n, float _scale);
		int distanceHamming(bool descriptor1[256], bool descriptor2[256]);
		bool ComputeDescriptorGPURot(char *_h_vdesctriptor, bool *_h_isvdesctriptor,
				int _img1width, int _img1height, int _amountofkeypoints, float _scale);
		void getIndexesOfKeypoints(int *_h_keypointsIndexX, int *_h_keypointsIndexY,int _img1width, int _img1height);
};

#endif
