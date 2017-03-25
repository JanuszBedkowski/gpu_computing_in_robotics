#ifndef __PROTOTYPES_H__
#define __PROTOTYPES_H__

extern "C"
{
	unsigned int getKeyPointsCount();
	bool cuda_computeKeyPointsRot(int *_d_keypointsIndexX, int *_d_keypointsIndexY, int *_d_keypointsRotation, unsigned char *_d_image, bool *_d_lookup_table, int * _d_rotation_lookup, int _h_width, int _h_height, int _thresholdIntensity);
	bool cuda_computeIntegralImage(unsigned int *_d_out_integralImage, unsigned char *_d_in_image, int _h_width, int _h_height, int _debugMode);
	bool computeDesctriptorCUDARot(bool *_d_isdescriptor, char *_d_vdescriptor,
		int *_d_keypointsIndexX, int *_d_keypointsIndexY, int *_d_keypointsRotation, int _amountofkeypoints, unsigned int *_d_integralImage, int _d_width, int _d_height, float _scale);
	
}

#endif
