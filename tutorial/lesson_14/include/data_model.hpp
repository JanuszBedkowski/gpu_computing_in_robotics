#ifndef __DATA_TRANSF_HPP
#define __DATA_TRANSF_HPP

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/optional/optional.hpp>
#include <Eigen/Eigen>
#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
class data_model
{
public:
	
	/// loads XML file containing set of transformations
	bool loadFile(std::string fn);
	/// saveFileves XML set of transformations
	bool saveFile(std::string fn);
	/// loads affine transform with given scan ID
	bool getAffine (std::string scanId, Eigen::Matrix4f &matrix);
	///loads affine transform with given scan ID
	bool getAffine (std::string scanId, Eigen::Vector3f &origin, Eigen::Quaternionf &quat);
	///get pointcloudName
	bool getPointcloudName (std::string scanId, std::string &fn);
	
	std::string getFullPathOfPointcloud(std::string id);

	void setDataSetPath(std::string path);
	void getDataSetPath(std::string &path);
	/// retrieve all scans in file 
	void getAllScansId(std::vector<std::string> &ids);
	/// sets affine to pointcloud wh given id
	void setAffine (std::string scanId, Eigen::Matrix4f &matrix);
	/// sets affine to pointcloud wh given id
	void setAffine (std::string scanId, Eigen::Vector3f origin, Eigen::Quaternionf quat);
	/// sets pcd filename of given scan with id
	void setPointcloudName (std::string scanId, std::string fn);
	
	/// sets invoked algorithm name
	void setAlgorithmName(std::string name);
	
	void addAlgorithmParam(std::string paramName, float paramValue);
	void addAlgorithmParam(std::string paramName, std::string paramValue);
	
	void setResult(std::string resultName, float result);
	void setResult(std::string scanId, std::string resultName, float result);
	void setResult(std::string scanId, std::string resultName, std::string result);
	void getResult(std::string scanId, std::string resultName, float &result);

	void getGlobalModelMatrix(Eigen::Matrix4f &matrix);
	void setGlobalModelMatrix(Eigen::Matrix4f matrix);

    void setTimestamp (std::string scanId, boost::posix_time::ptime ts);
    bool getTimestamp (std::string scanId, boost::posix_time::ptime &ts);

    /// sets gps to pointcloud wh given id
    void setGPS (std::string scanId, double lot, double lan, double att);
    bool getGPS (std::string scanId, double &lot, double &lan, double &att);
private:
	boost::property_tree::ptree pt_;

	void addIfNotExists(std::string path);
	bool checkIfExists(std::string path);
	boost::filesystem::path xmlPath;
};

#endif
