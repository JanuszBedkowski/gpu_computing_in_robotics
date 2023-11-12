/*
 * Software License Agreement (BSD License)
 *
 *  Data Registration Framework - Mobile Spatial Assistance System
 *  Copyright (c) 2014-2015, Institute of Mathematical Machinessssssssssssssssss
 *  http://lider.zms.imm.org.pl/
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Institute of Mathematical Machinessssssssssssssssss nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 */

#include "data_model.hpp"
#include <boost/property_tree/detail/xml_parser_writer_settings.hpp>
#include <boost/foreach.hpp>
#include <fstream>

	bool data_model::loadFile(std::string fn)
	{
		pt_.clear();

		xmlPath = boost::filesystem::path(fn);

		//check if the name is valid
		std::ifstream f;
        f.open(fn.c_str());
		if(f.good())
		{
			f.close();
			boost::property_tree::read_xml(fn, pt_);
			return true;
		}
		else
		{
			return false;
		}

	}

	bool data_model::saveFile(std::string fn)
    {
		//caused problems on VS2010, BOOST 1.55
        //boost::property_tree::write_xml(fn, pt_, std::locale(), boost::property_tree::xml_writer_make_settings<char>(' ', 1u));
		//boost::property_tree::write_xml(fn, pt_, std::locale(), boost::property_tree::xml_writer_make_settings(' ', 1u));

		//xmlPath = boost::filesystem::path(fn);
		//return true;
		std::cout << "data_model::saveFile " << "ToDo" << std::endl;
		return false;
	}
	
	bool data_model::getAffine (std::string scanId, Eigen::Matrix4f &matrix)
	{
		if (!checkIfExists("Model.Transformations."+scanId+".Affine.Type")) return false; 
		if (!checkIfExists("Model.Transformations."+scanId+".Affine.Data")) return false; 

		std::string type  = pt_.get<std::string>("Model.Transformations."+scanId+".Affine.Type");
		std::string data  = pt_.get<std::string>("Model.Transformations."+scanId+".Affine.Data");
		matrix = Eigen::Matrix4f::Identity();
		if (type.compare("matrix4f")==0)
		{
			std::stringstream ss (data);
			//column major order
			ss>>matrix(0,0)>>matrix(1,0)>>matrix(2,0)>>matrix(3,0);
			ss>>matrix(0,1)>>matrix(1,1)>>matrix(2,1)>>matrix(3,1);
			ss>>matrix(0,2)>>matrix(1,2)>>matrix(2,2)>>matrix(3,2);
			ss>>matrix(0,3)>>matrix(1,3)>>matrix(2,3)>>matrix(3,3);

			//row major order
			//ss>>matrix(0,0)>>matrix(0,1)>>matrix(0,2)>>matrix(0,3);
			//ss>>matrix(1,0)>>matrix(1,1)>>matrix(1,2)>>matrix(1,3);
			//ss>>matrix(2,0)>>matrix(2,1)>>matrix(2,2)>>matrix(2,3);
			//ss>>matrix(3,0)>>matrix(3,1)>>matrix(3,2)>>matrix(3,3);
			return true;
        }

		if (type.compare("Vector3f_Quaternionf")==0)
		{
			Eigen::Quaternionf q;
			Eigen::Vector3f o;
			std::stringstream ss (data);
			ss>>o.x()>>o.y()>>o.z();
			ss>>q.x()>>q.y()>>q.z()>>q.w();
			Eigen::Affine3f affine;
			affine.translate(o);
			affine.rotate(q);
			matrix = affine.matrix();
			return true;
		}

		return false;
		
	}

	bool data_model::getPointcloudName (std::string scanId, std::string &fn)
	{
		if (!checkIfExists("Model.Transformations."+scanId+".cloudname")) return false; 
		fn = pt_.get<std::string>("Model.Transformations."+scanId+".cloudname");
		return true;
	}

	///loads affine transform with given scan ID
	bool data_model::getAffine (std::string scanId, Eigen::Vector3f &origin, Eigen::Quaternionf &quat)
	{
		if (!checkIfExists("Model.Transformations."+scanId+".Affine.Type")) return false; 
		if (!checkIfExists("Model.Transformations."+scanId+".Affine.Data")) return false; 

		
		std::string type  = pt_.get<std::string>("Model.Transformations."+scanId+".Affine.Type");
		std::string data  = pt_.get<std::string>("Model.Transformations."+scanId+".Affine.Data");
		
		if (type.compare("matrix4f")==0)
		{
			Eigen::Matrix4f matrix;
			std::stringstream ss (data);

			ss>>matrix(0,0)>>matrix(1,0)>>matrix(2,0)>>matrix(3,0);
			ss>>matrix(0,1)>>matrix(1,1)>>matrix(2,1)>>matrix(3,1);
			ss>>matrix(0,2)>>matrix(1,2)>>matrix(2,2)>>matrix(3,2);
			ss>>matrix(0,3)>>matrix(1,3)>>matrix(2,3)>>matrix(3,3);


			Eigen::Affine3f ff;
			ff.matrix() = matrix;
			origin = ff.translation();
			quat = ff.rotation();
			return true;
		}
		if (type.compare("Vector3f_Quaternionf")==0)
		{
			
			std::stringstream ss (data);
			ss>>origin.x()>>origin.y()>>origin.z();
			ss>>quat.x()>>quat.y()>>quat.z()>>quat.w();
			return true;
		}

		return false;
		
	}
	
	void data_model::setAffine (std::string scanId, Eigen::Matrix4f &matrix)
	{
	

		pt_.put("Model.Transformations."+scanId+".Affine.Type", "matrix4f");
		std::stringstream ss;
		ss << matrix(0,0) << " "<< matrix(1,0) << " "<< matrix(2,0) << " "<< matrix(3,0) << " ";
		ss << matrix(0,1) << " "<< matrix(1,1) << " "<< matrix(2,1) << " "<< matrix(3,1) << " ";
		ss << matrix(0,2) << " "<< matrix(1,2) << " "<< matrix(2,2) << " "<< matrix(3,2) << " ";
		ss << matrix(0,3) << " "<< matrix(1,3) << " "<< matrix(2,3) << " "<< matrix(3,3) << " ";
		//ss << matrix(0,0) << " "<< matrix(1,0) << " "<< matrix(2,0) << " "<< matrix(3,0) << " ";
		//ss << matrix(0,1) << " "<< matrix(1,1) << " "<< matrix(2,1) << " "<< matrix(3,1) << " ";
		//ss << matrix(0,2) << " "<< matrix(1,2) << " "<< matrix(2,2) << " "<< matrix(3,2) << " ";
		//ss << matrix(0,3) << " "<< matrix(1,3) << " "<< matrix(2,3) << " "<< matrix(3,3) << " ";
		pt_.put("Model.Transformations."+scanId+".Affine.Data", ss.str());
		
	}


	void data_model::setAffine (std::string scanId, Eigen::Vector3f origin, Eigen::Quaternionf quat)
	{
		//check if node exists
		addIfNotExists("Model.Transformations."+scanId);

		pt_.put("Model.Transformations."+scanId+".Affine.Type", "Vector3f_Quaternionf");
		std::stringstream ss;
		ss << origin[0] << " "<<origin.x()<< " "<<origin.y()<< " "<<quat.z()<< " "<<quat.x()<< " "<<quat.y()<< " "<<quat.z()<< " "<<quat.w();
		pt_.put("Model.Transformations."+scanId+".Affine.Data", ss.str());
	}
	
	void data_model::addIfNotExists(std::string path)
	{
		boost::optional<boost::property_tree::ptree&> opt_loc_pt_  = pt_.get_child_optional(path);
		
		if (!opt_loc_pt_)
		{
			pt_.add_child(path, boost::property_tree::ptree());
		}
	}
	
	bool data_model::checkIfExists(std::string path)
	{
		boost::optional<boost::property_tree::ptree&> opt_loc_pt_  = pt_.get_child_optional(path);
		if(!opt_loc_pt_){
		    return false;
		}
		return true;
	}
	
	void data_model::setPointcloudName (std::string scanId, std::string fn)
	{
		pt_.put("Model.Transformations."+scanId+".cloudname", fn);
	}
	
	void data_model::getAllScansId(std::vector<std::string> &ids)
	{
		ids.clear();
		BOOST_FOREACH(boost::property_tree::ptree::value_type &v, pt_.get_child("Model.Transformations")) {
			ids.push_back(v.first);
		}

	}

	void data_model::setAlgorithmName(std::string name)
	{
		pt_.put("Model.Algorithms.name", name);
	}
	
	void data_model::addAlgorithmParam(std::string paramName, float paramValue)
	{
		pt_.put<float>("Model.Algorithms.params."+paramName, paramValue);
	}

	void data_model::addAlgorithmParam(std::string paramName, std::string paramValue)
	{
		pt_.put("Model.Algorithms.params."+paramName, paramValue);
	}
	
	void data_model::setResult(std::string resultName, float result)
	{
		pt_.put<float>("Model.Algorithms.results."+resultName, result);
	}

	void data_model::setResult(std::string scanId, std::string resultName, float result)
	{
		pt_.put<float>("Model.Algorithms.results."+scanId+"."+resultName, result);
	}


	
	void data_model::setDataSetPath(std::string path)
	{
		pt_.put("Model.DatasetPath", path);
	}
	void data_model::getDataSetPath(std::string &path)
	{
		try {
			path  = pt_.get<std::string>("Model.DatasetPath");
		}
		catch (...)
		{
			path ="";
		}
	}
	std::string data_model::getFullPathOfPointcloud(std::string id)
	{
		std::string dataPath;
		getDataSetPath(dataPath);
		std::string fileName;
		getPointcloudName(id, fileName);

		boost::filesystem::path t =  boost::filesystem::complete(xmlPath).parent_path();
		t/=boost::filesystem::path(dataPath);
		t/=fileName;
		return t.string();
	}

	void data_model::getResult(std::string scanId, std::string resultName, float &result)
	{
		result = pt_.get<float>("Model.Algorithms.results."+scanId+"."+resultName);
	}
	void data_model::getGlobalModelMatrix(Eigen::Matrix4f &matrix)
	{
		matrix = Eigen::Matrix4f::Identity();
		if (!checkIfExists("Model.GlobalTransformation.Affine.Type")) return ; 
		if (!checkIfExists("Model.GlobalTransformation.Affine.Data")) return ; 

		
		std::string type  = pt_.get<std::string>("Model.GlobalTransformation.Affine.Type");
		std::string data  = pt_.get<std::string>("Model.GlobalTransformation.Affine.Data");
		
		if (type.compare("matrix4f")==0)
		{
			std::stringstream ss (data);
			//column major order
			ss>>matrix(0,0)>>matrix(1,0)>>matrix(2,0)>>matrix(3,0);
			ss>>matrix(0,1)>>matrix(1,1)>>matrix(2,1)>>matrix(3,1);
            ss>>matrix(0,2)>>matrix(1,2)>>matrix(2,2)>>matrix(3,2);
			ss>>matrix(0,3)>>matrix(1,3)>>matrix(2,3)>>matrix(3,3);

			//row major order
			//ss>>matrix(0,0)>>matrix(0,1)>>matrix(0,2)>>matrix(0,3);
			//ss>>matrix(1,0)>>matrix(1,1)>>matrix(1,2)>>matrix(1,3);
			//ss>>matrix(2,0)>>matrix(2,1)>>matrix(2,2)>>matrix(2,3);
			//ss>>matrix(3,0)>>matrix(3,1)>>matrix(3,2)>>matrix(3,3);
			return;
		}
		if (type.compare("Vector3f_Quaternionf")==0)
		{
			Eigen::Quaternionf q;
			Eigen::Vector3f o;
			std::stringstream ss (data);
			ss>>o.x()>>o.y()>>o.z();
			ss>>q.x()>>q.y()>>q.z()>>q.w();
			Eigen::Affine3f affine;
			affine.translate(o);
			affine.rotate(q);
			matrix = affine.matrix();
			return;
		}
	}
	void data_model::setGlobalModelMatrix(Eigen::Matrix4f matrix)
	{
		pt_.put("Model.GlobalTransformation.Affine.Type", "matrix4f");
		std::stringstream ss;
		ss << matrix(0,0) << " "<< matrix(1,0) << " "<< matrix(2,0) << " "<< matrix(3,0) << " ";
		ss << matrix(0,1) << " "<< matrix(1,1) << " "<< matrix(2,1) << " "<< matrix(3,1) << " ";
		ss << matrix(0,2) << " "<< matrix(1,2) << " "<< matrix(2,2) << " "<< matrix(3,2) << " ";
		ss << matrix(0,3) << " "<< matrix(1,3) << " "<< matrix(2,3) << " "<< matrix(3,3) << " ";
		//ss << matrix(0,0) << " "<< matrix(1,0) << " "<< matrix(2,0) << " "<< matrix(3,0) << " ";
		//ss << matrix(0,1) << " "<< matrix(1,1) << " "<< matrix(2,1) << " "<< matrix(3,1) << " ";
		//ss << matrix(0,2) << " "<< matrix(1,2) << " "<< matrix(2,2) << " "<< matrix(3,2) << " ";
		//ss << matrix(0,3) << " "<< matrix(1,3) << " "<< matrix(2,3) << " "<< matrix(3,3) << " ";
		pt_.put("Model.GlobalTransformation.Affine.Data", ss.str());

	}
    void data_model::setTimestamp (std::string scanId, boost::posix_time::ptime ts)
    {
        pt_.put("Model.Transformations."+scanId+".timestamp", boost::posix_time::to_iso_string(ts));
    }

    bool data_model::getTimestamp (std::string scanId, boost::posix_time::ptime &ts)
    {
       if (!checkIfExists("Model.Transformations."+scanId+".timestamp")) return false;
       std::string isoTime;
       pt_.get("Model.Transformations."+scanId+".timestamp", isoTime);
       ts = boost::posix_time::from_iso_string(isoTime);
       return true;
    }

void data_model::setGPS (std::string scanId, double lot, double lan, double att)
{
	//check if node exists
	std::stringstream ss;

	ss<<lot << "\t" <<lan << "\t"<<att;
	pt_.put("Model.Transformations."+scanId+".gps", ss.str());
}

