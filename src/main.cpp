#include "association.h"
#include "color_util.h"
#include <opencv2/opencv.hpp>
#include <json/json.h>
#include <fstream>

std::map<std::string, Camera> ParseCameras(const std::string& filename)
{
	Json::Value json;
	std::ifstream fs(filename);
	if (!fs.is_open()) {
		std::cerr << "json file not exist: " << filename << std::endl;
		std::abort();
	}

	std::string errs;
	Json::parseFromStream(Json::CharReaderBuilder(), fs, &json, &errs);
	fs.close();

	if (errs != "") {
		std::cerr << "json read file error: " << errs << std::endl;
		std::abort();
	}

	std::map<std::string, Camera> cameras;
	for (auto camIter = json.begin(); camIter != json.end(); camIter++) {
		Camera camera;

		for (int i = 0; i < 9; i++)
			camera.K(i / 3, i % 3) = (*camIter)["K"][i].asFloat();

		Eigen::Vector3f r;
		for (int i = 0; i < 3; i++)
			r(i) = (*camIter)["R"][i].asFloat();
		camera.R = Eigen::AngleAxisf(r.norm(), r.normalized()).matrix();

		for (int i = 0; i < 3; i++)
			camera.T(i) = (*camIter)["T"][i].asFloat();

		camera.Update();
		cameras.insert(std::make_pair(camIter.key().asString(), camera));
	}
	return cameras;
}


std::vector<std::vector<SkelDetection>> ParseDetections(const std::string& filename)
{
	std::ifstream fs(filename);
	if (!fs.is_open()) {
		std::cerr << "json file not exist: " << filename << std::endl;
		std::abort();
	}

	int frameSize, camSize;
	fs >> frameSize >> camSize;

	std::vector<std::vector<SkelDetection>> detections(frameSize, std::vector<SkelDetection>(camSize));
	for (int frameIdx = 0; frameIdx < frameSize; frameIdx++) {
		for (int camIdx = 0; camIdx < camSize; camIdx++) {
			SkelDetection& detection = detections[frameIdx][camIdx];
			for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
				int jSize;
				fs >> jSize;
				detection.joints[jIdx].resize(3, jSize);
				for (int i = 0; i < 3; i++)
					for (int j = 0; j < jSize; j++)
						fs >> detection.joints[jIdx](i, j);
			}
			for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
				const int jAIdx = GetSkelDef().pafDict(0, pafIdx);
				const int jBIdx = GetSkelDef().pafDict(1, pafIdx);
				detection.pafs[pafIdx].resize(detection.joints[jAIdx].cols(), detection.joints[jBIdx].cols());
				for (int i = 0; i < detection.pafs[pafIdx].rows(); i++)
					for (int j = 0; j < detection.pafs[pafIdx].cols(); j++)
						fs >> detection.pafs[pafIdx](i, j);
			}
		}
	}
	fs.close();
	return detections;
}

std::vector<Person3D> SafeguardPerson3D(const std::vector<Person3D>& lastperson3D)
{
	std::vector<Person3D> m_lastperson3D = lastperson3D;
	for (auto person = m_lastperson3D.begin(); person != m_lastperson3D.end();) {
		int jointsnum = 0;
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
			if (person->joints(3, jIdx) > FLT_EPSILON)
				jointsnum++;
		}
		if (jointsnum < (GetSkelDef().jointSize - 6 )) m_lastperson3D.erase(person);
		else person++;
	}
	return m_lastperson3D;
}



void SaveResult(const int& frameIdx, const std::vector<cv::Mat>& images, const std::vector<SkelDetection>& detections, const std::vector<Camera>& cameras,
	const std::vector<std::vector<Person2D>>& persons2D, const std::vector<Person3D>& persons3D)
{
	const int rows = 2;
	const int cols = (cameras.size() + rows - 1) / rows;
	const Eigen::Vector2i imgSize(images.begin()->cols, images.begin()->rows);
	const int jointRadius = round(imgSize.x() *  1.f / 128.f);
	const int pafThickness = round(imgSize.x() * 1.f / 256.f);
	const float textScale = sqrtf(imgSize.x() / 512.f);
	cv::Mat detectImg(rows*imgSize.y(), cols*imgSize.x(), CV_8UC3);
	cv::Mat assocImg(rows*imgSize.y(), cols*imgSize.x(), CV_8UC3);
	cv::Mat reprojImg(rows*imgSize.y(), cols*imgSize.x(), CV_8UC3);

	for (int camIdx = 0; camIdx < cameras.size(); camIdx++) {
		cv::Rect roi(camIdx%cols * imgSize.x(), camIdx / cols * imgSize.y(), imgSize.x(), imgSize.y());
		images[camIdx].copyTo(detectImg(roi));
		images[camIdx].copyTo(assocImg(roi));
		images[camIdx].copyTo(reprojImg(roi));

		// draw detection
		const SkelDetection& detection = detections[camIdx];
		for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
			const int jaIdx = GetSkelDef().pafDict(0, pafIdx);
			const int jbIdx = GetSkelDef().pafDict(1, pafIdx);
			for (int jaCandiIdx = 0; jaCandiIdx < detection.joints[jaIdx].cols(); jaCandiIdx++) {
				for (int jbCandiIdx = 0; jbCandiIdx < detection.joints[jbIdx].cols(); jbCandiIdx++) {
					if (detection.joints[jaIdx](2, jaCandiIdx) > 0.f && detection.joints[jbIdx](2, jbCandiIdx) > 0.f) {
						const int thickness = round(detection.pafs[pafIdx](jaCandiIdx, jbCandiIdx)* pafThickness);
						if (thickness > 0) {
							const cv::Point jaPos(round(detection.joints[jaIdx](0, jaCandiIdx)*imgSize.x() - 0.5f), round(detection.joints[jaIdx](1, jaCandiIdx)*imgSize.y() - 0.5f));
							const cv::Point jbPos(round(detection.joints[jbIdx](0, jbCandiIdx)*imgSize.x() - 0.5f), round(detection.joints[jbIdx](1, jbCandiIdx)*imgSize.y() - 0.5f));
							cv::line(detectImg(roi), jaPos, jbPos, ColorUtil::GetColor("gray"), thickness);
						}
					}
				}
			}
		}
			
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
			for (int candiIdx = 0; candiIdx < detection.joints[jIdx].cols(); candiIdx++) {
				const cv::Point jPos(round(detection.joints[jIdx](0, candiIdx)*imgSize.x() - 0.5f), round(detection.joints[jIdx](1, candiIdx)*imgSize.y() - 0.5f));
				const int radius = round(detection.joints[jIdx](2, candiIdx) * jointRadius);
				if (radius > 0) {
					cv::circle(detectImg(roi), jPos, radius, ColorUtil::GetColor("gray"), 1);
					cv::putText(detectImg(roi), std::to_string(jIdx), jPos, cv::FONT_HERSHEY_PLAIN, textScale, ColorUtil::GetColor("gray"));
				}
			}
		}

		// draw assoc
		for (int pIdx = 0; pIdx < persons2D[camIdx].size(); pIdx++) {
			const Person2D& person = persons2D[camIdx][pIdx];
			const cv::Scalar& color = ColorUtil::GetColor(pIdx);
			for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
				if (person.joints(2, jIdx) < FLT_EPSILON)
					continue;
				cv::Point jPos(round(person.joints(0, jIdx)*imgSize.x() - 0.5f), round(person.joints(1, jIdx)*imgSize.y() - 0.5f));
				cv::circle(assocImg(roi), jPos, jointRadius, color, -1);
				cv::putText(assocImg(roi), std::to_string(jIdx), jPos, cv::FONT_HERSHEY_PLAIN, textScale, color);
			}

			for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
				const int jaIdx = GetSkelDef().pafDict(0, pafIdx);
				const int jbIdx = GetSkelDef().pafDict(1, pafIdx);
				if (person.joints(2, jaIdx) > FLT_EPSILON && person.joints(2, jbIdx) > FLT_EPSILON) {
					const cv::Point jaPos(round(person.joints(0, jaIdx)*imgSize.x() - 0.5f), round(person.joints(1, jaIdx)*imgSize.y() - 0.5f));
					const cv::Point jbPos(round(person.joints(0, jbIdx)*imgSize.x() - 0.5f), round(person.joints(1, jbIdx)*imgSize.y() - 0.5f));
					cv::line(assocImg(roi), jaPos, jbPos, color, pafThickness);
				}
			}
		}

		// draw proj
		for (int pIdx = 0; pIdx < persons3D.size(); pIdx++) {
			const Person2D& person = persons3D[pIdx].ProjSkel(cameras[camIdx].proj);
			const cv::Scalar& color = ColorUtil::GetColor(pIdx);
			for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
				if (person.joints(2, jIdx) < FLT_EPSILON)
					continue;
				cv::Point jPos(round(person.joints(0, jIdx)*imgSize.x() - 0.5f), round(person.joints(1, jIdx)*imgSize.y() - 0.5f));
				cv::circle(reprojImg(roi), jPos, jointRadius, color, -1);
				cv::putText(reprojImg(roi), std::to_string(jIdx), jPos, cv::FONT_HERSHEY_PLAIN, textScale, color);
			}

			for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
				const int jaIdx = GetSkelDef().pafDict(0, pafIdx);
				const int jbIdx = GetSkelDef().pafDict(1, pafIdx);
				if (person.joints(2, jaIdx) > FLT_EPSILON && person.joints(2, jbIdx) > FLT_EPSILON) {
					const cv::Point jaPos(round(person.joints(0, jaIdx)*imgSize.x() - 0.5f), round(person.joints(1, jaIdx)*imgSize.y() - 0.5f));
					const cv::Point jbPos(round(person.joints(0, jbIdx)*imgSize.x() - 0.5f), round(person.joints(1, jbIdx)*imgSize.y() - 0.5f));
					cv::line(reprojImg(roi), jaPos, jbPos, color, pafThickness);
				}
			}
		}
	}
	//cv::imwrite("../debug/detect/" + std::to_string(frameIdx) + ".jpg", detectImg);
	//cv::imwrite("../debug/assoc/" + std::to_string(frameIdx) + ".jpg", assocImg);
	cv::imwrite("../debug/reproj/" + std::to_string(frameIdx) + ".jpg", reprojImg);
}



int main()
{
	// load data and detections
	std::map<std::string, Camera> _cameras = ParseCameras("../data/calibration.json");
	std::vector<Camera> cameras;
	std::vector<cv::Mat> rawImgs;
	std::vector<cv::VideoCapture> videos;
	for (const auto& iter : _cameras) {
		cameras.emplace_back(iter.second);
		videos.emplace_back(cv::VideoCapture("../data/" + iter.first + ".mpeg"));
		rawImgs.emplace_back(cv::Mat());
	}
	std::cout << "init data" << std::endl; 
	std::vector<std::vector<SkelDetection>> detections = ParseDetections("../data/detection.txt");
	
	// init
	Associater associater(cameras);
	std::vector<Person3D> lastperson;


	//第0帧处理
		for (int camIdx = 0; camIdx < cameras.size(); camIdx++)
			videos[camIdx] >> rawImgs[camIdx];

		associater.SetDetection(detections[0]);//导入关键点的识别结果  frameIdx 帧数  camIdx 机位
		associater.ConstructJointRays();//根据关键点在当前机位的投影构造三维投影线
		associater.ConstructJointEpiEdges();//计算相同种类之间连线的距离
		associater.ClusterPersons2D();//确定2D人（利用关键点的连接）
		associater.ProposalCollocation();//生成所有2D人连接成3D人的可能
		associater.ClusterPersons3D();
		associater.ConstructPersons();
		lastperson = SafeguardPerson3D(associater.GetPersons3D());
		std::cout << std::to_string(0) << std::endl;
		SaveResult(0, rawImgs, detections[0], cameras, associater.GetPersons2D(), associater.GetPersons3D());
	

	for (int frameIdx = 1; frameIdx < detections.size(); frameIdx++) {
		for (int camIdx = 0; camIdx < cameras.size(); camIdx++)
			videos[camIdx] >> rawImgs[camIdx];

		associater.SetDetection(detections[frameIdx]);//导入关键点的识别结果  frameIdx 帧数  camIdx 机位
		associater.ConstructJointRays();//根据关键点在当前机位的投影构造三维投影线
		associater.ConstructJointEpiEdges();//计算相同种类之间连线的距离
		associater.FindTrackingPerson3D(lastperson);
		associater.ClusterPersons2D();//确定2D人（利用关键点的连接）
		associater.ProposalCollocation();//生成所有2D人连接成3D人的可能
		associater.ClusterPersons3D();
		associater.ConstructPersons();
		lastperson = SafeguardPerson3D(associater.GetPersons3D());
		std::cout << std::to_string(frameIdx) << std::endl;
		SaveResult(frameIdx, rawImgs, detections[frameIdx], cameras, associater.GetPersons2D(), associater.GetPersons3D());
	}
	return 0;
}
