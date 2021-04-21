#include "association.h"
#include "math_util.h"
#include "color_util.h"
#include <iostream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>


Associater::Associater(const std::vector<Camera>& _cams)
{
	m_cams = _cams;

	m_wTracking = 500.f;
	m_trackThresh = 0.001f;
	m_epiThresh = 0.2f;
	m_wEpi = 1.f;
	m_wView = 4.f;
	m_wPaf = 1.f;
	m_cPlaneTheta = 2e-3f;
	m_cViewCnt = 2.f;
	m_triangulateThresh = 0.05f;

	m_persons2D.resize(m_cams.size());
	m_personsMapByView.resize(m_cams.size());
	m_assignMap.resize(m_cams.size(), std::vector<Eigen::VectorXi>(GetSkelDef().jointSize));
	m_jointRays.resize(m_cams.size(), std::vector<Eigen::Matrix3Xf>(GetSkelDef().jointSize));
	m_jointEpiEdges.resize(GetSkelDef().jointSize);
	for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) 
		m_jointEpiEdges[jIdx].resize(m_cams.size(), std::vector<Eigen::MatrixXf>(m_cams.size()));
}


void Associater::SetDetection(const std::vector<SkelDetection>& detections)
{
	assert(m_cams.size() == detections.size());
	m_detections = detections;

	// reset assign map
	for (int view = 0; view < m_cams.size(); view++)
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
			m_assignMap[view][jIdx].setConstant(m_detections[view].joints[jIdx].cols(), -1);

	for (int view = 0; view < m_cams.size(); view++) {
		m_persons2D[view].clear();
		m_personsMapByView[view].clear();
	}
	m_persons3D.clear();

}


void Associater::ConstructJointRays()//根据关键点在当前机位的投影构造三维投影方向
{
	for (int view = 0; view < m_cams.size(); view++) {//所有机位  恒为5
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {//关键点数  恒为19
			const Eigen::Matrix3Xf& joints = m_detections[view].joints[jIdx];
			m_jointRays[view][jIdx].resize(3, joints.cols());
			for (int jCandiIdx = 0; jCandiIdx < joints.cols(); jCandiIdx++)
				m_jointRays[view][jIdx].col(jCandiIdx) = m_cams[view].CalcRay(joints.block<2, 1>(0, jCandiIdx));//
		}
	}
}


void Associater::ConstructJointEpiEdges()//背景3，有效值[0,1），无效值-1,0表最近
{
	for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {//对于每一种关键点
		for (int viewA = 0; viewA < m_cams.size() - 1; viewA++) {
			for (int viewB = viewA + 1; viewB < m_cams.size(); viewB++) {//针对两个相机之间
				Eigen::MatrixXf& epi = m_jointEpiEdges[jIdx][viewA][viewB];//存储两个相机中相同参数的距离约化值
				const Eigen::Matrix3Xf& jointsA = m_detections[viewA].joints[jIdx];//A相机中对应关键点序号的点
				const Eigen::Matrix3Xf& jointsB = m_detections[viewB].joints[jIdx];//B相机中
				const Eigen::Matrix3Xf& raysA = m_jointRays[viewA][jIdx];//投影方向
				const Eigen::Matrix3Xf& raysB = m_jointRays[viewB][jIdx];
				epi.setConstant(jointsA.cols(), jointsB.cols(), -1.f);
				for (int jaCandiIdx = 0; jaCandiIdx < epi.rows(); jaCandiIdx++) {
					for (int jbCandiIdx = 0; jbCandiIdx < epi.cols(); jbCandiIdx++) {
						const float dist = MathUtil::Line2LineDist(
							m_cams[viewA].pos, raysA.col(jaCandiIdx), m_cams[viewB].pos, raysB.col(jbCandiIdx));//计算距离
						if (dist < m_epiThresh)//如果距离小于0.2f则视为有效
							epi(jaCandiIdx, jbCandiIdx) = dist / m_epiThresh;
					}
				}
				m_jointEpiEdges[jIdx][viewB][viewA] = epi.transpose();//转置
			}
		}
	}
}

void Associater::FindTrackingPerson3D(const std::vector<Person3D>& lastperson3d)
{
	m_personsMapByIdx.clear();
	std::vector<Eigen::VectorXi> resizemodel(m_cams.size(), Eigen::VectorXi::Constant(GetSkelDef().jointSize, -1));
	m_personsMapByIdx.resize(lastperson3d.size(), resizemodel);
	for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
		TrackingProposalCollocation(lastperson3d, jIdx);
		ClusterTrackingPerson3D(lastperson3d, jIdx);
	}

	for (int pIdx = 0; pIdx < m_personsMapByIdx.size(); pIdx++) {
		for (int camIdx = 0; camIdx < m_cams.size(); camIdx++) {
			const Person2D& lastperson = lastperson3d[pIdx].ProjSkel(m_cams[camIdx].proj);
			for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {//扩展找点
				if (m_personsMapByIdx[pIdx][camIdx][jIdx] == -1) {
					for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
						if (jIdx == GetSkelDef().pafDict(0, pafIdx) || jIdx == GetSkelDef().pafDict(1, pafIdx)) {
							int jbIdx = (jIdx == GetSkelDef().pafDict(0, pafIdx) ? GetSkelDef().pafDict(1, pafIdx) : GetSkelDef().pafDict(0, pafIdx));
							float minpafScore = -1.f;
							for (int jCandiIdx = 0; jCandiIdx < m_detections[camIdx].joints[jIdx].cols() && (m_personsMapByIdx[pIdx][camIdx][jbIdx] != -1); jCandiIdx++) {
								if (minpafScore < (jIdx == GetSkelDef().pafDict(0, pafIdx) ? m_detections[camIdx].pafs[pafIdx](jCandiIdx, m_personsMapByIdx[pIdx][camIdx][jbIdx]) : m_detections[camIdx].pafs[pafIdx](m_personsMapByIdx[pIdx][camIdx][jbIdx], jCandiIdx))) {
									minpafScore = (jIdx == GetSkelDef().pafDict(0, pafIdx) ? m_detections[camIdx].pafs[pafIdx](jCandiIdx, m_personsMapByIdx[pIdx][camIdx][jbIdx]) : m_detections[camIdx].pafs[pafIdx](m_personsMapByIdx[pIdx][camIdx][jbIdx], jCandiIdx));
									if (minpafScore > 0 && m_detections[camIdx].joints[jIdx](2, jCandiIdx) != -1) m_personsMapByIdx[pIdx][camIdx][jIdx] = jCandiIdx;
									const float dist = MathUtil::Point2PointDistSquare(
										lastperson.joints.col(jIdx), m_detections[camIdx].joints[jIdx].col(jCandiIdx));
									if (dist > 10 * m_trackThresh)  m_personsMapByIdx[pIdx][camIdx][jIdx] = -1;
								}
							}
						}
					}
					if (m_personsMapByIdx[pIdx][camIdx][jIdx] != -1) {
						m_detections[camIdx].joints[jIdx](2, m_personsMapByIdx[pIdx][camIdx][jIdx]) = -1;
					}
				}
			}
		}
	}
}

void Associater::FindPossibleJoint(const std::vector<Person3D>& lastperson3d)
{
	m_personsMapByIdx.clear();

	for (int pIdx = 0; pIdx < lastperson3d.size(); pIdx++) {//遍历所有上一时刻的3D人
		std::vector<Eigen::VectorXi> personMap(m_cams.size(), Eigen::VectorXi::Constant(GetSkelDef().jointSize, -1));
		for (int camIdx = 0; camIdx < m_cams.size(); camIdx++) {//遍历每一个相机镜头
			const Person2D& lastperson = lastperson3d[pIdx].ProjSkel(m_cams[camIdx].proj);//建立当前相机面上的上一时刻的2D人
			std::vector<Eigen::VectorXi> assignMap;
			assignMap.resize(GetSkelDef().jointSize);//这个表示了关键点是否已经拥有归属
			for (int jIdx = 0; jIdx < assignMap.size(); jIdx++)
				assignMap[jIdx].setConstant(m_detections[camIdx].joints[jIdx].cols(), -1);//开始时所有点的归属均为-1，即不存在


			for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {//遍历该人的每一个关键节点
				if (lastperson.joints(2, jIdx) < FLT_EPSILON) {//若生成2D人对应关键点的概率过小（即原3D人的概率过小），则在平面上不进行寻找
					personMap[camIdx][jIdx] = -1;
					continue;
				}
				float distmin = FLT_MAX;//寻找距离最小值
				int realjCandiIdx = -1;
				for (int jCandiIdx = 0; jCandiIdx < m_detections[camIdx].joints[jIdx].cols(); jCandiIdx++){
					const float dist = MathUtil::Point2PointDistSquare(
						lastperson.joints.col(jIdx), m_detections[camIdx].joints[jIdx].col(jCandiIdx));
					if (dist < distmin) {
						distmin = dist;
                        realjCandiIdx = jCandiIdx;
					}
				}
				personMap[camIdx][jIdx] = distmin > m_trackThresh ? -1 : realjCandiIdx;
				if (distmin <= m_trackThresh) {
					m_detections[camIdx].joints[jIdx](2, realjCandiIdx) = -1;
					assignMap[jIdx][realjCandiIdx] = pIdx;
				}
			}


			for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {//扩展找点
				if (personMap[camIdx][jIdx] == -1) {
					for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
						if (jIdx == GetSkelDef().pafDict(0, pafIdx) || jIdx == GetSkelDef().pafDict(1, pafIdx)) {
							int jbIdx = (jIdx == GetSkelDef().pafDict(0, pafIdx) ? GetSkelDef().pafDict(1, pafIdx) : GetSkelDef().pafDict(0, pafIdx));
							float minpafScore = -1.f;
							for (int jCandiIdx = 0; jCandiIdx < m_detections[camIdx].joints[jIdx].cols() && (personMap[camIdx][jbIdx] != -1); jCandiIdx++) {
								if (minpafScore < (jIdx == GetSkelDef().pafDict(0, pafIdx) ? m_detections[camIdx].pafs[pafIdx](jCandiIdx, personMap[camIdx][jbIdx]) : m_detections[camIdx].pafs[pafIdx](personMap[camIdx][jbIdx], jCandiIdx))) {
									minpafScore = (jIdx == GetSkelDef().pafDict(0, pafIdx) ? m_detections[camIdx].pafs[pafIdx](jCandiIdx, personMap[camIdx][jbIdx]) : m_detections[camIdx].pafs[pafIdx](personMap[camIdx][jbIdx], jCandiIdx));
								    if(minpafScore > 0 && assignMap[jIdx][jCandiIdx] != -1 ) personMap[camIdx][jIdx] = jCandiIdx;
									const float dist = MathUtil::Point2PointDistSquare(
										lastperson.joints.col(jIdx), m_detections[camIdx].joints[jIdx].col(jCandiIdx));
									if (dist > 10 * m_trackThresh)  personMap[camIdx][jIdx] = -1;
								}
							}	
						}
					}
					if (personMap[camIdx][jIdx] != -1) {
						std::cout << "changed successfully" << std::endl;
						m_detections[camIdx].joints[jIdx](2, personMap[camIdx][jIdx]) = -1;
					}
				}
				
			}
		}
		m_personsMapByIdx.emplace_back(personMap);
	}
}



void Associater::ClusterPersons2D(const SkelDetection& detection, std::vector<Person2D>& persons, std::vector<Eigen::VectorXi>& assignMap)
{
	persons.clear();//将人先清空

	// generate valid pafs
	std::vector<std::tuple<float, int, int, int>> pafSet;//连接概率，边序号，A相机中    
	for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {//边序号（0-18）
		const int jaIdx = GetSkelDef().pafDict(0, pafIdx);
		const int jbIdx = GetSkelDef().pafDict(1, pafIdx);//从边字典中取出jaIdx，jbIdx构成端点，固定遍历
		for (int jaCandiIdx = 0; jaCandiIdx < detection.joints[jaIdx].cols(); jaCandiIdx++) {//遍历边的一头中对应关键点序号为jaIdx的点，一般遍历人数次
			for (int jbCandiIdx = 0; jbCandiIdx < detection.joints[jbIdx].cols(); jbCandiIdx++) {//遍历边的另一头中对应关键点序号为jbIdx的点，一般遍历人数次
				const float jaScore = detection.joints[jaIdx](2, jaCandiIdx);
				const float jbScore = detection.joints[jbIdx](2, jbCandiIdx);//取出对应相机中对应点的单图连接概率
				const float pafScore = detection.pafs[pafIdx](jaCandiIdx, jbCandiIdx);//这两个点连接起来的概率
				if (jaScore > 0.f && jbScore > 0.f && pafScore > 0.f)//若都大于0则可加入可能的边集
					pafSet.emplace_back(std::make_tuple(pafScore, pafIdx, jaCandiIdx, jbCandiIdx));
			}
		}
	}
	std::sort(pafSet.rbegin(), pafSet.rend());

	// construct bodies use minimal spanning tree
	assignMap.resize(GetSkelDef().jointSize);//这个表示了A类关键点的第B个属于的人
	for (int jIdx = 0; jIdx < assignMap.size(); jIdx++)
		assignMap[jIdx].setConstant(detection.joints[jIdx].cols(), -1);//开始时所有点的归属均为-1，即不存在

	for (const auto& paf : pafSet) {//概率最大边开始枚举
		const float pafScore = std::get<0>(paf);//连边概率
		const int pafIdx = std::get<1>(paf);//边序号
		const int jaCandiIdx = std::get<2>(paf);//边的一头下对应端点序号，一般小于等于人数
		const int jbCandiIdx = std::get<3>(paf);//边的另一头下对应端点
		const int jaIdx = GetSkelDef().pafDict(0, pafIdx);//始边关键点序号，0-18，查表
		const int jbIdx = GetSkelDef().pafDict(1, pafIdx);//终边关键点序号，0-18，查表

		int& aAssign = assignMap[jaIdx][jaCandiIdx];
		int& bAssign = assignMap[jbIdx][jbCandiIdx];

		// 1. A & B not assigned yet: Create new person
		//当前两个点均没有人对应，则建立新人
		if (aAssign == -1 && bAssign == -1) {
			Person2D person;
			person.joints.col(jaIdx) = detection.joints[jaIdx].col(jaCandiIdx);
			person.joints.col(jbIdx) = detection.joints[jbIdx].col(jbCandiIdx);
			person.pafs(pafIdx) = pafScore;
			aAssign = bAssign = persons.size();
			persons.emplace_back(person);
		}

		// 2. A assigned but not B: Add B to person with A (if no another B there) 
		// 3. B assigned but not A: Add A to person with B (if no another A there)
		//仅有一个点对应了人，那么将另一个点尝试归化至该人
		else if ((aAssign >= 0 && bAssign == -1) || (aAssign == -1 && bAssign >= 0)) {
			const int assigned = aAssign >= 0 ? aAssign : bAssign;
			int& unassigned = aAssign >= 0 ? bAssign : aAssign;
			const int unassignedIdx = aAssign >= 0 ? jbIdx : jaIdx;
			const int unassignedCandiIdx = aAssign >= 0 ? jbCandiIdx : jaCandiIdx;

			Person2D& person = persons[assigned];
			if (person.joints(2, unassignedIdx) < FLT_EPSILON) {//连接着的那棵树上是否缺少未被连接的点对应的种类，缺少就连接
				person.joints.col(unassignedIdx) = detection.joints[unassignedIdx].col(unassignedCandiIdx);
				person.pafs(pafIdx) = pafScore;
				unassigned = assigned;
			}
		}

		// 4. A & B already assigned to same person (circular/redundant PAF)
		//对应连接匹配，则只需要修改该人对应该线段的概率值
		//当前骨架dict没有自环，此种情况不会出现
		else if (aAssign == bAssign)
			persons[aAssign].pafs(pafIdx) = pafScore;

		// 5. A & B already assigned to different people: Merge people if key point intersection is null
		//对应连接不匹配，尝试将人进行合并
		else {
			const int assignFst = aAssign < bAssign ? aAssign : bAssign;//第一个人的编号（较小）
			const int assignSec = aAssign < bAssign ? bAssign : aAssign;//第二个人的编号（较大）
			Person2D& personFst = persons[assignFst];//第一个人
			const Person2D& personSec = persons[assignSec];//第二个人

			bool conflict = false;//冲突标记
			for (int jIdx = 0; jIdx < GetSkelDef().jointSize && !conflict; jIdx++)//不冲突情况下对关键点循环
				conflict |= (personFst.joints(2, jIdx) > 0.f && personSec.joints(2, jIdx) > 0.f);//两个人的对应点位都大于0则冲突

			if (!conflict) {//如果不冲突，则可以消灭第二个人
				for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
					if (personSec.joints(2, jIdx) > 0.f)
						personFst.joints.col(jIdx) = personSec.joints.col(jIdx);

				persons.erase(persons.begin() + assignSec);//删掉第二个人的全过程
				for (Eigen::VectorXi& tmp : assignMap) {
					for (int i = 0; i < tmp.size(); i++) {
						if (tmp[i] == assignSec)
							tmp[i] = assignFst;
						else if (tmp[i] > assignSec)
							tmp[i]--;
					}
				}
			}
		}
	}

	// filter
	//过滤器，将节点太少的人删掉
	const int jcntThresh = round(0.5f * GetSkelDef().jointSize);//一半的点数
	for (auto person = persons.begin(); person != persons.end();) {//对暂时的人循环
		if (person->GetJointCnt() < jcntThresh ) {//如果该人身上的关键点数没有达到要求则删除
			const int personIdx = person - persons.begin();
			for (Eigen::VectorXi& tmp : assignMap) {
				for (int i = 0; i < tmp.size(); i++) {
					if (tmp[i] == personIdx)
						tmp[i] = -1;
					else if (tmp[i] > personIdx)
						tmp[i]--;
				}
			}
			person = persons.erase(person);
		}
		else
			person++;//到达要求则下一个
	}
}


void Associater::ClusterPersons2D()
{
	// cluster 2D
	for (int view = 0; view < m_cams.size(); view++) {
		std::vector<Eigen::VectorXi> assignMap;
		std::vector<Person2D> persons2D;
		ClusterPersons2D(m_detections[view], persons2D, assignMap);
		m_personsMapByView[view].resize(persons2D.size(), Eigen::VectorXi::Constant(GetSkelDef().jointSize, -1));
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
			for (int candiIdx = 0; candiIdx < assignMap[jIdx].size(); candiIdx++) {
				const int pIdx = assignMap[jIdx][candiIdx];
				if (pIdx >= 0)
					m_personsMapByView[view][pIdx][jIdx] = candiIdx;
			}
		}
	}
}

void Associater::TrackingProposalCollocation(const std::vector<Person3D>& lastperson3d, const int& jIdx)
{
	std::function<void(const Eigen::VectorXi&, const int&, std::vector<Eigen::VectorXi>&)> Proposal
		= [&Proposal](const Eigen::VectorXi& candiCnt, const int& k, std::vector<Eigen::VectorXi>& proposals) {
		if (k == candiCnt.size()) {
			return;
		}//k如果是candiCnt的大小
		else if (k == 0) {
			proposals = std::vector<Eigen::VectorXi>(candiCnt[k] + 1, Eigen::VectorXi::Constant(candiCnt.size(), -1));
			for (int i = 0; i < candiCnt[k] + 1; i++)
				proposals[i][candiCnt.size() - 1] = 0;
			for (int i = 0; i < candiCnt[k]; i++)
				proposals[i + 1][k] = i;
			Proposal(candiCnt, k + 1, proposals);
		}
		else if (k == candiCnt.size() - 1) {
			std::vector<Eigen::VectorXi> proposalsBefore = proposals;
			for (int i = 1; i < candiCnt[k]; i++) {
				std::vector<Eigen::VectorXi> _proposals = proposalsBefore;
				for (auto&& _proposal : _proposals)
					_proposal[k] = i;
				proposals.insert(proposals.end(), _proposals.begin(), _proposals.end());
			}
			Proposal(candiCnt, k + 1, proposals);
		}
		else {
			std::vector<Eigen::VectorXi> proposalsBefore = proposals;
			for (int i = 0; i < candiCnt[k]; i++) {
				std::vector<Eigen::VectorXi> _proposals = proposalsBefore;
				for (auto&& _proposal : _proposals)
					_proposal[k] = i;
				proposals.insert(proposals.end(), _proposals.begin(), _proposals.end());
			}
			Proposal(candiCnt, k + 1, proposals);
		}
	};

	m_trackingpersonProposals.clear();
	Eigen::VectorXi candiCnt(m_cams.size()+1);
	for (int view = 0; view < m_cams.size(); view++)//相机个数遍历
		candiCnt[view] = int(m_detections[view].joints[jIdx].cols());
	candiCnt[m_cams.size()] = int(lastperson3d.size());
	Proposal(candiCnt, 0, m_trackingpersonProposals);
}

void Associater::ProposalCollocation()//将所有可能的情况计入personproposals中
{
	// proposal persons
	std::function<void(const Eigen::VectorXi&, const int&, std::vector<Eigen::VectorXi>&)> Proposal
		= [&Proposal](const Eigen::VectorXi& candiCnt, const int& k, std::vector<Eigen::VectorXi>& proposals) {
		if (k == candiCnt.size()) {
			return;
		}//k如果是candiCnt的大小
		else if (k == 0) {
			proposals = std::vector<Eigen::VectorXi>(candiCnt[k] + 1, Eigen::VectorXi::Constant(candiCnt.size(), -1));
			for (int i = 0; i < candiCnt[k]; i++)
				proposals[i + 1][k] = i;
			Proposal(candiCnt, k + 1, proposals);
		}
		else {
			std::vector<Eigen::VectorXi> proposalsBefore = proposals;
			for (int i = 0; i < candiCnt[k]; i++) {
				std::vector<Eigen::VectorXi> _proposals = proposalsBefore;
				for (auto&& _proposal : _proposals)
					_proposal[k] = i;
				proposals.insert(proposals.end(), _proposals.begin(), _proposals.end());
			}
			Proposal(candiCnt, k + 1, proposals);
		}
	};

	m_personProposals.clear();
	Eigen::VectorXi candiCnt(m_cams.size());
	for (int view = 0; view < m_cams.size(); view++)//相机个数遍历
		candiCnt[view] = int(m_personsMapByView[view].size());
	Proposal(candiCnt, 0, m_personProposals);
}

float Associater::CalcTrackingJointsLoss(const std::vector<Person3D>& lastperson3d, const int& jIdx, const int& trackingProposalIdx)
{
	const Eigen::VectorXi& proposal = m_trackingpersonProposals[trackingProposalIdx];
	bool valid = (proposal.array() >= 0).count() > 1;
	if (!valid)
		return -1.f;

	float loss = 0.f;

	//epiloss
	std::vector<float> epiLosses;
	for (int viewA = 0; viewA < m_cams.size() - 1 && valid; viewA++) {
		if (proposal[viewA] == -1)
			continue;
		//const Eigen::VectorXi& personMapA = m_personsMapByView[viewA][proposal[viewA]];
		for (int viewB = viewA + 1; viewB < m_cams.size() && valid; viewB++) {
			if (proposal[viewB] == -1)
				continue;
			//const Eigen::VectorXi& personMapB = m_personsMapByView[viewB][proposal[viewB]];
		    const float edge = m_jointEpiEdges[jIdx][viewA][viewB](proposal[viewA], proposal[viewB]);
		    if (edge < 0.f)
				valid = false;
			else
				epiLosses.emplace_back(edge);
		}
	}
	if (!valid)
		return -1.f;

	if (epiLosses.size() > 0)
		loss += m_wEpi * std::accumulate(epiLosses.begin(), epiLosses.end(), 0.f) / float(epiLosses.size());

	//trackingloss
	std::vector<float> trackingLosses;
	for (int view = 0; view < m_cams.size(); view++) {
		const Person2D& lastperson = lastperson3d[proposal[m_cams.size()]].ProjSkel(m_cams[view].proj);
		if (lastperson.joints(2, jIdx) < FLT_EPSILON) valid = false;
		else {
			if (proposal[view] == -1) trackingLosses.emplace_back(  m_trackThresh);
			else {
                const float dist = MathUtil::Point2PointDistSquare(
				    lastperson.joints.col(jIdx), m_detections[view].joints[jIdx].col(proposal[view]));
				if (dist > 5 * m_trackThresh) return -1.f;
				else trackingLosses.emplace_back(dist);
			}
		}
	}

	if (!valid)
		return -1.f;

	if (trackingLosses.size() > 0)
		loss += m_wTracking * std::accumulate(trackingLosses.begin(), trackingLosses.end(), 0.f) / float(trackingLosses.size());

	//viewloss
	loss += m_wView * (1.f - MathUtil::Welsch(m_cViewCnt, (proposal.array() >= 0).count()-(proposal[m_cams.size()] >= 0 )));
	return loss;
}

float Associater::CalcProposalLoss(const int& personProposalIdx)
{
	const Eigen::VectorXi& proposal = m_personProposals[personProposalIdx];
	bool valid = (proposal.array() >= 0).count() > 1;
	if (!valid)
		return -1.f;

	float loss = 0.f;

	//joint loss
	std::vector<float> epiLosses;
	for (int viewA = 0; viewA < m_cams.size() - 1 && valid; viewA++) {
		if (proposal[viewA] == -1)
			continue;
		const Eigen::VectorXi& personMapA = m_personsMapByView[viewA][proposal[viewA]];
		for (int viewB = viewA + 1; viewB < m_cams.size() && valid; viewB++) {
			if (proposal[viewB] == -1)
				continue;
			const Eigen::VectorXi& personMapB = m_personsMapByView[viewB][proposal[viewB]];

			for (int jIdx = 0; jIdx < GetSkelDef().jointSize && valid; jIdx++) {
				if (personMapA[jIdx] == -1 || personMapB[jIdx] == -1)
					epiLosses.emplace_back(m_epiThresh);
				else {
					const float edge = m_jointEpiEdges[jIdx][viewA][viewB](personMapA[jIdx], personMapB[jIdx]);
					if (edge < 0.f)
						valid = false;
					else
						epiLosses.emplace_back(edge);
				}
			}
		}
	}
	if (!valid)
		return -1.f;

	if (epiLosses.size() > 0)
		loss += m_wEpi * std::accumulate(epiLosses.begin(), epiLosses.end(), 0.f) / float(epiLosses.size());

	// paf loss
	std::vector<float> pafLosses;
	for (int view = 0; view < m_cams.size() && valid; view++) {
		if (proposal[view] == -1)
			continue;
		const Eigen::VectorXi& personMap = m_personsMapByView[view][proposal[view]];
		for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
			const Eigen::Vector2i candi(personMap[GetSkelDef().pafDict(0, pafIdx)], personMap[GetSkelDef().pafDict(1, pafIdx)]);
			if (candi.x() >= 0 && candi.y() >= 0)
				pafLosses.emplace_back(1.f - m_detections[view].pafs[pafIdx](candi.x(), candi.y()));
			else
				pafLosses.emplace_back(1.f);
		}
	}
	if (pafLosses.size() > 0)
		loss += m_wPaf * std::accumulate(pafLosses.begin(), pafLosses.end(), 0.f) / float(pafLosses.size());

	// view loss
	loss += m_wView * (1.f - MathUtil::Welsch(m_cViewCnt, (proposal.array() >= 0).count()));
	return loss;
};

void Associater::ClusterTrackingPerson3D(const std::vector<Person3D>& lastperson3d, const int& jIdx)
{

	std::vector<std::pair<float, int>> trackinglosses;
	for (int ProposalIdx = 0; ProposalIdx < m_trackingpersonProposals.size(); ProposalIdx++) {
		const float loss = CalcTrackingJointsLoss(lastperson3d,jIdx,ProposalIdx);
		if (loss > 0.f)
			trackinglosses.emplace_back(std::make_pair(loss, ProposalIdx));
	}

	std::sort(trackinglosses.begin(), trackinglosses.end());


	for (const auto& loss : trackinglosses) {
		const Eigen::VectorXi& trackingProposal = m_trackingpersonProposals[loss.second];
		bool available = true;

		for (int i = 0; i < (trackingProposal.size() - 1); i++) {
			if (m_personsMapByIdx[trackingProposal[m_cams.size()]][i][jIdx] != -1)
				available = false;
		}
		if (!available)
			continue;

		for (int i = 0; i < (trackingProposal.size() - 1)&&(trackingProposal[i] >= 0); i++) {
			if (m_detections[i].joints[jIdx](2, trackingProposal[i]) < 0)
			{
				available = false;
				break;
			}
		}
			
		if (!available)
			continue;

		for (int view = 0; view < m_cams.size(); view++) {
			m_personsMapByIdx[trackingProposal[m_cams.size()]][view][jIdx] = trackingProposal[view];
			if (trackingProposal[view] >= 0)
				m_detections[view].joints[jIdx](2, trackingProposal[view]) = -1;
		}
	}
}

void Associater::ClusterPersons3D()
{
	//m_personsMapByIdx.clear();

	// cluster 3D
	std::vector<std::pair<float, int>> losses;
	for (int personProposalIdx = 0; personProposalIdx < m_personProposals.size(); personProposalIdx++) {
		const float loss = CalcProposalLoss(personProposalIdx);
		if (loss > 0.f)
			losses.emplace_back(std::make_pair(loss, personProposalIdx));
	}

	// parse to cluster greedily
	std::sort(losses.begin(), losses.end());
	std::vector<Eigen::VectorXi> availableMap(m_cams.size());
	for (int view = 0; view < m_cams.size(); view++)
		availableMap[view] = Eigen::VectorXi::Constant(m_personsMapByView[view].size(), 1);

	for (const auto& loss : losses) {
		const Eigen::VectorXi& personProposal = m_personProposals[loss.second];

		bool available = true;
		for (int i = 0; i < personProposal.size() && available; i++)
			available &= (personProposal[i] == -1 || availableMap[i][personProposal[i]]);

		if (!available)
			continue;

		std::vector<Eigen::VectorXi> personMap(m_cams.size(), Eigen::VectorXi::Constant(GetSkelDef().jointSize, -1));
		for (int view = 0; view < m_cams.size(); view++)
			if (personProposal[view] != -1) {
				personMap[view] = m_personsMapByView[view][personProposal[view]];
				for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
					const int candiIdx = personMap[view][jIdx];
					if (candiIdx >= 0)
						m_assignMap[view][jIdx][candiIdx] = m_personsMapByIdx.size();
				}
				availableMap[view][personProposal[view]] = false;
			}
		m_personsMapByIdx.emplace_back(personMap);
	}

	// add remain persons
	for (int view = 0; view < m_cams.size(); view++) {
		for (int i = 0; i < m_personsMapByView[view].size(); i++) {
			if (availableMap[view][i]) {
				std::vector<Eigen::VectorXi> personMap(m_cams.size(), Eigen::VectorXi::Constant(GetSkelDef().jointSize, -1));
				personMap[view] = m_personsMapByView[view][i];
				m_personsMapByIdx.emplace_back(personMap);
			}
		}
	}
}


void Associater::ConstructPersons()
{
	// 2D
	for (int view = 0; view < m_cams.size(); view++) {
		m_persons2D[view].clear();
		for (int pIdx = 0; pIdx < m_personsMapByIdx.size(); pIdx++) {
			const std::vector<Eigen::VectorXi>& personMap = *std::next(m_personsMapByIdx.begin(),pIdx);
			const SkelDetection& detection = m_detections[view];
			Person2D person;
			for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++)
				if (personMap[view][jIdx] != -1)
					person.joints.col(jIdx) = detection.joints[jIdx].col(personMap[view][jIdx]);

			for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++) {
				const int jaIdx = GetSkelDef().pafDict(0, pafIdx);
				const int jbIdx = GetSkelDef().pafDict(1, pafIdx);
				if (personMap[view][jaIdx] != -1 && personMap[view][jbIdx] != -1)
					person.pafs[pafIdx] = detection.pafs[pafIdx](personMap[view][jaIdx], personMap[view][jbIdx]);
			}
			m_persons2D[view].emplace_back(person);
		}
	}

	// 3D
	m_persons3D.clear();
	for (int personIdx = 0; personIdx < m_personsMapByIdx.size(); personIdx++) {
		Person3D person;
		const std::vector<Eigen::VectorXi>& personMap = *std::next(m_personsMapByIdx.begin(), personIdx);
		for (int jIdx = 0; jIdx < GetSkelDef().jointSize; jIdx++) {
			MathUtil::Triangulator triangulator;
			for (int camIdx = 0; camIdx < m_cams.size(); camIdx++) {
				if (personMap[camIdx][jIdx] != -1) {
					triangulator.projs.emplace_back(m_cams[camIdx].proj);
					triangulator.points.emplace_back(m_persons2D[camIdx][personIdx].joints.col(jIdx).head(2));
				}
			}
			triangulator.Solve();
			if (triangulator.loss < m_triangulateThresh)
				person.joints.col(jIdx) = triangulator.pos.homogeneous();
			else
				person.joints.col(jIdx).setZero();
		}
		m_persons3D.emplace_back(person);
	}
}
