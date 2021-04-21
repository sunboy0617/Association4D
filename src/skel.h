#pragma once
#include <Eigen/Eigen>
#include <string>
#include <vector>


struct SkelDef
{
	int jointSize;
	int pafSize;
	int shapeSize;
	Eigen::Matrix2Xi pafDict;
};


inline const SkelDef& GetSkelDef()
{
	static const SkelDef skelDef = [] {
		SkelDef _skelDef;
		// SKEL19
		_skelDef.jointSize = 19;
		_skelDef.pafSize = 18;
		_skelDef.shapeSize = 10;
		_skelDef.pafDict.resize(2, 18);
		_skelDef.pafDict << 1, 2, 7, 0, 0, 3, 8, 1, 5, 11, 5, 1, 6, 12, 6, 1, 14, 13,
			0, 7, 13, 2, 3, 8, 14, 5, 11, 15, 9, 6, 12, 16, 10, 4, 17, 18;
		return _skelDef;
	}();
	return skelDef;
}


struct Person2D
{
	Person2D() {
		joints = Eigen::Matrix3Xf::Zero(3, GetSkelDef().jointSize);
		pafs = Eigen::VectorXf::Zero(GetSkelDef().pafSize);
	}
	float CalcScore() const { return joints.row(2).sum() + pafs.sum(); }
	int GetJointCnt() const { return int((joints.row(2).array() > 0.f).count()); }
	Eigen::Matrix3Xf joints;
	Eigen::VectorXf pafs;
};


struct Person3D
{
	Eigen::Matrix4Xf joints;
	Person3D() {
		joints = Eigen::Matrix4Xf::Zero(4, GetSkelDef().jointSize);
	}

	Person2D ProjSkel(const Eigen::Matrix<float, 3, 4>& proj) const {
		Person2D person;
		person.joints.topRows(2) = (proj* (joints.topRows(3).colwise().homogeneous())).colwise().hnormalized();
		person.joints.row(2) = joints.row(3);
		for (int pafIdx = 0; pafIdx < GetSkelDef().pafSize; pafIdx++)
			person.pafs[pafIdx] = person.joints(GetSkelDef().pafDict(0, pafIdx)) > FLT_EPSILON
			&&  person.joints(GetSkelDef().pafDict(1, pafIdx)) > FLT_EPSILON ? 1.f : 0.f;
		return person;
	}
};

struct SkelDetection
{
	SkelDetection() {
		joints.resize(GetSkelDef().jointSize);
		pafs.resize(GetSkelDef().pafSize);
	}
	std::vector<Eigen::Matrix3Xf> joints;
	std::vector<Eigen::MatrixXf> pafs;
};

