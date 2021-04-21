#pragma once
#include <Eigen/Eigen>
#include <string>
#include <vector>


struct Camera
{
	Eigen::Matrix3f K, Ki, R, Rt, RtKi;
	Eigen::Vector3f T, pos;
	Eigen::Matrix<float, 3, 4> proj;
	Camera()
	{
		K.setIdentity();
		K.block<2, 1>(0, 2) = Eigen::Vector2f(0.5f, 0.5f);
		R.setIdentity();
		T.setZero();
		Update();
	}

	void Update()
	{
		Ki = K.inverse();
		Rt = R.transpose();
		RtKi = Rt * Ki;
		proj.leftCols(3) = R;
		proj.rightCols(1) = T;
		proj = K * proj;
		pos = -Rt * T;;
	}

	Eigen::Vector3f CalcRay(const Eigen::Vector2f& uv) const
	{
		return (-RtKi * uv.homogeneous()).normalized();
	}//µ¥Î»»¯xyz
};

