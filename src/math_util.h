#pragma once

#include <cmath>
#include <Eigen/Eigen>
#include <fstream>


namespace MathUtil
{
	inline float Welsch(const float& c, const float& _x)
	{
		const float x = _x / c;
		return 1 - expf(-0.5f * x * x);
	}



	float Point2PointDistSquare(const Eigen::Vector3f& pA, const Eigen::Vector3f& pB)
	{
		return ( ( pA(0) - pB(0) ) * ( pA(0) - pB(0)) + ( pA(1) - pB(1)) * ( pA(1) - pB(1) ) );
	}

	float Point2LineDist(const Eigen::Vector3f& pA, const Eigen::Vector3f& pB, const Eigen::Vector3f& ray)
	{
		return ((pA - pB).cross(ray)).norm();
	}

	float Line2LineDist(const Eigen::Vector3f& pA, const Eigen::Vector3f& rayA, const Eigen::Vector3f& pB, const Eigen::Vector3f& rayB)
	{
		if (1 - fabsf(rayA.dot(rayB)) < 1e-3f)
			return Point2LineDist(pA, pB, rayA);
		else
			return fabsf((pA - pB).dot((rayA.cross(rayB)).normalized()));
	}

	struct Triangulator
	{
		std::vector<Eigen::Vector2f> points;
		std::vector<Eigen::Matrix<float, 3, 4>> projs;
		bool convergent = false;
		float loss = FLT_MAX;
		Eigen::Vector3f pos = Eigen::Vector3f::Zero();

		void Clear() {
			convergent = false;
			points.clear();
			projs.clear();
		}

		void Solve(const int& maxIterTime = 20, const float& updateTolerance = 1e-4f, const float& regularTerm = 1e-4f) {
			if (projs.size() < 2) {
				loss = FLT_MAX;
				pos.setZero();
				convergent = false;
				return;
			}
			convergent = false;

			for (int iterTime = 0; iterTime < maxIterTime && !convergent; iterTime++) {
				Eigen::MatrixXf jacobi = Eigen::MatrixXf::Zero(2 * projs.size(), 3);
				Eigen::VectorXf residual(2 * projs.size());
				for (int i = 0; i < projs.size(); i++) {
					const Eigen::Vector3f xyz = projs[i] * pos.homogeneous();
					Eigen::Matrix<float, 2, 3> tmp;
					tmp << 1.0f / xyz.z(), 0.0f, -xyz.x() / (xyz.z()*xyz.z()),
						0.0f, 1.0f / xyz.z(), -xyz.y() / (xyz.z()*xyz.z());
					jacobi.block<2, 3>(2 * i, 0) = tmp * projs[i].block<3, 3>(0, 0);
					residual.segment<2>(2 * i) = xyz.hnormalized() - points[i];
				}

				loss = residual.cwiseAbs().sum() / residual.rows();
				Eigen::MatrixXf hessian = jacobi.transpose()*jacobi + regularTerm * Eigen::MatrixXf::Identity(jacobi.cols(), jacobi.cols());
				Eigen::VectorXf gradient = jacobi.transpose()*residual;

				const Eigen::VectorXf deltaPos = hessian.ldlt().solve(-gradient);
				if (deltaPos.norm() < updateTolerance)
					convergent = true;
				else
					pos += deltaPos;
			}
		}
	};
}


