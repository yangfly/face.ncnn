// std
#include <iostream>
#include <iomanip>

#include "mtcnn.h"

using namespace std;

// ------------ helper function for debug ------------
string type2str(int type)
{
	string r;
	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);
	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}
	r += "C";
	r += (chans + '0');
	return move(r);
}

void print(const cv::Mat & mat, string name)
{
	cout << name << " cv::Mat " << type2str(mat.type()) << endl;
	cout << fixed << setprecision(4) << right;
	int c = mat.channels();
	for (int k = 0; k < mat.channels(); k++)
	{
		const float * data = mat.ptr<float>(k);
		for (int i = 0; i < mat.rows; i++)
		{
			for (int j = 0; j < mat.cols; j++)
				cout << setw(9) << data[i * mat.cols + j];
			cout << endl;
		}
		cout << endl;
	}
	cout.unsetf(ios_base::floatfield);
}
// ----------------------------------------------------

// xy = [x y]
// XY = [[x  y  1  0]
//       [y -x  0  1]]
cv::Mat stitch(const cv::Mat & xy)
{
	int M = xy.rows;
	cv::Mat x = xy.col(0);
	cv::Mat y = xy.col(1);
	cv::Mat zeros = cv::Mat::zeros(M, 1, CV_32FC1);
	cv::Mat ones = cv::Mat::ones(M, 1, CV_32FC1);

	cv::Mat XY(2 * M, 4, CV_32FC1);

	cv::Mat XY_up = XY.rowRange(0, M);
	x.copyTo(XY_up.col(0));
	y.copyTo(XY_up.col(1));
	ones.copyTo(XY_up.col(2));
	zeros.copyTo(XY_up.col(3));

	cv::Mat XY_down = XY.rowRange(M, 2 * M);
	y.copyTo(XY_down.col(0));
	cv::Mat _x = x * -1;
	_x.copyTo(XY_down.col(1));
	zeros.copyTo(XY_down.col(2));
	ones.copyTo(XY_down.col(3));

	return move(XY);
}

int rank(const cv::Mat & mat)
{
	cv::Mat1d w; // singular values
	cv::SVD::compute(mat, w);
	return cv::countNonZero(w > 0.f);
}

cv::Mat tformfwd(const cv::Mat & trans, const cv::Mat & uv)
{
	cv::Mat UV = cv::Mat::ones(uv.rows, 3, uv.type());
	uv.copyTo(UV.colRange(0, 2));
	cv::Mat XY = UV * trans;
	cv::Mat xy;
	XY.colRange(0, 2).copyTo(xy);
	return move(xy);
}

// 2-norm of matrix = largest singular value of mat.
double norm(const cv::Mat & mat)
{
	cv::Mat1d w; // singular values
	cv::SVD::compute(mat, w);
	double min_v, max_v;
	cv::minMaxIdx(w, &min_v, &max_v);
	return max_v;
}

cv::Mat findNonreflectiveSimilarity(const cv::Mat & uv,
	const cv::Mat & xy, int option_k = 2)
{
	int K = option_k;
	int M = xy.rows;
	cv::Mat X = stitch(xy);
	cv::Mat U = cv::Mat(uv.t()).reshape(1, 2 * M);

	// We know that X * r = U
	CV_Assert(::rank(X) >= 2 * K); // unique solution
	cv::Mat r(4, 1, CV_32FC1);
	cv::solve(X, U, r, cv::DECOMP_SVD);
	
	const float * pdata = r.ptr<float>(0);
	cv::Mat tinv = (cv::Mat_<float>(3, 3) <<
		pdata[0], -pdata[1], 0,
		pdata[1],  pdata[0], 0,
		pdata[2],  pdata[3], 1);

	return move(tinv.inv(cv::DECOMP_SVD));
}

cv::Mat findSimilarity(const cv::Mat & uv, const cv::Mat & xy,
	int option_k = 2)
{
	/// solve for trans1
	cv::Mat trans1 = findNonreflectiveSimilarity(uv, xy, option_k);
	/// solve for trans2
	// manually reflect the xy data across the Y-axis
	cv::Mat xyR;
	xy.copyTo(xyR);
	xyR.col(0) *= -1;
	cv::Mat trans2 = findNonreflectiveSimilarity(uv, xyR, option_k);
	// manually reflect the tfrom to undo the reflection done on xyR.
	trans2.col(0) *= -1;
	// print(trans1, "trans1");
	// print(trans2, "trans2");

	// Figure out if trans1 or trans2 is better
	cv::Mat xy1 = tformfwd(trans1, uv);
	double norm1 = ::norm(xy1 - xy);

	cv::Mat xy2 = tformfwd(trans2, uv);
	double norm2 = ::norm(xy2 - xy);

	if (norm1 <= norm2)
		return move(trans1);
	else
		return move(trans2);
}

using namespace face;
cv::Mat toMat(const FPoints & fpts)
{
  int num = fpts.size();
  cv::Mat mat(num, 2, CV_32FC1);
  for (int i = 0; i < num; i++)
  {
    mat.at<float>(i, 0) = fpts[i].x;
    mat.at<float>(i, 1) = fpts[i].y;
  }
  return move(mat);
}

cv::Mat Mtcnn::cp2tform(const FPoints & src, const FPoints & dst, const char * type)
{
  CHECK(src.size() == dst.size()) << "Number of src facial points and dst facial points don't match.";
  cv::Mat src_mat = toMat(src);
  cv::Mat dst_mat = toMat(dst);
	//print(src_mat, "src_mat");
	//print(dst_mat, "dst_mat");
  cv::Mat trans;
  if (strcmp(type, "nonreflective similarity") == 0)
    trans = findNonreflectiveSimilarity(src_mat, dst_mat);
  else if (strcmp(type, "similarity") == 0)
    trans = findSimilarity(src_mat, dst_mat);
  else
    CHECK(false) << "Unsupport type in cp2tform: " << type;
  return move(trans);
}

// ncnn::Mat Mtcnn::Align(const ncnn::Mat & sample, const FPoints & fpts, int width)
// {
//   CHECK(width == 96 || width == 112) << "Face align only support width to be 96 or 112";
//   ncnn::Mat tform, face;
//   if (width == 96)
//     tform = cp2tform(fpts, ref_96_112);
//   else
//     tform = cp2tform(fpts, ref_112_112);
//   tform = tform.colRange(0, 2).t();
//   warpAffine(sample, face, tform, Size(width, 112));
//   return face;
// }

// ncnn::Mat Mtcnn::DrawInfos(const ncnn::Mat & sample, const vector<FaceInfo> & infos)
// {
//   ncnn::Mat canvas = sample.clone();
//   for (const auto & info : infos) {
//     rectangle(canvas, Rect(info.bbox.x1, info.bbox.y1, info.bbox.x2 - info.bbox.x1,
//       info.bbox.y2 -  info.bbox.y1), Scalar(255, 0, 0), 2);
//     for (const auto pt : info.fpts)
//       circle(canvas, cv::Point(fix(pt.x), fix(pt.y)), 5, Scalar(0, 255, 0), -1);
//     putText(canvas, to_string(info.score), cv::Point((int)(info.bbox.x1), (int)(info.bbox.y1)),
//            FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, LINE_AA);
//   }
//   return canvas;
// }
