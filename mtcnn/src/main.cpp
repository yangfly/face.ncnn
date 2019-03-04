#include <iostream>
#include <opencv2/opencv.hpp>
#include "mtcnn.h"

using namespace std;
using namespace cv;
using namespace face;

#ifdef _MSC_VER
#include <intrin.h>
string cpu_info()
{
  int id[4] = { -1 };
  char info[32] = { 0 };
  string cpu_info;
  for (uint i = 0; i < 3; i++) {
    __cpuid(id, 0x80000002 + i);
    memcpy(info, id, sizeof(id));
    cpu_info += info;
  }
  return cpu_info;
}
#else
#include <fstream>
string cpu_info()
{
  ifstream file("/proc/cpuinfo");
  string line;
  for (int i = 0; i < 5; i++)
    getline(file, line);
  return line.substr(13);
}
#endif

Mat imdraw(const Mat im, const vector<BBox> & bboxes)
{
	Mat canvas = im.clone();
	for (const auto & bbox : bboxes) {
		rectangle(canvas, Rect(bbox.x1, bbox.y1, bbox.x2-bbox.x1, bbox.y2-bbox.y1), Scalar(255, 0, 0), 2);
    for (int i = 0; i < 5; i++) {
      circle(canvas, Point((int)bbox.fpoints[i], (int)bbox.fpoints[i+5]), 5, Scalar(0, 255, 0), -1);
    }
		putText(canvas, to_string(bbox.score), Point(bbox.x1, bbox.y1),
			FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, LINE_AA);
	}
	return canvas;
}

void demo() {
  Mtcnn mtcnn("../models");
  Mat im = imread("../../sample.jpg");
  ncnn::Mat image = ncnn::Mat::from_pixels(im.data, ncnn::Mat::PIXEL_BGR2RGB, im.cols, im.rows);
  vector<BBox> bboxes = mtcnn.Detect(image);
  Mat canvas = imdraw(im, bboxes);
  imshow("mtcnn face detector", canvas);
  cv::waitKey(0);
}

void performance(bool lnet = true, int ntimes = 50) {
  Mtcnn mtcnn("../models", lnet);
  Mat im = imread("../../sample.jpg");
  ncnn::Mat image = ncnn::Mat::from_pixels(im.data, ncnn::Mat::PIXEL_BGR2RGB, im.cols, im.rows);
  clock_t begin = clock();
  for (int i = 0; i < ntimes; i++)
    mtcnn.Detect(image);
  clock_t end = clock();
  string disc_head = lnet ? "With LNet" : "Without LNet";
  string disc_pad = "=============";
  cout << disc_pad << " " << disc_head << " " << disc_pad << endl;
  cout << "cpu info: " << cpu_info() << endl;
  cout << "image shape: (" << image.w << ", " << image.h << ", " << image.c << ")" << endl;
  cout << "performance : " << CLOCKS_PER_SEC / (double)(end - begin) * ntimes << " fps" << endl;
  cout << "detect time: " << (double)(end - begin) / ntimes << " ms" << endl;
}

int main() {
  performance(true);
  performance(false);
  demo();
  return 0;
}
