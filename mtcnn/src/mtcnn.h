#ifndef FACE_MTCNN_H_
#define FACE_MTCNN_H_

// ncnn
#include "net.h"

namespace face
{
// Bounding box for hold score, box and facial points
class BBox {
public:
  explicit BBox() : x1(0), y1(0), x2(0), y2(0), score(0.f) {};
  explicit BBox(float score, int x1, int y1, int x2, int y2, const float fpoints[])
    : score(score), x1(x1), y1(y1), x2(x2), y2(y2) {
    memcpy(this->fpoints, fpoints, sizeof(this->fpoints));
  }
  BBox(const BBox & bbox)
    : BBox(bbox.score, bbox.x1, bbox.y1, bbox.x2, bbox.y2, bbox.fpoints) {}

  int x1, y1, x2, y2;
  float score;
  float fpoints[10]; // [x1..x5, y1..y5]
  int area() const {
    return (x2 - x1) * (y2 - y1);
  }
};

// Regression offset of bbox
class BBoxReg {
  float x1, y1, x2, y2;
};

class Mtcnn
{
public:
	/// @brief Constructor.
  /// @brief Lnet: whether to load Lnet.
	Mtcnn(const std::string & model_dir, bool Lnet = true);
  ~Mtcnn();
	/// @brief Detect faces from image
	std::vector<BBox> Detect(const ncnn::Mat & image);
  /// @brief Get facial points of detect face by O/Lnet
  BBox Landmark(const ncnn::Mat & image, BBox bbox = BBox());

	// default settings
	int face_min_size = 40;
  int face_max_size = 500;
	float scale_factor = 0.709f;
	float thresholds[3] = {0.8f, 0.9f, 0.9f};
	bool precise_landmark = true;

private:
  // Inter _BBox extend outer BBox with location regression offsets.
  class _BBox : public BBox {
  public:
    explicit _BBox() {}
    explicit _BBox(const BBox & bbox) : BBox(bbox) {}
    float regs[4]; // [x1, y2, x2, y2]
    _BBox & operator=(const _BBox & _bbox) {
      x1 = _bbox.x1;
      y1 = _bbox.y1;
      x2 = _bbox.x2;
      y2 = _bbox.y2;
      score = _bbox.score;
      memcpy(regs, _bbox.regs, sizeof(regs));
      memcpy(fpoints, _bbox.fpoints, sizeof(fpoints));
      return *this;
    }
    BBox base() const {
      return BBox(score, x1, y1, x2, y2, fpoints);
    }
  };

  enum NMS_TYPE {
		IoM,	// Intersection over Union
		IoU		// Intersection over Minimum
	};

	// networks
  ncnn::Net Pnet, Rnet, Onet, Lnet;
  bool lnet;

	/// @brief Create scale pyramid: down order
	std::vector<float> ScalePyramid(const int min_len);
	/// @brief Get bboxes from maps of confidences and regressions.
	std::vector<_BBox> GetCandidates(const float scale,
    const ncnn::Mat & conf_blob, const ncnn::Mat & loc_blob);
	/// @brief Non Maximum Supression with type 'IoU' or 'IoM'.
	void NonMaximumSuppression(std::vector<_BBox> & _bboxes,
		const float threshold, const NMS_TYPE type);
	/// @brief Refine bounding box with regression
  /// @optional param square: where expand bbox to square.
	void BoxRegression(std::vector<_BBox> & _bboxes, bool square);
  /// @brief Crop proposals with padding 0.
  ncnn::Mat PadCrop(const ncnn::Mat & image, int x1, int y1, int x2, int y2);

	/// @brief Stage 1: Pnet get proposal bounding boxes
  std::vector<_BBox> ProposalNetwork(const ncnn::Mat & image);
	/// @brief Stage 2: Rnet refine and reject proposals
	void RefineNetwork(const ncnn::Mat & image, std::vector<_BBox> & _bboxes);
	/// @brief Stage 3: Onet refine and reject proposals and regress facial landmarks.
  void OutputNetwork(const ncnn::Mat & image, std::vector<_BBox> & _bboxes);
	/// @brief Stage 4: Lnet refine facial landmarks
	void LandmarkNetwork(const ncnn::Mat & image, std::vector<_BBox> & _bboxes);
};	// class MTCNN

} // namespace face

#endif // FACE_MTCNN_H_
