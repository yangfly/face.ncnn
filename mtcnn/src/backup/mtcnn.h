#ifndef FACE_MTCNN_H_
#define FACE_MTCNN_H_

// // std
// #include <memory>

// ncnn
#include "net.h"
// common types
#include "common.h"

namespace face
{

// Only do normalization once before any image processing to save time.
/// #define NORM_FARST

/// @brief face regression
using Reg = BBox;

/// @brief face proposal
struct Proposal : public FaceInfo
{
	Reg reg;	// face regression
	Proposal(BBox && bbox, float score, Reg && reg) :
    FaceInfo(std::move(bbox), score), reg(reg) {}
	Proposal(BBox && bbox, float score, FPoints && fpts, Reg && reg) :
    FaceInfo(std::move(bbox), score, std::move(fpts)), reg(reg) {}
};

class Mtcnn
{
public:
	/// @brief Constructor.
  /// @brief Lnet: whether to load Lnet.
	Mtcnn(const std::string & model_dir, bool Lnet = true);
  ~Mtcnn();
  /// @brief check whether contain Lnet
  bool HasLnet() const;
	/// @brief Detect faces from image
	std::vector<FaceInfo> Detect(ncnn::Mat & image);
  /// @brief Get facial points of detect face by O/Lnet
  FPoints Landmark(const ncnn::Mat & image, BBox bbox = BBox());
	/// @brief Align image with facial points, with cp2tform and cv::warpAffine
	// ncnn::Mat Align(const ncnn::Mat & image, const FPoints & fpts, int width = 112);
  /// @brief Draw face infos on image.
  // ncnn::Mat DrawInfos(const ncnn::Mat & image, const std::vector<FaceInfo> & infos);

	// default settings
	int face_min_size = 40;
  int face_max_size = 500;
	float scale_factor = 0.709f;
	float thresholds[3] = {0.8f, 0.9f, 0.9f};
	bool enable_precise_landmark = true;

private:
  enum NMS_TYPE {
		IoM,	// Intersection over Union
		IoU		// Intersection over Minimum
	};

	// networks
  ncnn::Net Pnet, Rnet, Onet, Lnet;
  bool has_Lnet;
	// reference standard facial points
	FPoints ref_96_112;
	FPoints ref_112_112;

	/// @brief Create scale pyramid: down order
	std::vector<float> ScalePyramid(const int min_len);
	/// @brief Get bboxes from maps of confidences and regressions.
	std::vector<Proposal> GetCandidates(const float scale, 
    const ncnn::Mat & conf_blob, const ncnn::Mat & loc_blob);
	/// @brief Non Maximum Supression with type 'IoU' or 'IoM'.
	std::vector<Proposal> NonMaximumSuppression(std::vector<Proposal> & pros,
		const float threshold, const NMS_TYPE type);
	/// @brief Refine bounding box with regression.
	void BoxRegression(std::vector<Proposal> & pros);
	/// @brief Convert bbox from float bbox to square.
	void Square(std::vector<BBox> & bboxes);
	/// @brief Crop proposals with padding 0.
	ncnn::Mat CropPadding(const ncnn::Mat & image, const BBox & bbox);

	/// @brief Stage 1: Pnet get proposal bounding boxes
	std::vector<BBox> ProposalNetwork(const ncnn::Mat & image);
	/// @brief Stage 2: Rnet refine and reject proposals
	std::vector<BBox> RefineNetwork(const ncnn::Mat & image, std::vector<BBox> & bboxes);
	/// @brief Stage 3: Onet refine and reject proposals and regress facial landmarks.
	std::vector<FaceInfo> OutputNetwork(const ncnn::Mat& image, std::vector<BBox> & bboxes);
	/// @brief Stage 4: Lnet refine facial landmarks
	void LandmarkNetwork(const ncnn::Mat & image, std::vector<FaceInfo> & infos);
	// /// @brief cp2tform matlab export to c++
	// /// @param type: 'similarity' or 'nonreflective similarity'
	// ncnn::Mat cp2tform(const FPoints & src, const FPoints & dst, const char * type="similarity");
};	// class MTCNN

} // namespace face

#endif // FACE_MTCNN_H_
