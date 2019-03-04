#include <algorithm>  // std::min, std::max, std::sort
//#define USE_OPENMP
#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "mtcnn.h"
using namespace std;
using namespace face;

#include <limits>
int fix(float f)
{
  f *= 1 + numeric_limits<float>::epsilon();
  return static_cast<int>(f);
}

Mtcnn::Mtcnn(const string & model_dir, bool Lnet) :
  has_Lnet(Lnet)
{
  // load models
  Pnet.load_param((model_dir + "/det1.param").data());
  Pnet.load_model((model_dir + "/det1.bin").data());
  Rnet.load_param((model_dir + "/det2.param").data());
  Rnet.load_model((model_dir + "/det2.bin").data());
  Onet.load_param((model_dir + "/det3.param").data());
  Onet.load_model((model_dir + "/det3.bin").data());
  if (HasLnet()) {
    this->Lnet.load_param((model_dir + "/det4.param").data());
    this->Lnet.load_model((model_dir + "/det4.bin").data());
  }

  // init reference facial points
  // as c++ or python index start from 0
  // w.r.t Matlab index start from 1
  // this will result a minus one on reference landmarks
  ref_96_112.reserve(5);
  ref_96_112.emplace_back(29.2946, 50.6963);
  ref_96_112.emplace_back(64.5318, 50.5014);
  ref_96_112.emplace_back(47.0252, 70.7366);
  ref_96_112.emplace_back(32.5493, 91.3655);
  ref_96_112.emplace_back(61.7299, 91.2041);

  ref_112_112.reserve(5);
  for (const auto & pt : ref_96_112)
    ref_112_112.emplace_back(pt.x + 8.f, pt.y);
}

Mtcnn::~Mtcnn() {
  Pnet.clear();
  Rnet.clear();
  Onet.clear();
  if (HasLnet())
    Lnet.clear();
}

bool Mtcnn::HasLnet() const {
  return has_Lnet;
}

vector<FaceInfo> Mtcnn::Detect(ncnn::Mat & image)
{
  vector<BBox> bboxes = ProposalNetwork(image);
  bboxes = RefineNetwork(image, bboxes);
  vector<FaceInfo> infos = OutputNetwork(image, bboxes);
  if (enable_precise_landmark && HasLnet())
    LandmarkNetwork(image, infos);

  // matlab CS to C++ CS
  for (auto & info : infos)
  {
    info.bbox.x1 -= 1;
    info.bbox.y1 -= 1;
    for (auto & fpt : info.fpts)
    {
      fpt.x -= 1;
      fpt.y -= 1;
    }
  }
    
  return infos;
}

FPoints Mtcnn::Landmark(const ncnn::Mat & image, BBox bbox) {
  vector<BBox> bboxes = { bbox };
  vector<FaceInfo> infos = OutputNetwork(image, bboxes);
  if (enable_precise_landmark && HasLnet())
    LandmarkNetwork(image, infos);

  FPoints fpts;
  if (!infos.empty()) {
    fpts = infos[0].fpts;
    // matlab CS to C++ CS
    for (auto & fpt : fpts)
    {
      fpt.x -= 1;
      fpt.y -= 1;
    }
  }
  return fpts;
}

vector<float> Mtcnn::ScalePyramid(const int min_len)
{
  vector<float> scales;
  float max_scale = 12.0f / face_min_size;
  float min_scale = 12.0f / std::min<int>(min_len, face_max_size);
  for (float scale = max_scale; scale >= min_scale; scale *= scale_factor)
    scales.push_back(scale);
  return scales;
}

vector<Proposal> Mtcnn::GetCandidates(const float scale, 
  const ncnn::Mat & conf_blob, const ncnn::Mat & loc_blob)
{
  int stride = 2;
  int cell_size = 12;
  vector<Proposal> pros;

  int id = 0;
  for (int i = 0; i < conf_blob.h; ++i)
    for (int j = 0; j < conf_blob.w; ++j) {
      if (conf_blob.channel(1)[id] >= thresholds[0]) {
        // bounding box
        BBox bbox(
          (j * stride + 1) / scale,	        // x1
          (i * stride + 1) / scale,	        // y1
          (j * stride + cell_size) / scale,	// x2
          (i * stride + cell_size) / scale);	// y2
        // bbox regression
        Reg reg(
          loc_blob.channel(0)[id],	// reg_x1
          loc_blob.channel(1)[id],	// reg_y1
          loc_blob.channel(2)[id],	// reg_x2
          loc_blob.channel(3)[id]);	// reg_y2
        // face confidence
        float score = conf_blob.channel(1)[id];
        pros.emplace_back(std::move(bbox), score, std::move(reg));
      }
      id++;
    }
  return pros;
}

vector<Proposal> Mtcnn::NonMaximumSuppression(vector<Proposal> & pros,
  const float threshold, const NMS_TYPE type)
{
  if (pros.size() <= 1)
    return pros;

  sort(pros.begin(), pros.end(),
    // Lambda function: descending order by score.
    [](const Proposal& x, const Proposal& y) -> bool { return x.score > y.score; });


  vector<Proposal> nms_pros;
  while (!pros.empty()) {
    // select maximun candidates.
    Proposal max = pros[0];
    pros.erase(pros.begin());
    float max_area = (max.bbox.x2 - max.bbox.x1 + 1)
      * (max.bbox.y2 - max.bbox.y1 + 1);
    // filter out overlapped candidates in the rest.
    int idx = 0;
    while (idx < pros.size()) {
      // computer intersection.
      float x1 = std::max<float>(max.bbox.x1, pros[idx].bbox.x1);
      float y1 = std::max<float>(max.bbox.y1, pros[idx].bbox.y1);
      float x2 = std::min<float>(max.bbox.x2, pros[idx].bbox.x2);
      float y2 = std::min<float>(max.bbox.y2, pros[idx].bbox.y2);
      float overlap = 0;
      if (x1 <= x2 && y1 <= y2)
      {
        float inter = (x2 - x1 + 1) * (y2 - y1 + 1);
        // computer denominator.
        float outer;
        float area = (pros[idx].bbox.x2 - pros[idx].bbox.x1 + 1)
          * (pros[idx].bbox.y2 - pros[idx].bbox.y1 + 1);
        if (type == IoM)	// Intersection over Minimum
          outer = std::min<float>(max_area, area);
        else	// Intersection over Union
          outer = max_area + area - inter;
        overlap = inter / outer;
      }

      if (overlap > threshold)	// erase overlapped candidate
        pros.erase(pros.begin() + idx);
      else
        idx++;	// check next candidate
    }
    nms_pros.push_back(max);
  }

  return nms_pros;
}

void Mtcnn::BoxRegression(vector<Proposal> & pros)
{
  for (auto& pro : pros) {
    float width = pro.bbox.x2 - pro.bbox.x1 + 1;
    float height = pro.bbox.y2 - pro.bbox.y1 + 1;
    pro.bbox.x1 += pro.reg.x1 * width;	// x1
    pro.bbox.y1 += pro.reg.y1 * height;	// y1
    pro.bbox.x2 += pro.reg.x2 * width;	// x2
    pro.bbox.y2 += pro.reg.y2 * height;	// y2
  }
}

void Mtcnn::Square(vector<BBox> & bboxes) {
  for (auto & bbox : bboxes) {
    float w = bbox.x2 - bbox.x1 + 1;
    float h = bbox.y2 - bbox.y1 + 1;
    float maxl = std::max<float>(w, h);
    bbox.x1 += (w - maxl) * 0.5f;
    bbox.y1 += (h - maxl) * 0.5f;
    bbox.x1 = round(bbox.x1);
    bbox.y1 = round(bbox.y1);
    bbox.x2 = bbox.x1 + fix(maxl) - 1;
    bbox.y2 = bbox.y1 + fix(maxl) - 1;
  }
}

ncnn::Mat Mtcnn::CropPadding(const ncnn::Mat & image, const BBox & bbox) {
  // inside box
  int x1 = std::max<int>(bbox.x1, 1);
  int y1 = std::max<int>(bbox.y1, 1);
  int x2 = std::min<int>(bbox.x2, image.w);
  int y2 = std::min<int>(bbox.y2, image.h);
  bool need_crop = x1 <= x2 && y1 <= y2;
  bool need_pad = x1 > bbox.x1 || y1 > bbox.y1 || x2 < bbox.x2 || y2 < bbox.y2;
  ncnn::Mat crop, pad;
  if (need_crop) {
    ncnn::copy_cut_border(image, crop, y1-1, image.h - y2, x1-1, image.w - x2);
    if (need_pad) {
      ncnn::copy_make_border(crop, pad, y1-fix(bbox.y1), fix(bbox.y2)-y2, x1-fix(bbox.x1), fix(bbox.x2)-x2, 0, 0);
    } else {
      pad = crop;
    }
  } else {
    pad.create(fix(bbox.x2-bbox.x1) + 1, fix(bbox.y2-bbox.y1) + 1, image.c, image.elemsize);
    pad.fill(0.f);
  }
  return pad;
}

vector<BBox> Mtcnn::ProposalNetwork(const ncnn::Mat & image)
{
  int min_len = std::min<int>(image.w, image.h);
  vector<float> scales = ScalePyramid(min_len);
  vector<Proposal> total_pros;
#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
  for (float scale : scales) {
    int width = static_cast<int>(ceil(image.w * scale));
    int height = static_cast<int>(ceil(image.h * scale));
    ncnn::Mat input;
    ncnn::resize_bilinear(image, input, width, height);
    ncnn::Extractor ex = Pnet.create_extractor();
    ex.input("data", input);
    ncnn::Mat conf_blob, loc_blob;
    ex.extract("prob1", conf_blob);
    ex.extract("conv4-2", loc_blob);
    vector<Proposal> pros = GetCandidates(scale, conf_blob, loc_blob);
    // intra scale nms
    pros = NonMaximumSuppression(pros, 0.5f, IoU);
    if (!pros.empty()) {
      total_pros.insert(total_pros.end(), pros.begin(), pros.end());
    }
  }
  // inter scale nms
  total_pros = NonMaximumSuppression(total_pros, 0.7f, IoU);
  BoxRegression(total_pros);
  vector<BBox> bboxes;
  for (auto& pro : total_pros)
    bboxes.emplace_back(pro.bbox);
  return bboxes;
}

vector<BBox> Mtcnn::RefineNetwork(const ncnn::Mat & image, vector<BBox> & bboxes)
{
  if (bboxes.empty())
    return bboxes;

  vector<Proposal> pros;
  Square(bboxes);	// convert bbox to square
#ifdef USE_OPENMP
   #pragma omp parallel for
#endif
  for (auto && bbox : bboxes) {
    ncnn::Mat pad = CropPadding(image, bbox);
    ncnn::Mat input;
    ncnn::resize_bilinear(pad, input, 24, 24);
    ncnn::Extractor ex = Rnet.create_extractor();
    ex.input("data", input);
    ncnn::Mat conf_blob, loc_blob;
    ex.extract("prob1", conf_blob);
    ex.extract("fc5-2", loc_blob);
    float score = conf_blob.channel(0)[1];
    if (score >= thresholds[1]) {
      // bbox regression
      Reg reg(
        loc_blob.channel(0)[0],
        loc_blob.channel(0)[1],
        loc_blob.channel(0)[2],
        loc_blob.channel(0)[3]
      );
      pros.emplace_back(std::move(bbox), score, std::move(reg));
    }
  }

  pros = NonMaximumSuppression(pros, 0.7f, IoU);
  BoxRegression(pros);

  bboxes.clear();
  for (auto& pro : pros)
    bboxes.emplace_back(pro.bbox);
  return bboxes;
}

vector<FaceInfo> Mtcnn::OutputNetwork(const ncnn::Mat & image, vector<BBox> & bboxes)
{
  vector<FaceInfo> infos;
  if (bboxes.empty())
    return infos;

  Square(bboxes);	// convert bbox to square
  vector<Proposal> pros;
#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
  for (auto && bbox : bboxes) {
    ncnn::Mat pad = CropPadding(image, bbox);
    ncnn::Mat input;
    ncnn::resize_bilinear(pad, input, 48, 48);
    ncnn::Extractor ex = Onet.create_extractor();
    ex.input("data", input);
    ncnn::Mat conf_blob, loc_blob, kpt_blob;
    ex.extract("prob1", conf_blob);
    ex.extract("fc6-2", loc_blob);
    ex.extract("fc6-3", kpt_blob);
    float score = conf_blob.channel(0)[1];
    if (score >= thresholds[2]) {
      // bbox regression
      Reg reg(
        loc_blob.channel(0)[0],
        loc_blob.channel(0)[1],
        loc_blob.channel(0)[2],
        loc_blob.channel(0)[3]
      );
      // facial landmarks
      float width = bbox.x2 - bbox.x1 + 1;
      float height = bbox.y2 - bbox.y1 + 1;
      FPoints fpt;
      for (int i = 0; i < 5; i++) {
        fpt.emplace_back(kpt_blob.channel(0)[i] * width + bbox.x1,
                         kpt_blob.channel(0)[i+5] * height + bbox.y1);
      }
      pros.emplace_back(std::move(bbox), score, std::move(fpt), std::move(reg));
    }
  }

  BoxRegression(pros);
  pros = NonMaximumSuppression(pros, 0.7f, IoM);
  
  for (auto & pro : pros)
    infos.emplace_back(std::move(pro.bbox), pro.score, std::move(pro.fpts));

  return infos;
}

void Mtcnn::LandmarkNetwork(const ncnn::Mat & image, vector<FaceInfo> & infos)
{
  if (infos.empty())
    return;

#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
  for (auto & info : infos) {
    float patchw = std::max<float>(info.bbox.x2 - info.bbox.x1,
      info.bbox.y2 - info.bbox.y1) + 1;
    int patchs = fix(patchw * 0.25f);
    if (patchs % 2 == 1)
      patchs += 1;
    ncnn::Mat input;
    input.create(24, 24, image.c * 5, image.elemsize);
    for (int i = 0; i < 5; i++) {
      BBox patch;
      info.fpts[i].x = round(info.fpts[i].x);
      info.fpts[i].y = round(info.fpts[i].y);
      patch.x1 = info.fpts[i].x - patchs / 2;
      patch.y1 = info.fpts[i].y - patchs / 2;
      patch.x2 = patch.x1 + patchs - 1;
      patch.y2 = patch.y1 + patchs - 1;
      ncnn::Mat pad = CropPadding(image, patch);
      ncnn::Mat channels = input.channel_range(image.c * i, image.c);
      ncnn::resize_bilinear(pad, channels, 24, 24);
    }
    ncnn::Extractor ex = Lnet.create_extractor();
    ex.input("data", input);
    vector<ncnn::Mat> blobs(5);
    ex.extract("fc5_1", blobs[0]);
    ex.extract("fc5_2", blobs[1]);
    ex.extract("fc5_3", blobs[2]);
    ex.extract("fc5_4", blobs[3]);
    ex.extract("fc5_5", blobs[4]);
    for (int i = 0; i < 5; i++) {
      float off_x = blobs[i].channel(0)[0] - 0.5f;
      float off_y = blobs[i].channel(0)[1] - 0.5f;
      // Dot not make large movement with relative offset > 0.35
      if (fabs(off_x) <= 0.35 && fabs(off_y) <= 0.35) {
        info.fpts[i].x += off_x * patchs;
        info.fpts[i].y += off_y * patchs;
      }
    }
  }
}
