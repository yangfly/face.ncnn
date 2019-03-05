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
  lnet(Lnet)
{
  // load models
  Pnet.load_param((model_dir + "/det1.param").data());
  Pnet.load_model((model_dir + "/det1.bin").data());
  Rnet.load_param((model_dir + "/det2.param").data());
  Rnet.load_model((model_dir + "/det2.bin").data());
  Onet.load_param((model_dir + "/det3.param").data());
  Onet.load_model((model_dir + "/det3.bin").data());
  if (lnet) {
    this->Lnet.load_param((model_dir + "/det4.param").data());
    this->Lnet.load_model((model_dir + "/det4.bin").data());
  }
}

Mtcnn::~Mtcnn() {
  Pnet.clear();
  Rnet.clear();
  Onet.clear();
  if (lnet)
    Lnet.clear();
}

vector<BBox> Mtcnn::Detect(const ncnn::Mat & image)
{
  vector<_BBox> _bboxes = ProposalNetwork(image);
  RefineNetwork(image, _bboxes);
  OutputNetwork(image, _bboxes);
  if (precise_landmark && lnet)
    LandmarkNetwork(image, _bboxes);
  vector<BBox> bboxes;
  for (const _BBox & _bbox : _bboxes)
    bboxes.emplace_back(_bbox.base());
  return bboxes;
}

BBox Mtcnn::Landmark(const ncnn::Mat & image, BBox bbox) {
  vector<_BBox> _bboxes = { _BBox(bbox) };
  OutputNetwork(image, _bboxes);
  if (precise_landmark && lnet)
    LandmarkNetwork(image, _bboxes);
  if (!_bboxes.empty()) {
    return _bboxes[0].base();
  }
  else {
    return BBox();
  }
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

vector<Mtcnn::_BBox> Mtcnn::GetCandidates(const float scale, 
  const ncnn::Mat & conf_blob, const ncnn::Mat & loc_blob)
{
  int stride = 2;
  int cell_size = 12;
  float inv_scale = 1.0f / scale;
  vector<_BBox> condidates;

  int id = 0;
  for (int i = 0; i < conf_blob.h; ++i)
    for (int j = 0; j < conf_blob.w; ++j) {
      float score = conf_blob.channel(1)[id];
      if (score >= thresholds[0]) {
        condidates.emplace_back();
        _BBox & _bbox = *condidates.rbegin();
        _bbox.score = score;
        _bbox.x1 = round(j * stride * inv_scale);
        _bbox.y1 = round(i * stride * inv_scale);
        //_bbox.x1 = round((j * stride + 1) * inv_scale) - 1;
        //_bbox.y1 = round((i * stride + 1) * inv_scale) - 1;
        _bbox.x2 = round((j * stride + cell_size) * inv_scale);
        _bbox.y2 = round((i * stride + cell_size) * inv_scale);
        for (int i = 0; i < 4; i++)
          _bbox.regs[i] = loc_blob.channel(i)[id];
      }
      id++;
    }
  return condidates;
}

void Mtcnn::NonMaximumSuppression(std::vector<Mtcnn::_BBox> & _bboxes,
  const float threshold, const NMS_TYPE type) {
  if (_bboxes.size() <= 1)
    return;

  sort(_bboxes.begin(), _bboxes.end(),
    // Lambda function: descending order by score.
    [](const _BBox & x, const _BBox & y) -> bool { return x.score > y.score; });

  int keep = 0;  // index of candidates to be keeped
  while (keep < _bboxes.size()) {
    // pass maximun candidates.
    const _BBox & _max = _bboxes[keep++];
    int max_area = _max.area();
    // filter out overlapped candidates in the rest.
    for (int i = keep; i < _bboxes.size(); ) {
      // computer intersection.
      const _BBox & _bbox = _bboxes[i];
      int x1 = std::max<int>(_max.x1, _bbox.x1);
      int y1 = std::max<int>(_max.y1, _bbox.y1);
      int x2 = std::min<int>(_max.x2, _bbox.x2);
      int y2 = std::min<int>(_max.y2, _bbox.y2);
      float overlap = 0.f;
      if (x1 < x2 && y1 < y2) {
        int inter = (x2 - x1) * (y2 - y1);
        int outer;
        if (type == IoM)  // Intersection over Minimum
          outer = std::min<int>(max_area, _bbox.area());
        else  // Intersection over Union
          outer = max_area + _bbox.area() - inter;
        overlap = static_cast<float>(inter) / outer;
      }
      if (overlap > threshold)
        // erase overlapped candidate.
        _bboxes.erase(_bboxes.begin() + i);
      else
        i++;  // check next candidate.
    }
  }
}

void Mtcnn::BoxRegression(std::vector<Mtcnn::_BBox> & _bboxes, bool square)
{
  for (auto & _bbox : _bboxes) {
    // bbox regression.
    float w = static_cast<float>(_bbox.x2 - _bbox.x1);
    float h = static_cast<float>(_bbox.y2 - _bbox.y1);
    float x1 = _bbox.x1 + _bbox.regs[0] * w;
    float y1 = _bbox.y1 + _bbox.regs[1] * h;
    float x2 = _bbox.x2 + _bbox.regs[2] * w;
    float y2 = _bbox.y2 + _bbox.regs[3] * h;
    // expand bbox to square.
    if (square) {
      w = x2 - x1;
      h = y2 - y1;
      float maxl = std::max<float>(w, h);
      _bbox.x1 = static_cast<int>(round(x1 + (w - maxl) * 0.5f));
      _bbox.y1 = static_cast<int>(round(y1 + (h - maxl) * 0.5f));
      _bbox.x2 = _bbox.x1 + fix(maxl);
      _bbox.y2 = _bbox.y1 + fix(maxl);
    }
    else {
      _bbox.x1 = static_cast<int>(round(x1));
      _bbox.y1 = static_cast<int>(round(y1));
      _bbox.x2 = static_cast<int>(round(x2));
      _bbox.y2 = static_cast<int>(round(y2));
    }
  }
}

ncnn::Mat Mtcnn::PadCrop(const ncnn::Mat & image, int x1, int y1, int x2, int y2) 
{
  bool need_pad = x1 < 0 || y1 < 0 || x2 > image.w || y2 >= image.h;
  ncnn::Mat crop, pad;
  bool need_crop = true;
  if (need_pad) {
    // inside box
    int _x1 = std::max<int>(x1, 0);
    int _y1 = std::max<int>(y1, 0);
    int _x2 = std::min<int>(x2, image.w);
    int _y2 = std::min<int>(y2, image.h);
    need_crop = _x1 < _x2 && _y1 < _y2;
    if (need_crop) {
      ncnn::copy_cut_border(image, crop, _y1, image.h-_y2, _x1, image.w-_x2);
      ncnn::copy_make_border(crop, pad, _y1-y1, y2-_y2, _x1-x1, x2-_x2, 0, 0);
    }
  }
  else {
    need_crop = x1 < x2 && y1 < y2;
    if (need_crop)
      ncnn::copy_cut_border(image, pad, y1, image.h-y2, x1, image.w-x2);
  }
  if (need_crop == false) {
    pad.create(x2 - x1, y2 - y1, image.c, image.elemsize);
    pad.fill(0.f);
  }
  return pad;
}

vector<Mtcnn::_BBox> Mtcnn::ProposalNetwork(const ncnn::Mat & image)
{
  int min_len = std::min<int>(image.w, image.h);
  vector<float> scales = ScalePyramid(min_len);
  vector<_BBox> total_bboxes;
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
    vector<_BBox> scale_bboxes = GetCandidates(scale, conf_blob, loc_blob);
    // intra scale nms
    NonMaximumSuppression(scale_bboxes, 0.5f, IoU);
    if (!scale_bboxes.empty()) {
      total_bboxes.insert(total_bboxes.end(), scale_bboxes.begin(), scale_bboxes.end());
    }
  }
  // inter scale nms
  NonMaximumSuppression(total_bboxes, 0.7f, IoU);
  BoxRegression(total_bboxes, true);
  return total_bboxes;
}

void Mtcnn::RefineNetwork(const ncnn::Mat & image, vector<Mtcnn::_BBox> & _bboxes)
{
  if (_bboxes.empty())
    return;

  vector<int> keep;
#ifdef USE_OPENMP
   #pragma omp parallel for
#endif
  for (int i = 0; i < _bboxes.size(); i++) {
    _BBox & _bbox = _bboxes[i];
    ncnn::Mat pad = PadCrop(image, _bbox.x1, _bbox.y1, _bbox.x2, _bbox.y2);
    ncnn::Mat input;
    ncnn::resize_bilinear(pad, input, 24, 24);
    ncnn::Extractor ex = Rnet.create_extractor();
    ex.input("data", input);
    ncnn::Mat conf_blob, loc_blob;
    ex.extract("prob1", conf_blob);
    ex.extract("fc5-2", loc_blob);
    float score = conf_blob.channel(0)[1];
    if (score >= thresholds[1]) {
      _bbox.score = score;
      for (int j = 0; j < 4; j++)
        _bbox.regs[j] = loc_blob.channel(0)[j];
      keep.push_back(i);
    }
  }
  if (keep.size() < _bboxes.size()) {
    for (int i = 0; i < keep.size(); i++) {
      if (i < keep[i])
        _bboxes[i] = _bboxes[keep[i]];
    }
    _bboxes.erase(_bboxes.begin() + keep.size(), _bboxes.end());
  }

  NonMaximumSuppression(_bboxes, 0.7f, IoU);
  BoxRegression(_bboxes, true);
}

void Mtcnn::OutputNetwork(const ncnn::Mat & image, vector<Mtcnn::_BBox> & _bboxes)
{
  if (_bboxes.empty())
    return;

  vector<int> keep;
#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
  for (int i = 0; i < _bboxes.size(); i++) {
    _BBox & _bbox = _bboxes[i];
    ncnn::Mat pad = PadCrop(image, _bbox.x1, _bbox.y1, _bbox.x2, _bbox.y2);
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
      _bbox.score = score;
      for (int j = 0; j < 4; j++)
        _bbox.regs[j] = loc_blob.channel(0)[j];
      // facial landmarks
      int w = _bbox.x2 - _bbox.x1;
      int h = _bbox.y2 - _bbox.y1;
      for (int j = 0; j < 5; j++) {
        _bbox.fpoints[j] = kpt_blob.channel(0)[j] * w + _bbox.x1;
        _bbox.fpoints[j + 5] = kpt_blob.channel(0)[j + 5] * h + _bbox.y1;
      }
      keep.push_back(i);
    }
  }
  if (keep.size() < _bboxes.size()) {
    for (int i = 0; i < keep.size(); i++) {
      if (i < keep[i])
        _bboxes[i] = _bboxes[keep[i]];
    }
    _bboxes.erase(_bboxes.begin() + keep.size(), _bboxes.end());
  }

  BoxRegression(_bboxes, false);
  NonMaximumSuppression(_bboxes, 0.7f, IoM);
}

void Mtcnn::LandmarkNetwork(const ncnn::Mat & image, vector<Mtcnn::_BBox> & _bboxes)
{
  if (_bboxes.empty())
    return;

#ifdef USE_OPENMP
  #pragma omp parallel for
#endif
  for (auto & _bbox : _bboxes) {
    int patchw = std::max<int>(_bbox.x2 - _bbox.x1, _bbox.y2 - _bbox.y1);
    patchw = fix(patchw * 0.25f);
    if (patchw % 2 == 1)
      patchw += 1;
    ncnn::Mat input;
    input.create(24, 24, image.c * 5, image.elemsize);
    for (int i = 0; i < 5; i++) {
      _bbox.fpoints[i] = round(_bbox.fpoints[i]);
      _bbox.fpoints[i+5] = round(_bbox.fpoints[i+5]);
      int x1 = fix(_bbox.fpoints[i]) - patchw / 2;
      int y1 = fix(_bbox.fpoints[i+5]) - patchw / 2;
      int x2 = x1 + patchw;
      int y2 = y1 + patchw;
      ncnn::Mat pad = PadCrop(image, x1, y1, x2, y2);
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
        _bbox.fpoints[i] += off_x * patchw;
        _bbox.fpoints[i+5] += off_y * patchw;
      }
    }
  }
}
