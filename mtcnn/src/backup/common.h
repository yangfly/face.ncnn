#ifndef FACE_COMMON_H_
#define FACE_COMMON_H_

namespace face
{

/// @brief Point standalone
struct Point
{
  float x, y;
  explicit Point(float x = 0, float y = 0) : x(x), y(y) {}
};

/// @brief Bounding Box
struct BBox
{
  float x1, y1, x2, y2;
  explicit BBox(float x1 = 0, float y1 = 0, float x2 = 0, float y2 = 0) :
    x1(x1), y1(y1), x2(x2), y2(y2) {}
  BBox(const Point & pt1, const Point & pt2) :
    x1(pt1.x), y1(pt1.y), x2(pt2.x), y2(pt2.y) {};
  bool empty() { return x1 > x2 || y1 > y2; }
};

/// @ brief Facial Points
using FPoints = std::vector<Point>;

struct FaceInfo
{
public:
  BBox bbox;		// bounding box
  float score;	// face confidence 	
  FPoints fpts;	// facial landmarks
  FaceInfo(BBox && bbox, float score) :
    bbox(bbox), score(score) 
  {
    fpts.reserve(5);
  }
  FaceInfo(BBox && bbox, float score, FPoints && fpts) : 
    bbox(bbox), score(score), fpts(fpts)
  {
    fpts.reserve(5);
  }
};

} //  namespace face

#endif // FACE_COMMON_H_
