#include <Python.h>
#include <sstream>

// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
#include "mtcnn.h"

#if CV_VERSION_MAJOR == 3

namespace py = boost::python;

//===================    MACROS    =================================================================
static PyObject* opencv_error = 0;
#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}

//===================   ERROR HANDLING     =========================================================

static int failmsg(const char *fmt, ...) {
  char str[1000];

  va_list ap;
  va_start(ap, fmt);
  vsnprintf(str, sizeof(str), fmt, ap);
  va_end(ap);

  throw std::runtime_error(str);
  return 0;
}

//===================   THREADING     ==============================================================

class PyAllowThreads {
public:
  PyAllowThreads() :
      _state(PyEval_SaveThread()) {
  }
  ~PyAllowThreads() {
    PyEval_RestoreThread(_state);
  }
private:
  PyThreadState* _state;
};

class PyEnsureGIL {
public:
  PyEnsureGIL() :
      _state(PyGILState_Ensure()) {
  }
  ~PyEnsureGIL() {
    PyGILState_Release(_state);
  }
private:
  PyGILState_STATE _state;
};

enum {
  ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2
};

static size_t REFCOUNT_OFFSET = (size_t)&(((PyObject*)0)->ob_refcnt) +
    (0x12345678 != *(const size_t*)"\x78\x56\x34\x12\0\0\0\0\0")*sizeof(int);

static inline PyObject* pyObjectFromRefcount(const int* refcount)
{
  return (PyObject*)((size_t)refcount - REFCOUNT_OFFSET);
}

static inline int* refcountFromPyObject(const PyObject* obj)
{
  return (int*)((size_t)obj + REFCOUNT_OFFSET);
}

//===================   NUMPY ALLOCATOR FOR OPENCV     =============================================
class NumpyAllocator: public cv::MatAllocator {
public:
  NumpyAllocator() {
    stdAllocator = cv::Mat::getStdAllocator();
  }
  ~NumpyAllocator() {
  }

  cv::UMatData* allocate(PyObject* o, int dims, const int* sizes, int type,
      size_t* step) const {
    cv::UMatData* u = new cv::UMatData(this);
    u->data = u->origdata = (uchar*) PyArray_DATA((PyArrayObject*) o);
    npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
    for (int i = 0; i < dims - 1; i++)
      step[i] = (size_t) _strides[i];
    step[dims - 1] = CV_ELEM_SIZE(type);
    u->size = sizes[0] * step[0];
    u->userdata = o;
    return u;
  }

  cv::UMatData* allocate(int dims0, const int* sizes, int type, void* data,
      size_t* step, int flags, cv::UMatUsageFlags usageFlags) const {
    if (data != 0) {
      CV_Error(cv::Error::StsAssert, "The data should normally be NULL!");
      // probably this is safe to do in such extreme case
      return stdAllocator->allocate(dims0, sizes, type, data, step, flags,
          usageFlags);
    }
    PyEnsureGIL gil;

    int depth = CV_MAT_DEPTH(type);
    int cn = CV_MAT_CN(type);
    const int f = (int) (sizeof(size_t) / 8);
    int typenum =
        depth == CV_8U ? NPY_UBYTE :
        depth == CV_8S ? NPY_BYTE :
        depth == CV_16U ? NPY_USHORT :
        depth == CV_16S ? NPY_SHORT :
        depth == CV_32S ? NPY_INT :
        depth == CV_32F ? NPY_FLOAT :
        depth == CV_64F ?
                  NPY_DOUBLE :
                  f * NPY_ULONGLONG + (f ^ 1) * NPY_UINT;
    int i, dims = dims0;
    cv::AutoBuffer<npy_intp> _sizes(dims + 1);
    for (i = 0; i < dims; i++)
      _sizes[i] = sizes[i];
    if (cn > 1)
      _sizes[dims++] = cn;
    PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
    if (!o)
      CV_Error_(cv::Error::StsError,
          ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
    return allocate(o, dims0, sizes, type, step);
  }

  bool allocate(cv::UMatData* u, int accessFlags,
      cv::UMatUsageFlags usageFlags) const {
    return stdAllocator->allocate(u, accessFlags, usageFlags);
  }

  void deallocate(cv::UMatData* u) const {
    if (u) {
      PyEnsureGIL gil;
      PyObject* o = (PyObject*) u->userdata;
      Py_XDECREF(o);
      delete u;
    }
  }

  const MatAllocator* stdAllocator;
};

//===================   ALLOCATOR INITIALIZTION   ==================================================
NumpyAllocator g_numpyAllocator;

//===================   STANDALONE CONVERTER FUNCTIONS     =========================================
PyObject* fromMatToNDArray(const cv::Mat& m) {
  if (!m.data)
    Py_RETURN_NONE;
    cv::Mat temp,
  *p = (cv::Mat*) &m;
  if (!p->u || p->allocator != &g_numpyAllocator) {
    temp.allocator = &g_numpyAllocator;
    ERRWRAP2(m.copyTo(temp));
    p = &temp;
  }
  PyObject* o = (PyObject*) p->u->userdata;
  Py_INCREF(o);
  return o;
}

cv::Mat fromNDArrayToMat(PyObject* o) {
  cv::Mat m;
  bool allowND = true;
  if (!PyArray_Check(o)) {
    std::runtime_error("argument is not a numpy array");
    if (!m.data)
      m.allocator = &g_numpyAllocator;
  } else {
    PyArrayObject* oarr = (PyArrayObject*) o;

    bool needcopy = false, needcast = false;
    int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
    int type = typenum == NPY_UBYTE ? CV_8U : typenum == NPY_BYTE ? CV_8S :
          typenum == NPY_USHORT ? CV_16U :
          typenum == NPY_SHORT ? CV_16S :
          typenum == NPY_INT ? CV_32S :
          typenum == NPY_INT32 ? CV_32S :
          typenum == NPY_FLOAT ? CV_32F :
          typenum == NPY_DOUBLE ? CV_64F : -1;

    if (type < 0) {
      if (typenum == NPY_INT64 || typenum == NPY_UINT64
          || type == NPY_LONG) {
        needcopy = needcast = true;
        new_typenum = NPY_INT;
        type = CV_32S;
      } else {
        std::runtime_error("Argument data type is not supported");
        m.allocator = &g_numpyAllocator;
        return m;
      }
    }

#ifndef CV_MAX_DIM
    const int CV_MAX_DIM = 32;
#endif

    int ndims = PyArray_NDIM(oarr);
    if (ndims >= CV_MAX_DIM) {
      std::runtime_error("Dimensionality of argument is too high");
      if (!m.data)
        m.allocator = &g_numpyAllocator;
      return m;
    }

    int size[CV_MAX_DIM + 1];
    size_t step[CV_MAX_DIM + 1];
    size_t elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(oarr);
    const npy_intp* _strides = PyArray_STRIDES(oarr);
    bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;

    for (int i = ndims - 1; i >= 0 && !needcopy; i--) {
      // these checks handle cases of
      //  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
      //  b) transposed arrays, where _strides[] elements go in non-descending order
      //  c) flipped arrays, where some of _strides[] elements are negative
      if ((i == ndims - 1 && (size_t) _strides[i] != elemsize)
          || (i < ndims - 1 && _strides[i] < _strides[i + 1]))
        needcopy = true;
    }

    if (ismultichannel && _strides[1] != (npy_intp) elemsize * _sizes[2])
      needcopy = true;

    if (needcopy) {

      if (needcast) {
        o = PyArray_Cast(oarr, new_typenum);
        oarr = (PyArrayObject*) o;
      } else {
        oarr = PyArray_GETCONTIGUOUS(oarr);
        o = (PyObject*) oarr;
      }

      _strides = PyArray_STRIDES(oarr);
    }

    for (int i = 0; i < ndims; i++) {
      size[i] = (int) _sizes[i];
      step[i] = (size_t) _strides[i];
    }

    // handle degenerate case
    if (ndims == 0) {
      size[ndims] = 1;
      step[ndims] = elemsize;
      ndims++;
    }

    if (ismultichannel) {
      ndims--;
      type |= CV_MAKETYPE(0, size[2]);
    }

    if (ndims > 2 && !allowND) {
      std::runtime_error("%s has more than 2 dimensions");
    } else {
      m = cv::Mat(ndims, size, type, PyArray_DATA(oarr), step);
      m.u = g_numpyAllocator.allocate(o, ndims, size, type, step);
      m.addref();

      if (!needcopy) {
        Py_INCREF(o);
      }
    }
    m.allocator = &g_numpyAllocator;
  }
  return m;
}

void showInfo(const cv::Mat image) {
    int b = image.at<cv::Vec3b>(50,50)[0];
    int g = image.at<cv::Vec3b>(50,50)[1];
    int r = image.at<cv::Vec3b>(50,50)[2];
    std::cout << "pixel at <50,50>: " << b << " " << g << " " << r << std::endl;
    std::cout << "image shape: " << image.rows << " " << image.cols << std::endl;
}

//===================   MTCNN WRAPPER FOR PYTHON BINGING     =============================================

using namespace std;
using namespace cv;
using namespace face;
using namespace caffe;
class MTcnn : private Mtcnn {
 public:
  MTcnn(const string model_dir, bool preciseLandmark) : 
    Mtcnn(model_dir, preciseLandmark) {}

  void setFactor(const float factor) {
    if (factor <= 0 || factor >= 1)
      failmsg("Invalid factor to set: %f.", factor);
    else
      this->scale_factor = factor;
  }
  double getFactor() const {
    return this->scale_factor;
  }

  void setMinSize(const int minSize) {
    if (minSize <= 0)
      failmsg("Invalid minSize to set: %d.", minSize);
    else
      this->face_min_size = minSize;
  }
  int getMinSize() const {
    return this->face_min_size;
  }

  void setMaxSize(const int maxSize) {
    if (maxSize <= 0)
      failmsg("Invalid maxSize to set: %d.", maxSize);
    else
      this->face_max_size = maxSize;
  }
  int getMaxSize() const {
    return this->face_max_size;
  }

  void setPreciseLandmark(const bool preciseLandmark) {
    // empty pass
    // this->enable_precise_landmark = this->enable_precise_landmark && preciseLandmark;
  }
  int getPreciseLandmark() const {
    return this->enable_precise_landmark;
  }

  void setThresholds(const py::list& thresholds) {
    if (len(thresholds) != 3)
      failmsg("Invalid thresholds to set: len = %d.", len(thresholds));
    else {
      for (int i = 0; i < 3; ++i)
        this->thresholds[i] = py::extract<float>(thresholds[i]);
    }
  }
  py::list getThresholds() const {
    py::list thres;
    for (int i = 0; i < 3; ++i)
      thres.append(this->thresholds[i]);
    return thres;
  }

  py::list detect(PyObject* np, bool preciseLandmark) {
    cv::Mat img = fromNDArrayToMat(np);
    vector<FaceInfo> infos = Detect(img, preciseLandmark);
    py::list pyinfos;
    for (const auto & info : infos)
    {
      pyinfos.append(info.score);   // score
      pyinfos.append(info.bbox.x1); // x1
      pyinfos.append(info.bbox.y1); // y1
      pyinfos.append(info.bbox.x2); // x2
      pyinfos.append(info.bbox.y2); // y2
      // l&r eye, nose, l&r mouth
      for (const auto & pt : info.fpts)
      {
        pyinfos.append(pt.x); // x
        pyinfos.append(pt.y); // y
      }
    }
    return pyinfos;
  }

  py::list detect_again(PyObject* np, const py::list& py_bbox, bool preciseLandmark) {
    cv::Mat img = fromNDArrayToMat(np);
    //! Todo Debug why?
    // Wrok wll
    BBox bbox = BBox(py::extract<float>(py_bbox[0]),
                     py::extract<float>(py_bbox[1]),
                     py::extract<float>(py_bbox[2]),
                     py::extract<float>(py_bbox[3]));
    // Report Error
    /* BBox bbox(py::extract<float>(py_bbox[0]), */
    /*           py::extract<float>(py_bbox[1]), */
    /*           py::extract<float>(py_bbox[2]), */
    /*           py::extract<float>(py_bbox[3])); */
    vector<FaceInfo> infos = DetectAgain(img, bbox, preciseLandmark);
    py::list pyinfos;
    for (const auto & info : infos)
    {
      pyinfos.append(info.score);   // score
      pyinfos.append(info.bbox.x1); // x1
      pyinfos.append(info.bbox.y1); // y1
      pyinfos.append(info.bbox.x2); // x2
      pyinfos.append(info.bbox.y2); // y2
      // l&r eye, nose, l&r mouth
      for (const auto & pt : info.fpts)
      {
        pyinfos.append(pt.x); // x
        pyinfos.append(pt.y); // y
      }
    }
    return pyinfos;
  }

  PyObject* align(PyObject* np, const py::list& py_fpts, int width=112) {
    Mat img = fromNDArrayToMat(np);
    FPoints fpts;
    fpts.reserve(5);
    fpts.emplace_back(py::extract<float>(py_fpts[0]), py::extract<float>(py_fpts[1]));
    fpts.emplace_back(py::extract<float>(py_fpts[2]), py::extract<float>(py_fpts[3]));
    fpts.emplace_back(py::extract<float>(py_fpts[4]), py::extract<float>(py_fpts[5]));
    fpts.emplace_back(py::extract<float>(py_fpts[6]), py::extract<float>(py_fpts[7]));
    fpts.emplace_back(py::extract<float>(py_fpts[8]), py::extract<float>(py_fpts[9]));
    Mat face = Align(img, fpts, width);
    PyObject* py_face = fromMatToNDArray(face);
    return py_face;
  }

  PyObject* draw_infos(PyObject* np, const py::list& py_infos) {
    Mat img = fromNDArrayToMat(np);
    int num = py::len(py_infos);
    std::vector<FaceInfo> infos;
    for (int i = 0; i < num; i += 15)
    {
      float score = py::extract<float>(py_infos[i]);
      BBox bbox(py::extract<float>(py_infos[i+1]),
                py::extract<float>(py_infos[i+2]),
                py::extract<float>(py_infos[i+3]),
                py::extract<float>(py_infos[i+4]));
      FPoints fpts;
      fpts.reserve(5);
      fpts.emplace_back(py::extract<float>(py_infos[i+5]), py::extract<float>(py_infos[i+6]));
      fpts.emplace_back(py::extract<float>(py_infos[i+7]), py::extract<float>(py_infos[i+8]));
      fpts.emplace_back(py::extract<float>(py_infos[i+9]), py::extract<float>(py_infos[i+10]));
      fpts.emplace_back(py::extract<float>(py_infos[i+11]), py::extract<float>(py_infos[i+12]));
      fpts.emplace_back(py::extract<float>(py_infos[i+13]), py::extract<float>(py_infos[i+14]));
      infos.emplace_back(move(bbox), score, move(fpts));
    }
    Mat canvas = DrawInfos(img, infos);
    PyObject* py_canvas = fromMatToNDArray(canvas);
    return py_canvas;
  }

};

//================== Useful Setting Functions
//=======================================================
// Selecting mode.

void InitLog(int level) {
  FLAGS_logtostderr = 1;
  FLAGS_minloglevel = level;
  ::google::InitGoogleLogging("");
  ::google::InstallFailureSignalHandler();
}
void InitLogInfo() {
  InitLog(google::INFO);
}
void Log(const string& s) {
  LOG(INFO) << s;
}

//===================   Expose Python Module     =============================================

#if (PY_VERSION_HEX >= 0x03000000)
  static void *init_ar() {
#else
  static void init_ar(){
#endif
    Py_Initialize();

    import_array();
    return NUMPY_IMPORT_ARRAY_RETVAL;
  }

BOOST_PYTHON_MODULE(_mtcnn) {
  init_ar();
  py::def("init_log", &InitLog);
  py::def("init_log", &InitLogInfo);
  py::def("log", &Log);
  py::def("close_log", &::google::ShutdownGoogleLogging);
  py::def("set_device", &Caffe::SetDevice);
  py::class_<MTcnn>("MTcnn", py::init<std::string, bool>())
    .add_property("factor", &MTcnn::getFactor, &MTcnn::setFactor)
    .add_property("minSize", &MTcnn::getMinSize, &MTcnn::setMinSize)
    .add_property("maxSize", &MTcnn::getMaxSize, &MTcnn::setMaxSize)
    .add_property("thresholds", &MTcnn::getThresholds, &MTcnn::setThresholds)
    .add_property("preciseLandmark", &MTcnn::getPreciseLandmark, &MTcnn::setPreciseLandmark)
    .def("detect", &MTcnn::detect)
    .def("detect_again", &MTcnn::detect_again)
    .def("align", &MTcnn::align)
    .def("draw_infos", &MTcnn::draw_infos);
}
 
#endif
