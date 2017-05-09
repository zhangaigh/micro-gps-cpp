#include "image.h"
#include "image_func.h"

// SiftGPU
#include "SiftGPU.h"
#ifdef ON_MAC
#include "OpenGL/gl.h"
#else
#include "GL/gl.h"
#endif


namespace MicroGPS {

SiftGPU g_sift_gpu;
bool siftgpu_initialized = false;
void initSiftGPU() {
  char * sift_gpu_argv[] ={(char*)"-t", (char*)"0", (char*)"-v", (char*)"0", (char*)"-cuda"};
  g_sift_gpu.ParseParam(5, sift_gpu_argv); 
  int support = g_sift_gpu.CreateContextGL();
  if(support != SiftGPU::SIFTGPU_FULL_SUPPORTED) {
    return; 
  } else {
    printf("SiftGPU supported\n");
    siftgpu_initialized = true;
  }
}

// constructors
Image::Image() :
        m_width(0),
        m_height(0),
        m_channels(0),
        m_data(NULL)  
{
  strcpy(m_image_path, "");
  strcpy(m_precomputed_feat_path, "");
  strcpy(m_precomputed_sift_path, "");
}

Image::Image(const char* image_path) :
        m_width(0),
        m_height(0),
        m_channels(0),
        m_data(NULL)  
{
  strcpy(m_image_path, image_path);
  strcpy(m_precomputed_feat_path, "");
  strcpy(m_precomputed_sift_path, "");
}

Image::Image(const char* image_path, 
             const char* precomputed_feat_path, 
             const char* precomputed_sift_path) :
              m_width(0),
              m_height(0),
              m_channels(0),
              m_data(NULL)  
{
  strcpy(m_image_path, image_path);  
  strcpy(m_precomputed_feat_path, precomputed_feat_path);
  strcpy(m_precomputed_sift_path, precomputed_sift_path);
}


Image::Image(const size_t width, const size_t height, const size_t channels) :
              m_width(width),
              m_height(height),
              m_channels(channels),
              m_data(new uchar[width * height * channels])  
{
  strcpy(m_image_path, "");
  strcpy(m_precomputed_feat_path, "");
  strcpy(m_precomputed_sift_path, "");
}

// destructor
Image::~Image() {
  this->release();
}

// load / release buffer
void Image::loadImage() {
  printf("Image::loadImage(): loading %s\n", m_image_path);
  cv::Mat bgr = cv::imread(m_image_path, CV_LOAD_IMAGE_UNCHANGED);
  printf("Image::loadImage(): image loaded: %d x %d x %d\n", bgr.rows, bgr.cols, bgr.channels());

  m_width = bgr.cols;
  m_height = bgr.rows;
  m_channels = bgr.channels();
  
  // reset m_data anyway
  if (m_data) {
    delete[] m_data;
    m_data = NULL;
  }
  m_data = new uchar[m_width * m_height * m_channels];
  memcpy(m_data, bgr.data, m_width * m_height * m_channels);
}

void Image::create(const size_t width, const size_t height, const size_t channels) {
  m_width = width;
  m_height = height;
  m_channels = channels;

  if (m_data) {
    delete[] m_data;
    m_data = NULL;
  }
  m_data = new uchar[m_width * m_height * m_channels];
}

void Image::release() {
  if (m_data) {
    delete[] m_data;
    m_data = NULL;
  }

  for (int i = 0; i < m_local_features.size(); i++) {
    delete m_local_features[i];  
  }
  m_local_features.clear();

  // Don't clear other info!
}

// access functions
const char* Image::getImagePath() const {
  return m_image_path;
}

size_t Image::width() const {
  return m_width;
}

size_t Image::height() const {
  return m_height;
}

size_t Image::channels() const {
  return m_channels;
}

uchar* Image::data() const {
  return m_data;
}

uchar& Image::getPixel(unsigned row, unsigned col, unsigned ch) const {
  return m_data[row * m_width * m_channels + col * m_channels + ch];
}

uchar& Image::operator() (unsigned row, unsigned col, unsigned ch) const {
  return getPixel(row, col, ch);
}

void Image::write(const char* image_path) const{
  cv::Mat cv_im = convertToCvMat();
  printf("Image::write(): writing image to %s\n", image_path);
  cv::imwrite(image_path, cv_im);
}

void Image::show(const char* win_name) const{
  cv::Mat cv_im = convertToCvMat();
  cv::imshow(win_name, cv_im);
  cv::waitKey(0);
}

// basic processing
float Image::bilinearSample (float y, float x, unsigned ch) const {  
  // p11 ----------- p12
  //  |      lyy      |
  //  | lxx (x,y) uxx |
  //  |      uyy      |
  // p21 ----------- p22

  float lx = floor(x);
  float ux = lx + 1.0f;
  float ly = floor(y);
  float uy = ly + 1.0f;
  
  float p11, p12, p21, p22;
  float lxx, uxx, lyy, uyy;
  
  p11 = (float)getPixel(ly, lx, ch); 
  // printf("p11 = %f\n", p11);
  p12 = (float)getPixel(ly, ux, ch);  
  p21 = (float)getPixel(uy, lx, ch);  
  p22 = (float)getPixel(uy, ux, ch);  
  
  lxx = x - lx;
  uxx = ux - x;
  lyy = y - ly;
  uyy = uy - y;

  float sum = p11 * uyy * uxx + 
              p12 * uyy * lxx +
              p21 * lyy * uxx + 
              p22 * lyy * lxx;
      
  return sum;
}

void Image::bgr2gray() {
  if (m_channels != 3) {
    printf("Image::bgr2gray(): num channels != 3\n");
    exit(-1);
  }

  // MATLAB formula: 0.2989 * R + 0.5870 * G + 0.1140 * B 
  uchar* data_new = new uchar[m_width * m_height];
  for (size_t y = 0; y < m_height; y++) {
    for (size_t x = 0; x < m_width; x++) {
      data_new[y * m_width + x] = 0.1140 * (float) m_data[(y * m_width + x) * 3 + 0] + 
                                  0.5870 * (float) m_data[(y * m_width + x) * 3 + 1] +
                                  0.2989 * (float) m_data[(y * m_width + x) * 3 + 2];
    }
  }
  
  delete[] m_data;
  m_data = data_new;
  m_channels = 1;
}

void Image::resize(const float ratio) {
  cv::Mat cv_im = convertToCvMat();
  cv::Mat cv_im_small;
  cv::resize(cv_im, cv_im_small, cv::Size(), ratio, ratio);

  printf("WorkImage::resize(): resized to %d x %d x %d\n", cv_im_small.rows, 
                                                           cv_im_small.cols, 
                                                           cv_im_small.channels());

  m_width = (size_t)cv_im_small.cols;
  m_height = (size_t)cv_im_small.rows;

  uchar* m_data_old = m_data;
  m_data = new uchar[m_width * m_height * m_channels];
  memcpy(m_data, cv_im_small.data, m_width * m_height * m_channels);
  delete[] m_data_old; // clear old buffer
  m_data_old = NULL;
}

void Image::rotate90(bool clockwise) {
  uchar* m_data_old = m_data;
  m_data = new uchar[m_width * m_height * m_channels];
  
  // copy elements to the new array
  size_t new_width = m_height;
  size_t new_height = m_width;
  for (size_t y = 0; y < new_height; y++) {
    for (size_t x = 0; x < new_width; x++) {
      size_t old_x;
      size_t old_y;
      if (!clockwise) {
        old_x = new_height - 1 - y;
        old_y = x;        
      } else {
        old_x = y;
        old_y = new_width - 1 - x;        
      }
      for (size_t c = 0; c < m_channels; c++) {
        m_data[(y * new_width + x) * m_channels + c] = 
                m_data_old[(old_y * m_width + old_x) * m_channels + c];
      }
    }
  }
  m_width = new_width;
  m_height = new_height;

  delete[] m_data_old;
  m_data_old = NULL;
}

Image* Image::clone() const {
  Image* image_cloned = new Image(m_width, m_height, m_channels);
  memcpy(image_cloned->data(), m_data, m_height * m_width * m_channels);
  return image_cloned;
}

Image* Image::cropPatch(const float x, const float y, const float orientation,
                        const int win_width, const int win_height) const {
  unsigned n_channels = this->channels();
  Image* patch = new Image (win_width, win_height, n_channels);
  uchar* patch_data = patch->data();
  
  float half_w = (float)win_width / 2;
  float half_h = (float)win_height / 2;

  // rotated x-axis
  float x_axis_x = cos(orientation);
  float x_axis_y = sin(orientation);

  // rotated y-axis
  float y_axis_x = -sin(orientation);
  float y_axis_y = cos(orientation);

  // [-half_w + 0.5, half_w - 0.5]
  unsigned pixel_cnt = 0;
  for (float patch_y = - half_h + 0.5; patch_y <= half_h - 0.5; patch_y += 1.0) {
    for (float patch_x = - half_w + 0.5; patch_x <= half_w - 0.5; patch_x += 1.0) {
      float sample_x = x + patch_x * x_axis_x + patch_y * y_axis_x;
      float sample_y = y + patch_x * x_axis_y + patch_y * y_axis_y;

      if (sample_x < 0 || sample_x >= this->width()-1 || sample_y < 0 || sample_y >= this->height()-1) {
        delete patch;
        printf("Image::cropPatch(): not valid patch\n");
        return NULL;
      }

      for (unsigned ch = 0; ch < n_channels; ch++) {
        patch_data[pixel_cnt * n_channels + ch] = (uchar)(this->bilinearSample(sample_y, sample_x, ch));
      }
      pixel_cnt++;
    }
  }

  return patch; 
}

void Image::extractSIFT(float downsampling) {
  if (!siftgpu_initialized) {
    printf("Image::extractSIFT(): SiftGPU is not initialized.\n");
    exit(-1);
    return;
  }
 
  cv::Mat im_converted;

  if (m_channels == 3) {
    cv::Mat im = convertToCvMat();
    cv::cvtColor(im, im_converted, CV_BGR2GRAY);
    cv::resize(im_converted, im_converted, cv::Size(), downsampling, downsampling);
  } else {
    cv::Mat im = convertToCvMat();
    cv::resize(im, im_converted, cv::Size(), downsampling, downsampling);  
  }

  printf("start running sift gpu\n");
  g_sift_gpu.RunSIFT (im_converted.cols, im_converted.rows, im_converted.data, GL_LUMINANCE, GL_UNSIGNED_BYTE);   
  printf("finish running sift\n");
  int num = g_sift_gpu.GetFeatureNum();//get feature count
  printf("detected %d features\n", num);

  // std::vector<float> descriptors(128*num);
  float* descriptors = new float[128 * num]; 
  std::vector<SiftGPU::SiftKeypoint> keypoints(num);
  //read back keypoints and normalized descritpros
  //specify NULL if you don’t need keypionts or descriptors
  g_sift_gpu.GetFeatureVector(&keypoints[0], &descriptors[0]);

  // delete original features if there's any
  for (int i = 0; i < m_local_features.size(); i++) {
    delete m_local_features[i];  
  }
  m_local_features.resize(num);


  for (int i = 0; i < num; i++) {
    // printf("reading feature %d\n", i);
    LocalFeature* f = new LocalFeature;
    f->x = keypoints[i].x / downsampling;
    f->y = keypoints[i].y / downsampling;
    f->angle = -keypoints[i].o; //siftgpu uses different conventions, we need to flip the sign of the angle
    f->scale = keypoints[i].s / downsampling;
    f->strength = 0.0f;
    f->descriptor = Eigen::Map<Eigen::RowVectorXf>((float*)(descriptors + 128 * i), 128);

    // double angle = -keypoints[i].o; 

    // compute local pose
    f->local_pose(0, 0) = cos(f->angle);
    f->local_pose(1, 0) = sin(f->angle);
    f->local_pose(2, 0) = 0;
    f->local_pose(0, 1) = -sin(f->angle);
    f->local_pose(1, 1) = cos(f->angle);
    f->local_pose(2, 1) = 0;
    f->local_pose(0, 2) = f->x;
    f->local_pose(1, 2) = f->y;
    f->local_pose(2, 2) = 1;

    m_local_features[i] = f;
  }
  delete[] descriptors;

  printf("Image::extractSIFT(): detected %lu keypoints\n", m_local_features.size());

}

void Image::saveLocalFeatures(const char* path, const int num_samples, const float margin_thresh) {
  FILE* fp = fopen(path, "w");

  std::vector<LocalFeature*> valid_features;
  int num_features = m_local_features.size();
  for (int i = 0; i < num_features; i++) {
    LocalFeature* f = m_local_features[i];
    if (f->x > margin_thresh && f->x < m_width-1-margin_thresh &&
        f->y > margin_thresh && f->y < m_height-1-margin_thresh) {
      valid_features.push_back(f);
    }
  }

  // randsample features
  num_features = valid_features.size();
  std::vector<int> sel;
  util::randomSample(num_features, num_samples, sel);
  num_features = sel.size();

  int feat_dim;
  if (num_features > 0) {
    feat_dim = valid_features[0]->descriptor.size();
  } else {
    feat_dim = 0;
  }

  fwrite(&num_features, sizeof(int), 1, fp);
  fwrite(&feat_dim, sizeof(int), 1, fp);

  for (int i = 0; i < num_features; i++) {
    LocalFeature* f = valid_features[sel[i]];
    float kp[4];
    kp[0] = f->x;
    kp[1] = f->y;
    kp[2] = f->scale;
    kp[3] = f->angle;
    fwrite(kp, sizeof(float), 4, fp);   
    fwrite(f->descriptor.data(), sizeof(float), f->descriptor.size(), fp);    
  }

  fclose(fp);
}

bool Image::loadPrecomputedFeatures(const bool load_sift) {
  char* path;
  if (load_sift) {
    path = (char*)m_precomputed_sift_path;
  } else {
    path = (char*)m_precomputed_feat_path;
  }
  
  if (strcmp(path, "") == 0) {
    printf("cannot read external feature!\n");
    return false;
  }
  printf("Image::loadPrecomputedFeatures(): loading external features %s\n", path);
  
  FILE* fp = fopen(path, "r");
  // char image_path[256];
  // fgets(image_path , sizeof(image_path) , fp); 
  // printf("features computed from image %s", image_path);

  int size[2];
  fread(size, sizeof(int), 2, fp);
  //fread(size, sizeof(int), 1, fp);
  //size[1] = 128;
  printf("size[] = %d x %d\n", size[0], size[1]);

  int chunk_size = size[1] + 4; // 4 floats for locations
  float* buffer = new float[size[0] * chunk_size];  
  fread(buffer, sizeof(float), chunk_size * size[0], fp);
  fclose(fp);

  // clear the buffer
  for (size_t i = 0; i < m_local_features.size(); i++) {
    delete m_local_features[i];
  }
  m_local_features.resize(size[0]);

  // assign values  
  for (int i = 0; i < size[0]; i++) {
    LocalFeature* f = new LocalFeature;
    f->x = buffer[i * chunk_size + 0];
    f->y = buffer[i * chunk_size + 1];
    f->scale = buffer[i * chunk_size + 2];
    f->angle = buffer[i * chunk_size + 3];
    if (f->scale < 1.0) {
      printf("%d: scale = %f!\n", i, f->scale);
      exit(-1);
    }
    f->descriptor = Eigen::Map<Eigen::RowVectorXf>(buffer + chunk_size * i + 4, size[1]);

    // printf("i = %d\n", i);
    f->local_pose(0, 0) = cos(f->angle);
    f->local_pose(1, 0) = sin(f->angle);
    f->local_pose(2, 0) = 0;
    f->local_pose(0, 1) = -sin(f->angle);
    f->local_pose(1, 1) = cos(f->angle);
    f->local_pose(2, 1) = 0;
    f->local_pose(0, 2) = f->x;
    f->local_pose(1, 2) = f->y;
    f->local_pose(2, 2) = 1;

    m_local_features[i] = f;
  }
  delete[] buffer;  

  return true;
}

size_t Image::getNumLocalFeatures() {
  return m_local_features.size();
}

LocalFeature* Image::getLocalFeature(size_t idx) {
  return m_local_features[idx];
}


void Image::linearFeatureCompression(const Eigen::MatrixXf& basis) {
  for (size_t i = 0; i < m_local_features.size(); i++) {
    LocalFeature* f = m_local_features[i];
    f->descriptor_compressed = f->descriptor * basis; // still row vector
  }
  printf("linearFeatureCompression: feature dimension reduced\n");
}


// private helpers
cv::Mat Image::convertToCvMat() const{
  // cv::Mat cv_im;
  // if (m_channels == 3) {
  //   cv_im = cv::Mat(m_height, m_width, CV_8UC3, m_data);
  // } else {
  //   cv_im = cv::Mat(m_height, m_width, CV_8UC1, m_data);
  // }

  int cv_mat_type;
  switch (m_channels) {
    case 0:
      cv_mat_type = CV_8UC1;
      break;
    case 1:
      cv_mat_type = CV_8UC2;
      break;
    case 3:
      cv_mat_type = CV_8UC3;
      break;
    case 4:
      cv_mat_type = CV_8UC4;
      break;
    default:
      printf("Image::resize(): invalid number of channels\n");
      exit(-1);
  } 

  cv::Mat cv_im = cv::Mat(m_height, m_width, cv_mat_type, m_data);

  return cv_im;   
}



}
