// C
#include <ctime>
#include <cstdlib>
// C++
#include <iostream>
// Libraries
// #define FLANN_USE_CUDA
#include "flann/flann.h"
// Project
#include "work_image.h"
#include "database.h"
#include "micro_gps.h"

extern "C" {
#include "vl/sift.h"
}

std::string type2str2(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
    case CV_8U:
      r = "8U";
      break;
    case CV_8S:
      r = "8S";
      break;
    case CV_16U:
      r = "16U";
      break;
    case CV_16S:
      r = "16S";
      break;
    case CV_32S:
      r = "32S";
      break;
    case CV_32F:
      r = "32F";
      break;
    case CV_64F:
      r = "64F";
      break;
    default:
      r = "User";
      break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}

int test_sift(int argc, char const *argv[]) {
  char image_path[256];
  int n_features = 0;
  int n_octave_layers = 3;
  double contrast_threshold = 0.04f;
  double edge_threshold = 10.0f;
  double sigma = 1.6f;
  double ratio = 0.5f;

  argc--;
  argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-i")) {
        argc--;
        argv++;
        sprintf(image_path, "%s", (*argv));
      } else if (!strcmp(*argv, "-f")) {
        argc--;
        argv++;
        n_features = atoi(*argv);
      } else if (!strcmp(*argv, "-o")) {
        argc--;
        argv++;
        n_octave_layers = atoi(*argv);
      } else if (!strcmp(*argv, "-c")) {
        argc--;
        argv++;
        contrast_threshold = atof(*argv);
      } else if (!strcmp(*argv, "-o")) {
        argc--;
        argv++;
        edge_threshold = atof(*argv);
      } else if (!strcmp(*argv, "-s")) {
        argc--;
        argv++;
        sigma = atof(*argv);
      } else if (!strcmp(*argv, "-r")) {
        argc--;
        argv++;
        ratio = atof(*argv);
      } else {
        printf("invalid argument, exiting... \n");
        exit(EXIT_FAILURE);
      }
    }
    argc--;
    argv++;
  }

  cv::SIFT sift_detector(n_features, n_octave_layers, contrast_threshold,
                         edge_threshold, sigma);

  cv::Mat bgr = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
  cv::Mat gray_small;
  cv::cvtColor(bgr, gray_small, CV_BGR2GRAY);
  cv::resize(gray_small, gray_small, cv::Size(), ratio, ratio);

  std::vector<cv::KeyPoint> keypoints_raw;
  cv::Mat descriptors;
  sift_detector(gray_small, cv::noArray(), keypoints_raw, descriptors, false);

  // printf("weakest response = %f\n");
  std::vector<cv::KeyPoint> keypoints(keypoints_raw.size());
  int cnt = 0;
  for (int i = 0; i < keypoints_raw.size(); i++) {
    // printf("response = %f\n", keypoints_raw[i].response);
    if (keypoints_raw[i].response > 0.025f) {
      keypoints[cnt++] = keypoints_raw[i];
    }
  }
  keypoints.resize(cnt);

  printf("detected %lu keypoints\n", keypoints_raw.size());
  printf("keeping %lu keypoints\n", keypoints.size());

  printf("descriptor matrix size: %d x %d, type = %s\n", descriptors.rows,
         descriptors.cols, type2str2(descriptors.type()).c_str());

  int n_keypoints = keypoints.size();
  FILE *fp = fopen("opencv_keypoints.bin", "w");
  fwrite(&n_keypoints, sizeof(int), 1, fp);
  for (int i = 0; i < n_keypoints; i++) {
    fwrite(&keypoints[i].pt.x, sizeof(float), 1, fp);
    fwrite(&keypoints[i].pt.y, sizeof(float), 1, fp);
    fwrite(&keypoints[i].angle, sizeof(float), 1, fp);
    fwrite(&keypoints[i].size, sizeof(float), 1, fp);
    fwrite(&keypoints[i].response, sizeof(float), 1, fp);
  }

  fwrite(descriptors.data, sizeof(float), n_keypoints * 128, fp);

  fclose(fp);

  return 0;
}

int test_inpolygon(int argc, char const *argv[]) {
  double px[5] = {0, 0.5, 1, 0.5, 0};
  double py[5] = {0, 0.3, 0.5, 1, 0.5};
  double cx[4] = {0, 1, 1, 0};
  double cy[4] = {0, 0, 1, 1};
  bool points_in_on[5];
  bool points_in[5];
  bool points_on[5];
  inpolygon(5, px, py, cx, cy, points_in_on, points_in, points_on);

  for (int i = 0; i < 5; i++) {
    printf("points_in_on[i] = %d, points_in[i] = %d, points_on[i] = %d\n",
           points_in_on[i], points_in[i], points_on[i]);
  }
}

int test_vlsift(int argc, char const *argv[]) {
  char image_path[256];
  int n_features = 0;
  int n_octave_layers = 3;
  double contrast_threshold = 0.04f;
  double edge_threshold = 10.0f;
  double sigma = 1.6f;
  double ratio = 0.5f;

  argc--;
  argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-i")) {
        argc--;
        argv++;
        sprintf(image_path, "%s", (*argv));
      } else if (!strcmp(*argv, "-f")) {
        argc--;
        argv++;
        n_features = atoi(*argv);
      } else if (!strcmp(*argv, "-o")) {
        argc--;
        argv++;
        n_octave_layers = atoi(*argv);
      } else if (!strcmp(*argv, "-c")) {
        argc--;
        argv++;
        contrast_threshold = atof(*argv);
      } else if (!strcmp(*argv, "-o")) {
        argc--;
        argv++;
        edge_threshold = atof(*argv);
      } else if (!strcmp(*argv, "-s")) {
        argc--;
        argv++;
        sigma = atof(*argv);
      } else if (!strcmp(*argv, "-r")) {
        argc--;
        argv++;
        ratio = atof(*argv);
      } else {
        printf("invalid argument, exiting... \n");
        exit(EXIT_FAILURE);
      }
    }
    argc--;
    argv++;
  }

  // cv::Mat bgr = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
  // cv::Mat gray_small;
  // cv::cvtColor(bgr, gray_small, CV_BGR2GRAY);
  // cv::resize(gray_small, gray_small, cv::Size(), ratio, ratio);

  cv::Mat gray_small = cv::imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);


  printf("image size: %d x %d\n", gray_small.rows, gray_small.cols);

  bool first = 1;
  vl_bool  err;
  VlSiftFilt* filt = 0 ;
  vl_sift_pix* fdata = 0 ;
  vl_uint8* data = 0;

  data = (vl_uint8*)malloc(gray_small.cols * gray_small.rows *
                sizeof(vl_uint8));
  fdata = (vl_sift_pix*)malloc(gray_small.cols * gray_small.rows *
                  sizeof(vl_sift_pix));

  filt = vl_sift_new(gray_small.cols, gray_small.rows, -1, 3, 0); // -1 is default I believe
  // vl_sift_set_edge_thresh(filt, -1);
  // vl_sift_set_peak_thresh(filt, -1);
  // vl_sift_set_magnif(filt, -1);

  printf("sift: filter settings:\n");
  printf("sift:   octaves      (O)     = %d\n", vl_sift_get_noctaves(filt));
  printf("sift:   levels       (S)     = %d\n", vl_sift_get_nlevels(filt));
  printf("sift:   first octave (o_min) = %d\n",
          vl_sift_get_octave_first(filt));
  printf("sift:   edge thresh           = %g\n",
          vl_sift_get_edge_thresh(filt));
  printf("sift:   peak thresh           = %g\n",
          vl_sift_get_peak_thresh(filt));
  printf("sift:   norm thresh           = %g\n",
          vl_sift_get_norm_thresh(filt));
  printf("sift:   magnif                = %g\n", vl_sift_get_magnif(filt));
  printf("sift:   window size           = %g\n", vl_sift_get_window_size(filt));
  
  // printf("sift: will source frames? %s\n", ikeys ? "yes" : "no");
  // printf("sift: will force orientations? %s\n",
  //         force_orientations ? "yes" : "no");
  
  for (int q = 0; q < (unsigned)(gray_small.cols * gray_small.rows); ++q) {
    fdata[q] = (float)(gray_small.data[q]);
  }

  
  int total_num_features = 0;

  while (1) {
    VlSiftKeypoint const *keys = 0;
    int nkeys;

    printf("vl_sift: processing octave %d\n", vl_sift_get_octave_index (filt));

    if (first) {
      first = 0;
      err = vl_sift_process_first_octave(filt, fdata);
    } else {
      err = vl_sift_process_next_octave(filt);
    }

    if (err) {
      err = VL_ERR_OK;
      break;
    }

    vl_sift_detect(filt);
    keys = vl_sift_get_keypoints(filt);
    nkeys = vl_sift_get_nkeypoints(filt);
    // i = 0;

    printf("sift: detected %d (unoriented) keypoints\n", nkeys);
   
    for (int i = 0; i < nkeys; ++i) {
      double angles[4];
      int nangles;
      VlSiftKeypoint ik;
      VlSiftKeypoint const *k;

      /* obtain keypoint orientations ........................... */
      k = keys + i;
      nangles = vl_sift_calc_keypoint_orientations(filt, angles, k);

      /* for each orientation ................................... */
      for (int q = 0; q < (unsigned)nangles; ++q) {
        vl_sift_pix descr[128];

        /* compute descriptor (if necessary) */
        vl_sift_calc_keypoint_descriptor(filt, descr, k, angles[q]);
        total_num_features++;
      }
    }
  }

  printf("total number of features = %d\n", total_num_features);

  if (filt) {
    vl_sift_delete(filt);
    filt = 0;
  }

  /* release image data */
  if (fdata) {
    free(fdata);
    fdata = 0;
  }

  /* release image data */
  if (data) {
    free(data);
    data = 0;
  }
}

int test_eigen_pca(int argc, char const *argv[]) {
  char input_path[256];
  int m = 5;
  int n = 5;

  argc--;
  argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-i")) {
        argc--;
        argv++;
        sprintf(input_path, "%s", (*argv));
      } else if (!strcmp(*argv, "-m")) {
        argc--;
        argv++;
        m = atoi(*argv);
      } else if (!strcmp(*argv, "-n")) {
        argc--;
        argv++;
        n = atoi(*argv);
      }
    }
    argc--;
    argv++;
  }

  FILE* fp = fopen(input_path, "r");
  Eigen::MatrixXf data(m, n);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      fscanf(fp, "%f", &data(i, j));
    }
  }
  fclose(fp);
  std::cout << "data = \n" << data << std::endl;
 
 
  Eigen::MatrixXf mean_deducted = data.rowwise() - data.colwise().mean();

  Eigen::JacobiSVD<Eigen::MatrixXf> svd(mean_deducted, Eigen::ComputeThinV);

  Eigen::MatrixXf m_PCA_basis = svd.matrixV().leftCols(5);
  printf("MicroGPS::preprocessDatabase(): PCA basis size: %ld x %ld\n", m_PCA_basis.rows(), m_PCA_basis.cols());


  std::cout << "pca_basis = \n" << m_PCA_basis << std::endl;
  
}


int test_gps (int argc, char const *argv[]) {
  char dataset_root[] = "/Users/lgzhang/Documents/DATA/micro_gps_packed";
  char PCA_basis_root[] = "pca_bases";
  char database_root[] = "databases";

  // offline data
  char dataset_name[] = "fc_hallway_long_packed";
  char PCA_basis_name[] = "fc_dims8_pca_basis.bin";
  char database_name[] = "fc_dims8_ss50.bin";
  
  // test params
  float cell_size = 50.0f;
  int test_index = 0;



 argc--;
  argv++;
  while (argc > 0) {
    if ((*argv)[0] == '-') {
      if (!strcmp(*argv, "-i")) {
        argc--;
        argv++;
        test_index = atoi(*argv);
      } else if (!strcmp(*argv, "-m")) {
        argc--;
        argv++;
        // m = atoi(*argv);
      } else if (!strcmp(*argv, "-n")) {
        argc--;
        argv++;
        // n = atoi(*argv);
      }
    }
    argc--;
    argv++;
  }


  // load dataset
  char dataset_path[256];
  sprintf(dataset_path, "%s/%s", dataset_root, dataset_name);

  Database* dataset = new Database(dataset_path);
  dataset->loadDatabase();
  dataset->loadDefaultTestSequence();

  MicroGPS* micro_gps = new MicroGPS();
  micro_gps->loadDatabaseOnly(dataset);

  micro_gps->setVotingCellSize(cell_size);

  // reload precomputed values
  char s[256];
  sprintf(s, "%s/%s", PCA_basis_root, PCA_basis_name);
  micro_gps->loadPCABasis(s);
  sprintf(s, "%s/%s", database_root, database_name);
  micro_gps->loadFeatures(s);
  micro_gps->buildSearchIndex();

  // test
  WorkImage* current_test_frame = new WorkImage(dataset->getTestImage(test_index));
  current_test_frame->loadImage();

  WorkImage* alignment_image = NULL;
  MicroGPSTiming timing;
  MicroGPSDebug debug;
  MicroGPSOptions options;
  MicroGPSResult result;

  bool success_flag = micro_gps->locate(current_test_frame, alignment_image,
                                                      options, result,
                                                      timing, debug);
  delete current_test_frame;

  printf("------------------ REPORT ------------------\n");
  printf("Final test result = %s\n", success_flag? "SUCCESSFUL" : "FAILED");
  printf("Total time: %.3lf ms\n", timing.total * 1000.0f);
  printf("SIFT extraction : %.3lf ms\n", timing.sift_extraction * 1000.0f);
  printf("KNN search : %.3lf ms\n", timing.knn_search * 1000.0f);
  printf("Compute candidate poses : %.3lf ms\n", timing.candidate_image_pose * 1000.0f);
  printf("Voting : %.3lf ms\n", timing.voting * 1000.0f);
  printf("RANSAC : %.3lf ms\n", timing.ransac * 1000.0f);

}


int main(int argc, char const *argv[]) {
  argc--;
  argv++;

  printf("%s\n", *argv);
  if (!strcmp(*argv, "-test_sift")) {
    test_sift(argc, argv);
  } else if (!strcmp(*argv, "-test_inpolygon")) {
    test_inpolygon(argc, argv);
  } else if (!strcmp(*argv, "-test_vlsift")) {
    test_vlsift(argc, argv);
  } else if (!strcmp(*argv, "-test_eigen_pca")) {
    test_eigen_pca(argc, argv);
  } else if (!strcmp(*argv, "-test_gps")) {
    test_gps(argc, argv);
  } else {
  }

  return 0;
}
