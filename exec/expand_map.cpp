#include "micro_gps.h"
#include <sys/stat.h>
#include <gflags/gflags.h>
#include "refine_poses.h"


#if ON_MAC
  DEFINE_string(dataset_root, "/Users/lgzhang/Documents/DATA/micro_gps_packed", "dataset_root");
#endif
#if ON_AMD64
  DEFINE_string(dataset_root, "/data/linguang/micro_gps_packed", "dataset_root");
#endif

DEFINE_string(dataset, "acee_unloading_long_packed", "dataset to use");
DEFINE_string(testset, "sequence_expand_map.test", "test sequence");
DEFINE_double(sift_extraction_scale, 0.5, "extract sift at this scale");
DEFINE_string(feature_database, "acee_unloading_siftgpu.bin", "database features");
DEFINE_string(output, "features/acee_unloading_long_packed-database-sift", "where to output the features");

void mkdirIfNotExists(const char* path) {
  struct stat buffer;   
  if (stat (path, &buffer) != 0) {
    printf("%s doesn't exist!\n", path);
    char cmd[256];
    sprintf(cmd, "mkdir %s", path);
    system(cmd);
  } else {
    printf("%s exists, ignore...\n", path);
  }
}

bool checkFileExists(char* path) {
  struct stat buffer;   
  if (stat (path, &buffer) != 0) {
    return false;
  }
  return true;
}

bool computeCorrespondences(WorkImage& prev_frame, WorkImage& curr_frame,
                            Eigen::MatrixXf& loc_a, Eigen::MatrixXf& loc_b,
                            Eigen::MatrixXf& pose_prev_curr) {

  std::vector<int> matched_idx1;
  std::vector<int> matched_idx2;

  // WorkImage prev_frame(dataset->getTestImage(i-1)); 
  // WorkImage curr_frame(dataset->getTestImage(i)); 

  prev_frame.loadImage();
  curr_frame.loadImage();

  std::vector<int> inliers;
  prev_frame.siftMatch(&curr_frame, matched_idx1, matched_idx2);
  if (matched_idx1.size() >= 7) {
    prev_frame.siftMatchEstimatePose(&curr_frame, matched_idx1, matched_idx2,
                                      pose_prev_curr, inliers);
  }

  // printf("num inliers = %d\n", inliers.size());
  if (inliers.size() <= 7) {
    pose_prev_curr = Eigen::MatrixXf::Identity(3, 3);
    return false;
  } else {
    loc_a = Eigen::MatrixXf(2, inliers.size());
    loc_b = Eigen::MatrixXf(2, inliers.size());
    for (int k = 0; k < inliers.size(); k++) {
      SIFTFeature* f1 = prev_frame.getSIFTFeature(matched_idx1[inliers[k]]);
      SIFTFeature* f2 = curr_frame.getSIFTFeature(matched_idx2[inliers[k]]);
      loc_a(0, k) = f1->x;
      loc_a(1, k) = f1->y;
      loc_b(0, k) = f2->x;
      loc_b(1, k) = f2->y;
    }
  }
  return true;
}

void loadPoses (const char* path,
                Eigen::MatrixXf& x_y_angle_array) {
  // read time based_pairs
  FILE* fp = fopen(path, "r");

  int num_frames;
  // read poses  
  fread(&num_frames, sizeof(int), 1, fp);
  x_y_angle_array.resize(3, num_frames);
  fread(x_y_angle_array.data(), sizeof(float), 3 * num_frames, fp);
  
  fclose(fp);  
}


void loadPairs (const char* path,
                std::vector<int>& pose_id_a_array,
                std::vector<int>& pose_id_b_array,
                std::vector<Eigen::MatrixXf>& sift_loc_a_array,
                std::vector<Eigen::MatrixXf>& sift_loc_b_array) {
  // read time based_pairs
  FILE* fp = fopen(path, "r");

  // read correspondences
  int num_pairs;
  fread(&num_pairs, sizeof(int), 1, fp);
  pose_id_a_array.resize(num_pairs);
  pose_id_b_array.resize(num_pairs);
  sift_loc_a_array.resize(num_pairs);
  sift_loc_b_array.resize(num_pairs);
  for (int i = 0; i < num_pairs; i++) {
    int num_correspondences;
    fread(&num_correspondences, sizeof(int), 1, fp);
    fread(&pose_id_a_array[i], sizeof(int), 1, fp);
    fread(&pose_id_b_array[i], sizeof(int), 1, fp);
    sift_loc_a_array[i].resize(2, num_correspondences);
    sift_loc_b_array[i].resize(2, num_correspondences);
    fread(sift_loc_a_array[i].data(), sizeof(float), num_correspondences * 2, fp);
    fread(sift_loc_b_array[i].data(), sizeof(float), num_correspondences * 2, fp);
  }
  fclose(fp);  
}

void savePoses (const char* path,
                Eigen::MatrixXf& x_y_angle_array) {
  FILE* fp = fopen(path, "w");
  // write poses
  int num_frames = x_y_angle_array.cols();
  fwrite(&num_frames, sizeof(int), 1, fp);
  fwrite(x_y_angle_array.data(), sizeof(float), num_frames * 3, fp);
  
  fclose(fp);
}

void savePairs (const char* path,
                std::vector<int>& pose_id_a_array,
                std::vector<int>& pose_id_b_array,
                std::vector<Eigen::MatrixXf>& sift_loc_a_array,
                std::vector<Eigen::MatrixXf>& sift_loc_b_array) {

  FILE* fp = fopen(path, "w");

  int num_pairs = sift_loc_a_array.size();
  fwrite(&num_pairs, sizeof(int), 1, fp);
  for (int i = 0; i < num_pairs; i++) {
    int num_correspondences = sift_loc_a_array[i].cols();
    fwrite(&num_correspondences, sizeof(int), 1, fp);
    fwrite(&pose_id_a_array[i], sizeof(int), 1, fp);
    fwrite(&pose_id_b_array[i], sizeof(int), 1, fp);
    fwrite(sift_loc_a_array[i].data(), sizeof(float), num_correspondences * 2, fp);
    fwrite(sift_loc_b_array[i].data(), sizeof(float), num_correspondences * 2, fp);
  }
  fclose(fp);  
}


void convertFromMinimalParamsToPoses(Eigen::MatrixXf& x_y_angle_array,
                                      std::vector<Eigen::Matrix3f>& poses) {
  
  int num_poses = x_y_angle_array.cols();
  poses.resize(num_poses);

  for (int i = 0; i < num_poses; i++) {
    float x = x_y_angle_array(0, i);
    float y = x_y_angle_array(1, i);
    float angle = x_y_angle_array(2, i);

    poses[i] = Eigen::MatrixXf::Identity(3, 3);
    poses[i](0, 0) = cos(angle);
    poses[i](1, 0) = sin(angle);
    poses[i](0, 1) = -sin(angle);
    poses[i](1, 1) = cos(angle);
    poses[i](0, 2) = x;
    poses[i](1, 2) = y;
  }
}

void convertFromPosesToMinimalParams(std::vector<Eigen::Matrix3f>& poses, 
                                      Eigen::MatrixXf& x_y_angle_array) {

  int num_poses = poses.size();
  x_y_angle_array.resize(3, num_poses);

  for (int i = 0; i < num_poses; i++) {
    x_y_angle_array(0, i) = poses[i](0, 2);
    x_y_angle_array(1, i) = poses[i](1, 2);
    x_y_angle_array(2, i) = atan2(poses[i](1, 0), poses[i](0, 0));
  }
}






int main (int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

#ifdef USE_SIFT_GPU
  initSiftGPU();
#endif

  mkdirIfNotExists("expand");
  
  // load dataset
  char dataset_path[256];
  sprintf(dataset_path, "%s/%s", FLAGS_dataset_root.c_str(), FLAGS_dataset.c_str());
  Database* dataset = new Database(dataset_path);
  dataset->loadTestSequenceByName(FLAGS_testset.c_str());

  int num_frames = dataset->getTestSize();
  // int num_frames = 12;
  std::vector<int> pose_id_a_array;
  std::vector<int> pose_id_b_array;
  std::vector<Eigen::MatrixXf> sift_loc_a_array;
  std::vector<Eigen::MatrixXf> sift_loc_b_array;
  Eigen::MatrixXf x_y_angle_array(3, num_frames);
  std::vector<Eigen::Matrix3f> time_based_poses(num_frames);


  if (!checkFileExists("expand/time_based_pairs.bin")) {

    // time based matching
    time_based_poses[0] = Eigen::MatrixXf::Identity(3, 3);  

    for (int i = 1; i < num_frames; i++) {
      WorkImage prev_frame(dataset->getTestImage(i-1)); 
      WorkImage curr_frame(dataset->getTestImage(i)); 

      Eigen::MatrixXf loc_a;
      Eigen::MatrixXf loc_b;
      Eigen::MatrixXf pose_prev_curr;
      bool success = computeCorrespondences(prev_frame, curr_frame,
                                            loc_a, loc_b,
                                            pose_prev_curr);

      time_based_poses[i] = time_based_poses[i-1] * pose_prev_curr;

      if (success) {
        pose_id_a_array.push_back(i-1);
        pose_id_b_array.push_back(i);
        sift_loc_a_array.push_back(loc_a);
        sift_loc_b_array.push_back(loc_b);
        x_y_angle_array(0, i) = time_based_poses[i](0, 2);
        x_y_angle_array(1, i) = time_based_poses[i](1, 2);
        x_y_angle_array(2, i) = atan2(time_based_poses[i](1, 0), time_based_poses[i](0, 0));
      } else {
        printf("matching %d-th and %d-th frames failed!\n", i-1, i);
      }
      std::cout << time_based_poses[i] << std::endl;
    }

    // save time based pairs
    savePoses("expand/time_based_poses.bin",
              x_y_angle_array);
    savePairs("expand/time_based_pairs.bin",
              pose_id_a_array, pose_id_b_array,
              sift_loc_a_array, sift_loc_b_array);


  } else {
    loadPoses("expand/time_based_poses.bin",
              x_y_angle_array);
    loadPairs("expand/time_based_pairs.bin",
              pose_id_a_array, pose_id_b_array,
              sift_loc_a_array, sift_loc_b_array);
  }


  convertFromMinimalParamsToPoses(x_y_angle_array, time_based_poses);
  
  if (!checkFileExists("expand/time-based.png")) {
    std::vector<WorkImage*> test_image_array(num_frames);
    for (int i = 0; i < num_frames; i++) {
      test_image_array[i] = new WorkImage(dataset->getTestImage(i));
    }
    WorkImage* refined_map = warpImageArray(test_image_array, time_based_poses, 0.25);
    refined_map->write("expand/time-based.png");
    delete refined_map;
  }



  std::vector<int> pose_id_a_array_loop_closure;
  std::vector<int> pose_id_b_array_loop_closure;
  std::vector<Eigen::MatrixXf> sift_loc_a_array_loop_closure;
  std::vector<Eigen::MatrixXf> sift_loc_b_array_loop_closure;



  if (!checkFileExists("expand/loop_closure_pairs.bin")) {

    // compute distance
    float dist_thresh = 1000.0f;
    float dist_thresh_far = 5000.0f;

    int num_trials = 0;
    for (int i = 0; i < num_frames; i++) {
      for (int j = i+2; j < num_frames; j++) {
        float frame_dist = (x_y_angle_array.block(0, i, 2, 1) - x_y_angle_array.block(0, j, 2, 1)).norm();
        if (abs(i - j) > 30 && frame_dist < dist_thresh_far || 
            abs(i - j) <= 3 && frame_dist < dist_thresh) {
          num_trials++;
          printf("%d: examing frame %d and %d\n", num_trials, i, j);

          WorkImage frame_i(dataset->getTestImage(i)); 
          WorkImage frame_j(dataset->getTestImage(j)); 

          Eigen::MatrixXf loc_a;
          Eigen::MatrixXf loc_b;
          Eigen::MatrixXf pose_ij;
          bool success = computeCorrespondences(frame_i, frame_j,
                                                loc_a, loc_b,
                                                pose_ij);

          if (success) {
            pose_id_a_array_loop_closure.push_back(i);
            pose_id_b_array_loop_closure.push_back(j);
            sift_loc_a_array_loop_closure.push_back(loc_a);
            sift_loc_b_array_loop_closure.push_back(loc_b);
          }
        }
      }
    }

    savePairs("expand/loop_closure_pairs.bin",
              pose_id_a_array_loop_closure, pose_id_b_array_loop_closure,
              sift_loc_a_array_loop_closure, sift_loc_b_array_loop_closure);

  } else {
    loadPairs("expand/loop_closure_pairs.bin",
              pose_id_a_array_loop_closure, pose_id_b_array_loop_closure,
              sift_loc_a_array_loop_closure, sift_loc_b_array_loop_closure);
  }



  if (!checkFileExists("expand/refined_poses.bin")) {
    std::vector<alignmentPair*> all_pairs;
    std::vector<double*> all_poses;

    appendAlignmentPairs(pose_id_a_array, pose_id_b_array,
                        sift_loc_a_array,sift_loc_b_array,
                        all_pairs);

    appendAlignmentPairs(pose_id_a_array_loop_closure, pose_id_b_array_loop_closure,
                        sift_loc_a_array_loop_closure,sift_loc_b_array_loop_closure,
                        all_pairs);


    for (int i = 0; i < pose_id_a_array_loop_closure.size(); i++) {
      printf("id_a = %d, id_b = %d\n", pose_id_a_array_loop_closure[i], pose_id_b_array_loop_closure[i]);
      printf("x=%f, y=%f\n", sift_loc_a_array_loop_closure[i](0, 0), sift_loc_a_array_loop_closure[i](1, 0));
      printf("x=%f, y=%f\n", sift_loc_b_array_loop_closure[i](0, 0), sift_loc_b_array_loop_closure[i](1, 0));
    }

    for (int i = 0; i < x_y_angle_array.cols(); i++) {
      printf("x=%f, y=%f, angle=%f\n", x_y_angle_array(0, i), x_y_angle_array(1, i), x_y_angle_array(2, i));
    }
    
    convertPoses(x_y_angle_array, all_poses);
    printf("num pairs: %d\n", all_pairs.size());
    printf("num poses: %d\n", all_poses.size());
    refinePoses(all_pairs, all_poses);

    convertPosesBack(all_poses, x_y_angle_array);
    
    savePoses("expand/refined_poses.bin", x_y_angle_array);
  } else {
    loadPoses("expand/refined_poses.bin", x_y_angle_array);
  }


  // stitching
  std::vector<Eigen::Matrix3f> refined_poses;
  convertFromMinimalParamsToPoses(x_y_angle_array, refined_poses);

  if (!checkFileExists("expand/refined.png")) {
    std::vector<WorkImage*> test_image_array(num_frames);
    for (int i = 0; i < num_frames; i++) {
      test_image_array[i] = new WorkImage(dataset->getTestImage(i));
    }
    WorkImage* refined_map = warpImageArray(test_image_array, refined_poses, 0.25);
    refined_map->write("expand/refined.png");
    delete refined_map;
  }

#define USE_GPS
#ifdef USE_GPS
  Eigen::MatrixXf database_x_y_angle_array;
  Eigen::MatrixXf database_test_x_y_angle_array;
  std::vector<Eigen::Matrix3f> database_image_poses;
  std::vector<Eigen::Matrix3f> database_test_poses;
  std::vector<int> pose_id_a_array_gps;
  std::vector<int> pose_id_b_array_gps;
  std::vector<Eigen::MatrixXf> sift_loc_a_array_gps;
  std::vector<Eigen::MatrixXf> sift_loc_b_array_gps;

  dataset->loadDatabase();
  if (!checkFileExists("expand/gps_pairs.bin")) {
    // TODO: include micro-gps and compute hard constraints
    // init micro-gps

    MicroGPS* micro_gps = new MicroGPS();
    micro_gps->loadDatabaseOnly(dataset);

    micro_gps->setVotingCellSize(50.0f);
    micro_gps->setNumScaleGroups(10);
    micro_gps->loadDatabaseOnly(dataset);

    char s[256];
    sprintf(s, "databases/%s", FLAGS_feature_database.c_str());
    micro_gps->loadFeatures(s);
    micro_gps->computePCABasis();
    micro_gps->PCAreduction(16);
    micro_gps->buildSearchIndexMultiScales();

    MicroGPSOptions options;
    MicroGPSResult result;
    MicroGPSTiming timing;
    MicroGPSDebug debug;

    options.do_alignment = false;
    options.do_siftmatch = true;
    options.debug_mode = true;
    options.confidence_thresh = 0.8f;
    
    for (int i = 0; i < num_frames; i++ ){
      WorkImage current_test_frame(dataset->getTestImage(i));
      current_test_frame.loadImage();

      timing.reset();
      result.reset();
      debug.reset();  
      WorkImage* alignment_image = NULL;
      bool success_flag = micro_gps->locate(&current_test_frame, alignment_image,
                                                          options, result,
                                                          timing, debug);
      if (!success_flag) {
        continue;      
      }

      WorkImage database_frame(dataset->getDatabaseImage(debug.closest_database_image_idx));

      Eigen::MatrixXf loc_a;
      Eigen::MatrixXf loc_b;
      Eigen::MatrixXf pose_database_curr;
      bool flag = computeCorrespondences(database_frame, current_test_frame,
                                        loc_a, loc_b,
                                        pose_database_curr);

      if (flag) {
        database_image_poses.push_back(dataset->getDatabasePose(debug.closest_database_image_idx));
        database_test_poses.push_back(pose_database_curr); // duplicated poses are ok
        pose_id_a_array_gps.push_back((int)database_image_poses.size()-1);
        pose_id_b_array_gps.push_back(i);
        sift_loc_a_array_gps.push_back(loc_a);      
        sift_loc_b_array_gps.push_back(loc_b);      
      }
    }

    convertFromPosesToMinimalParams(database_image_poses, database_x_y_angle_array);
    convertFromPosesToMinimalParams(database_test_poses, database_test_x_y_angle_array);

    savePoses("expand/gps_database_poses.bin", database_x_y_angle_array);
    savePoses("expand/gps_database_test_poses.bin", database_test_x_y_angle_array);

    savePairs("expand/gps_pairs.bin",
            pose_id_a_array_gps, pose_id_b_array_gps,
            sift_loc_a_array_gps, sift_loc_b_array_gps);

  } else {
    loadPoses ("expand/gps_database_poses.bin", database_x_y_angle_array);
    loadPoses ("expand/gps_database_test_poses.bin", database_test_x_y_angle_array);

    loadPairs("expand/gps_pairs.bin",
            pose_id_a_array_gps, pose_id_b_array_gps,
            sift_loc_a_array_gps, sift_loc_b_array_gps);
  }

  printf("num of matched database images: %d\n", database_x_y_angle_array.cols());

  if (database_x_y_angle_array.cols() <= 0) {
    printf("cannot register to the database!\n");
    exit(-1);
  } 

  convertFromMinimalParamsToPoses(database_x_y_angle_array, database_image_poses);
  convertFromMinimalParamsToPoses(database_test_x_y_angle_array, database_test_poses);


  // T_Ww = T_Wd * T_dt * T_tw
  Eigen::Matrix3f T_register = database_image_poses[0] * database_test_poses[0] * refined_poses[pose_id_b_array_gps[0]].inverse();

  // register test images
  for (int i = 0; i < refined_poses.size(); i++) {
    refined_poses[i] = T_register * refined_poses[i];
  }


  // search for more constraints
  if (!checkFileExists("expand/gps_pairs_more.bin")) {
    float dist_thresh = 2000.0f;
    for (int i = 0; i < dataset->getDatabaseSize(); i++ ){
      for (int j = 0; j < num_frames; j++) {
        Eigen::Matrix3f t_pose = refined_poses[j];
        Eigen::Matrix3f d_pose = dataset->getDatabasePose(i);

        if ((t_pose.block(0, 2, 2, 1) - d_pose.block(0, 2, 2, 1)).norm() < dist_thresh) {
          WorkImage frame_i(dataset->getDatabaseImage(i)); 
          WorkImage frame_j(dataset->getTestImage(j)); 

          Eigen::MatrixXf loc_a;
          Eigen::MatrixXf loc_b;
          Eigen::MatrixXf pose_ij;
          bool success = computeCorrespondences(frame_i, frame_j,
                                                loc_a, loc_b,
                                                pose_ij);

          if (success) {
            database_image_poses.push_back(dataset->getDatabasePose(i));
            database_test_poses.push_back(pose_ij); // duplicated poses are ok
            pose_id_a_array_gps.push_back((int)database_image_poses.size()-1);
            pose_id_b_array_gps.push_back(j);
            sift_loc_a_array_gps.push_back(loc_a);      
            sift_loc_b_array_gps.push_back(loc_b);      
          }
        }
      }
    }

    convertFromPosesToMinimalParams(database_image_poses, database_x_y_angle_array);
    convertFromPosesToMinimalParams(database_test_poses, database_test_x_y_angle_array);

    savePoses ("expand/gps_database_poses_more.bin", database_x_y_angle_array);
    savePoses ("expand/gps_database_test_poses_more.bin", database_test_x_y_angle_array);
    
    savePairs("expand/gps_pairs_more.bin",
            pose_id_a_array_gps, pose_id_b_array_gps,
            sift_loc_a_array_gps, sift_loc_b_array_gps);
  } else {
    loadPoses ("expand/gps_database_poses_more.bin", database_x_y_angle_array);
    loadPoses ("expand/gps_database_test_poses_more.bin", database_test_x_y_angle_array);

    loadPairs("expand/gps_pairs_more.bin",
            pose_id_a_array_gps, pose_id_b_array_gps,
            sift_loc_a_array_gps, sift_loc_b_array_gps);
  }


  convertFromPosesToMinimalParams(refined_poses, x_y_angle_array);

  // compute bounds for test images
  float extra_range = 4000.0f;
  Eigen::MatrixXf test_images_bounds_min = x_y_angle_array.rowwise().minCoeff();
  Eigen::MatrixXf test_images_bounds_max = x_y_angle_array.rowwise().maxCoeff();
  float stitching_min_x = test_images_bounds_min(0) - extra_range;
  float stitching_min_y = test_images_bounds_min(1) - extra_range;
  float stitching_max_x = test_images_bounds_max(0) + extra_range;
  float stitching_max_y = test_images_bounds_max(1) + extra_range;

  std::cout << test_images_bounds_min << std::endl;
  std::cout << test_images_bounds_max << std::endl;

  printf("bounds = %f %f %f %f\n", stitching_min_x, stitching_max_x, stitching_min_y, stitching_max_y);


  std::vector<int> database_image_idx_for_stitching;
  for (int i = 0; i < dataset->getDatabaseSize(); i++) {
    Eigen::Matrix3f p = dataset->getDatabasePose(i);
    float x = p(0, 2);
    float y = p(1, 2);
    // printf("x = %f, y = %f\n", x, y);
    if (x > stitching_min_x && x < stitching_max_x &&
        y > stitching_min_y && y < stitching_max_y) {
      database_image_idx_for_stitching.push_back(i);
    }
  }

  printf("num of database images = %d\n", database_image_idx_for_stitching.size());
  // exit(0);
  if (!checkFileExists("expand/register.png")) {
    std::vector<WorkImage*> image_array(database_image_idx_for_stitching.size() + num_frames);
    std::vector<Eigen::Matrix3f> image_poses_for_stitching(database_image_idx_for_stitching.size() + num_frames);
    int cnt = 0;
    for (int i = 0; i < database_image_idx_for_stitching.size(); i++) {
      image_array[cnt] = new WorkImage(dataset->getDatabaseImage(database_image_idx_for_stitching[i]));
      image_poses_for_stitching[cnt] = dataset->getDatabasePose(database_image_idx_for_stitching[i]);
      cnt++;
    }
    printf("added %d frames\n", cnt);
    for (int i = 0; i < num_frames; i++) {
      image_array[cnt] = new WorkImage(dataset->getTestImage(i));
      image_poses_for_stitching[cnt] = refined_poses[i];
      cnt++;
    }
    
    WorkImage* register_map = warpImageArray(image_array, image_poses_for_stitching, 0.25);
    register_map->write("expand/register.png");
    delete register_map;
  }


  std::vector<alignmentPair*> all_pairs;
  std::vector<double*> all_poses;

  std::vector<alignmentPair*> all_hard_constraint_pairs;
  std::vector<double*> all_hard_constraint_poses;

  appendAlignmentPairs(pose_id_a_array, pose_id_b_array,
                      sift_loc_a_array, sift_loc_b_array,
                      all_pairs);

  appendAlignmentPairs(pose_id_a_array_loop_closure, pose_id_b_array_loop_closure,
                    sift_loc_a_array_loop_closure,sift_loc_b_array_loop_closure,
                    all_pairs);

  appendAlignmentPairs(pose_id_a_array_gps, pose_id_b_array_gps,
                      sift_loc_a_array_gps, sift_loc_b_array_gps,
                      all_hard_constraint_pairs);
  
  convertPoses(x_y_angle_array, all_poses); // redo this since we have updated x_y_angle_array;
  convertPoses(database_x_y_angle_array, all_hard_constraint_poses);

  refinePosesWithHardConstraints (all_pairs, all_hard_constraint_pairs,
                                  all_poses, all_hard_constraint_poses);

  
  convertPosesBack(all_poses, x_y_angle_array);
  convertFromMinimalParamsToPoses(x_y_angle_array, refined_poses);

  if (!checkFileExists("expand/refined_register.png")) {
    std::vector<WorkImage*> image_array(database_image_idx_for_stitching.size() + num_frames);
    std::vector<Eigen::Matrix3f> image_poses_for_stitching(database_image_idx_for_stitching.size() + num_frames);
    int cnt = 0;
    for (int i = 0; i < database_image_idx_for_stitching.size(); i++) {
      image_array[cnt] = new WorkImage(dataset->getDatabaseImage(database_image_idx_for_stitching[i]));
      image_poses_for_stitching[cnt] = dataset->getDatabasePose(database_image_idx_for_stitching[i]);
      cnt++;
    }
    printf("added %d frames\n", cnt);
    for (int i = 0; i < num_frames; i++) {
      image_array[cnt] = new WorkImage(dataset->getTestImage(i));
      image_poses_for_stitching[cnt] = refined_poses[i];
      cnt++;
    }
    
    WorkImage* register_map = warpImageArray(image_array, image_poses_for_stitching, 0.25);
    register_map->write("expand/refined_register.png");
    delete register_map;
  }



#endif

  delete dataset;

  return 0;
}