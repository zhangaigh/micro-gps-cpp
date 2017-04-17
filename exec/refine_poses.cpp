#include "stdio.h"
#include "refine_poses.h"
#include <vector>


#define RIGID       0
#define SIMILARITY  1
#define AFFINE      2
#define PROJECTIVE  3
int ALIGNMENT_MODE = RIGID;
// 0: rigid
// 1: similarity
// 2: affine
// 3: projective

struct alignKeyPoint {
  alignKeyPoint(float* kp_a_in, float* kp_b_in):
      kp_a(kp_a_in),
      kp_b(kp_b_in) {}

  template <typename T>
  bool operator()(const T* const pose_a, const T* const pose_b, T* residuals) const {
    T* inv_pose_b;
    T x, y;

    switch (ALIGNMENT_MODE) {
      case RIGID:
        inv_pose_b = new T[3];
        invPoseRigid(pose_b, inv_pose_b);
        applyPoseRigid (T((double)kp_a[0]), T((double)kp_a[1]), pose_a, x, y);
        applyPoseRigid (x, y, inv_pose_b, x, y);
        break;
      case SIMILARITY:
        inv_pose_b = new T[4];
        invPoseSimilarity(pose_b, inv_pose_b);
        applyPoseSimilarity (T((double)kp_a[0]), T((double)kp_a[1]), pose_a, x, y);
        applyPoseSimilarity (x, y, inv_pose_b, x, y);
        break;
      case AFFINE:
        inv_pose_b = new T[6];
        invPose6DOF(pose_b, inv_pose_b);
        applyPose6DOF (T((double)kp_a[0]), T((double)kp_a[1]), pose_a, x, y);
        applyPose6DOF (x, y, inv_pose_b, x, y);
        break;
      case PROJECTIVE:
        inv_pose_b = new T[8];
        invPose8DOF(pose_b, inv_pose_b);
        applyPose8DOF (T((double)kp_a[0]), T((double)kp_a[1]), pose_a, x, y);
        applyPose8DOF (x, y, inv_pose_b, x, y);
        break;
      default:
        x = T((double)kp_b[0]);
        y = T((double)kp_b[1]);
    }
    delete[] inv_pose_b;

    residuals[0] = x - T((double)kp_b[0]);
    residuals[1] = y - T((double)kp_b[1]);
    return true;
  }

  float* kp_a;
  float* kp_b;
};


struct alignKeyPointFixedConstraint {
  alignKeyPointFixedConstraint(float* kp_a_in, float* kp_b_in, double* pose_b_in):
      kp_a(kp_a_in),
      kp_b(kp_b_in),
      pose_b(pose_b_in) {}

  template <typename T>
  bool operator()(const T* const pose_a, T* residuals) const {
    T inv_pose_b[3];
    T pose_b_jet[3];
    T x, y;
    for (int i = 0; i < 3; i++) {
      pose_b_jet[i] = T(pose_b[i]);
    }
    invPose(pose_b_jet, inv_pose_b);
    applyPose (T((double)kp_a[0]), T((double)kp_a[1]), pose_a, x, y);
    applyPose (x, y, inv_pose_b, x, y);

    residuals[0] = x - T((double)kp_b[0]);
    residuals[1] = y - T((double)kp_b[1]);
    return true;
  }

  float* kp_a;
  float* kp_b;
  double* pose_b;
};


void addOnePair(alignmentPair* pair, std::vector<double*>& all_poses, ceres::Problem* problem) {
  for (int i = 0; i < pair->num_of_pts; i++) {
    ceres::LossFunction* loss_function = NULL;
    ceres::CostFunction* cost_function;
    switch (ALIGNMENT_MODE) {
      case RIGID:
        cost_function = new ceres::AutoDiffCostFunction<alignKeyPoint, 2, 3, 3>
                        (new alignKeyPoint(pair->kps_a+i*2, pair->kps_b+i*2));
        break;
      case SIMILARITY:
        cost_function = new ceres::AutoDiffCostFunction<alignKeyPoint, 2, 4, 4>
                        (new alignKeyPoint(pair->kps_a+i*2, pair->kps_b+i*2));
        break;
      case AFFINE:
        cost_function = new ceres::AutoDiffCostFunction<alignKeyPoint, 2, 6, 6>
                        (new alignKeyPoint(pair->kps_a+i*2, pair->kps_b+i*2));
        break;
      case PROJECTIVE:
        cost_function = new ceres::AutoDiffCostFunction<alignKeyPoint, 2, 8, 8>
                        (new alignKeyPoint(pair->kps_a+i*2, pair->kps_b+i*2));
        break;
      default:
        cost_function = NULL;
    }
    if (cost_function != NULL) {
      problem->AddResidualBlock(cost_function, loss_function, all_poses[pair->pose_a_id], all_poses[pair->pose_b_id]);
    }
  }
}


void addOnePairFixedConstraint(alignmentPair* pair,
                              std::vector<double*>& all_poses,
                              std::vector<double*>& all_fixed_poses,
                              ceres::Problem* problem) {
  for (int i = 0; i < pair->num_of_pts; i++) {
    ceres::LossFunction* loss_function = NULL;
    ceres::CostFunction* cost_function =
      new ceres::AutoDiffCostFunction<alignKeyPointFixedConstraint, 2, 3>
      (new alignKeyPointFixedConstraint(pair->kps_b+i*2, pair->kps_a+i*2, all_fixed_poses[pair->pose_a_id]));
    problem->AddResidualBlock(cost_function, loss_function, all_poses[pair->pose_b_id]);
  }
}


//TODO: read formatted pairs
void parseMatlabOutput(const char* filename,
                      std::vector<alignmentPair*>& all_pairs,
                      std::vector<double*>& all_poses) {
  FILE* fp = fopen(filename, "r");

  // read alignment mode
  fread(&ALIGNMENT_MODE, sizeof(int), 1, fp);

  printf("alignment mode: %d\n", ALIGNMENT_MODE);

  // read poses
  int n_poses;
  fread(&n_poses, sizeof(int), 1, fp);
  for (int i = 0; i < n_poses; i++) {
    double* pose;
    switch (ALIGNMENT_MODE) {
      case RIGID:
        pose = new double[3];
        fread(pose, sizeof(double), 3, fp);
        break;
      case SIMILARITY:
        pose = new double[4];
        fread(pose, sizeof(double), 4, fp);
        break;
      case AFFINE:
        pose = new double[6];
        fread(pose, sizeof(double), 6, fp);
        break;
      case PROJECTIVE:
        pose = new double[8];
        fread(pose, sizeof(double), 8, fp);
        break;
      default:
        pose = NULL;
    }
    all_poses.push_back(pose);
  }

  // read pairs
  int n_pairs;
  fread(&n_pairs, sizeof(int), 1, fp);
  for (int i = 0; i < n_pairs; i++) {
    int n_pts;
    fread(&n_pts, sizeof(int), 1, fp);
    alignmentPair* pair = new alignmentPair(n_pts);
    fread(&pair->pose_a_id, sizeof(int), 1, fp);
    fread(&pair->pose_b_id, sizeof(int), 1, fp);
    fread(pair->kps_a, sizeof(float), n_pts * 2, fp);
    fread(pair->kps_b, sizeof(float), n_pts * 2, fp);
    all_pairs.push_back(pair);
  }

  fclose(fp);

  // report
  printf("total %ld poses\n", all_poses.size());
  printf("total %ld pairs\n", all_pairs.size());

  // printf("\n3646th pair:\n");
  // printf("n_pts: %d\n", all_pairs[3646]->num_of_pts);
  // printf("pose_a_id: %d\n", all_pairs[3646]->pose_a_id);
  // printf("pose_b_id: %d\n", all_pairs[3646]->pose_b_id);
  // printf("kps_a[0]: %f\n", all_pairs[3646]->kps_a[2]);
  // printf("kps_a[1]: %f\n", all_pairs[3646]->kps_a[3]);
}


void appendAlignmentPairs(std::vector<int>& pose_id_a_array,
                          std::vector<int>& pose_id_b_array,
                          std::vector<Eigen::MatrixXf>& sift_loc_a_array,
                          std::vector<Eigen::MatrixXf>& sift_loc_b_array,
                          std::vector<alignmentPair*>& all_pairs) {
  int old_len = all_pairs.size();
  int incoming_len = pose_id_a_array.size();
  all_pairs.resize(old_len + incoming_len);

  for (int i = 0; i < incoming_len; i++) {
    int num_points = sift_loc_a_array[i].cols();
    alignmentPair* pair = new alignmentPair(num_points);  
    pair->pose_a_id = pose_id_a_array[i]; 
    pair->pose_b_id = pose_id_b_array[i]; 
    memcpy(pair->kps_a, sift_loc_a_array[i].data(), sizeof(float)*num_points*2);
    memcpy(pair->kps_b, sift_loc_b_array[i].data(), sizeof(float)*num_points*2);
    all_pairs[old_len + i] = pair;
  }
}

void convertPoses(Eigen::MatrixXf& x_y_angle_array,
                  std::vector<double*>& all_poses) {
  int num_poses = x_y_angle_array.cols();
  all_poses.resize(num_poses);
  for (int i = 0; i < num_poses; i++) {
    double* pose = new double[3];
    // pose = [angle, x, y]!!!
    pose[0] = x_y_angle_array(2, i);
    pose[1] = x_y_angle_array(0, i);
    pose[2] = x_y_angle_array(1, i);
    all_poses[i] = pose;
  }
}

void convertPosesBack(std::vector<double*>& all_poses,
                      Eigen::MatrixXf& x_y_angle_array) {
  x_y_angle_array.resize(3, all_poses.size());

  for (int i = 0; i < all_poses.size(); i++) {
    x_y_angle_array(2, i) = all_poses[i][0];
    x_y_angle_array(0, i) = all_poses[i][1];
    x_y_angle_array(1, i) = all_poses[i][2];
  }

}

void refinePoses (std::vector<alignmentPair*>& all_pairs,
                  std::vector<double*>& all_poses) {
  ceres::Problem problem;
  for (unsigned int i = 0; i < all_pairs.size(); i++) {
    addOnePair(all_pairs[i], all_poses, &problem);
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 200;
  options.minimizer_progress_to_stdout = true;
  options.num_threads = 32;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  //ceres::SPARSE_SCHUR;  //ceres::DENSE_SCHUR;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
}

void refinePosesWithHardConstraints (std::vector<alignmentPair*>& all_pairs,
                                    std::vector<alignmentPair*>& all_fixed_constraint_pairs,
                                    std::vector<double*>& all_poses,
                                    std::vector<double*>& all_fixed_poses) {
  ceres::Problem problem;

  printf("enforcing pairwise constraint\n");
  for (unsigned int i = 0; i < all_pairs.size(); i++) {
    addOnePair(all_pairs[i], all_poses, &problem);
  }

  printf("enforcing fixed pose constraint\n");
  for (unsigned int i = 0; i < all_fixed_constraint_pairs.size(); i++) {
    addOnePairFixedConstraint(all_fixed_constraint_pairs[i], all_poses, all_fixed_poses, &problem);
  }

  // printf("enforcing initial pose constraint\n");
  // initialPoseConstraint(all_poses, &problem);
  printf("begin optimization\n");

  ceres::Solver::Options options;
  options.max_num_iterations = 200;
  options.minimizer_progress_to_stdout = true;
  options.num_threads = 32;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  //ceres::SPARSE_SCHUR;  //ceres::DENSE_SCHUR;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  printf("done optimization\n");
}



void outputPoses(const char* output_path, std::vector<double*>& all_poses) {
  FILE* fp = fopen(output_path, "w");
  int n_poses = all_poses.size();
  fwrite (&n_poses, sizeof(int), 1, fp);
  for (unsigned int i = 0; i < all_poses.size(); i++) {
    switch (ALIGNMENT_MODE) {
      case RIGID:
        fwrite(all_poses[i], sizeof(double), 3, fp);
        break;
      case SIMILARITY:
        fwrite(all_poses[i], sizeof(double), 4, fp);
        break;
      case AFFINE:
        fwrite(all_poses[i], sizeof(double), 6, fp);
        break;
      case PROJECTIVE:
        fwrite(all_poses[i], sizeof(double), 8, fp);
        break;
      default:
        break;
    }
  }
  fclose(fp);
}

// int main(int argc, char const *argv[]) {
//   std::vector<alignmentPair*> all_pairs;
//   std::vector<double*> all_poses;

//   parseMatlabOutput(argv[1], all_pairs, all_poses);
//   refinePoses(all_pairs, all_poses);
//   outputPoses(argv[2], all_poses);

//   printf("sizeof autodiff func: %d\n", sizeof(alignKeyPoint));
// }
