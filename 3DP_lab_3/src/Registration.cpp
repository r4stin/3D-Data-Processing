#include "Registration.h"


struct PointDistance
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // This class should include an auto-differentiable cost function. 
  // To rotate a point given an axis-angle rotation, use
  // the Ceres function:
  // AngleAxisRotatePoint(...) (see ceres/rotation.h)
  // Similarly to the Bundle Adjustment case initialize the struct variables with the source and  the target point.
  // You have to optimize only the 6-dimensional array (rx, ry, rz, tx ,ty, tz).
  // WARNING: When dealing with the AutoDiffCostFunction template parameters,
  // pay attention to the order of the template parameters
  ////////////////////////////////////////////////////////////////////////////////////////////////////
    const Eigen::Vector3d target_point;
    const Eigen::Vector3d source_point;

    PointDistance(Eigen::Vector3d target, Eigen::Vector3d source)
            : target_point(target), source_point(source) {}

    template<typename T>
    bool operator()(const T *const rotation, const T *const translation, T *residual) const {

        Eigen::Matrix<T, 3, 1> rotation_vector(rotation[0], rotation[1], rotation[2]);
        Eigen::Matrix<T, 3, 1> translation_vector(translation[0], translation[1], translation[2]);

        // Rotate and translate points
        Eigen::Matrix<T, 3, 1> source_point_t(source_point.cast<T>());
        Eigen::Matrix<T, 3, 1> target_point_t(target_point.cast<T>());
        Eigen::Matrix<T, 3, 1> transformed_point;

        ceres::AngleAxisRotatePoint(rotation_vector.data(), source_point_t.data(), transformed_point.data());

        // Apply the translation to the rotated source point.
        transformed_point += translation_vector;

        // Compute the residuals as the difference between the transformed source point and the target point.
        residual[0] = transformed_point[0] - target_point_t[0];
        residual[1] = transformed_point[1] - target_point_t[1];
        residual[2] = transformed_point[2] - target_point_t[2];

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d &target, const Eigen::Vector3d &source) {
        return (new ceres::AutoDiffCostFunction<PointDistance, 3, 3, 3>(
                new PointDistance(target, source)));
    }
};





Registration::Registration(std::string cloud_source_filename, std::string cloud_target_filename)
{
  open3d::io::ReadPointCloud(cloud_source_filename, source_ );
  open3d::io::ReadPointCloud(cloud_target_filename, target_ );
  Eigen::Vector3d gray_color;
  source_for_icp_ = source_;
}


Registration::Registration(open3d::geometry::PointCloud cloud_source, open3d::geometry::PointCloud cloud_target)
{
  source_ = cloud_source;
  target_ = cloud_target;
  source_for_icp_ = source_;
}


void Registration::draw_registration_result()
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  //different color
  Eigen::Vector3d color_s;
  Eigen::Vector3d color_t;
  color_s<<1, 0.706, 0;
  color_t<<0, 0.651, 0.929;

  target_clone.PaintUniformColor(color_t);
  source_clone.PaintUniformColor(color_s);
  source_clone.Transform(transformation_);

  auto src_pointer =  std::make_shared<open3d::geometry::PointCloud>(source_clone);
  auto target_pointer =  std::make_shared<open3d::geometry::PointCloud>(target_clone);
  open3d::visualization::DrawGeometries({src_pointer, target_pointer});
  return;
}



void Registration::execute_icp_registration(double threshold, int max_iteration, double relative_rmse, std::string mode)
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //ICP main loop
  //Check convergence criteria and the current iteration.
  //If mode=="svd" use get_svd_icp_transformation if mode=="lm" use get_lm_icp_transformation.
  //Remember to update transformation_ class variable, you can use source_for_icp_ to store transformed 3d points.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    double previous_rmse = std::numeric_limits<double>::max();
    double previous_error = std::numeric_limits<double>::max();
    int iteration = 0;

    while (iteration < max_iteration) {
        auto [source_indices, target_indices, rmse] = find_closest_point(threshold);

        // Check for convergence
        if (rmse > previous_rmse && (previous_error - rmse) < relative_rmse) {
            break;
        }
        previous_error = rmse;

        if (mode == "svd") {
            transformation_ = get_svd_icp_transformation(source_indices, target_indices);
        } else if (mode == "lm") {
            transformation_ = get_lm_icp_registration(source_indices, target_indices);
        } else {
            std::cout << "Unknown mode." << std::endl;
            return;
        }

        source_for_icp_.Transform(transformation_);

        previous_rmse = rmse;
        iteration++;

    }
}


std::tuple<std::vector<size_t>, std::vector<size_t>, double> Registration::find_closest_point(double threshold)
{ ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find source and target indices: for each source point find the closest one in the target and discard if their 
  //distance is bigger than threshold
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  std::vector<size_t> target_indices;
  std::vector<size_t> source_indices;
  Eigen::Vector3d source_point;
  double rmse;

// Initialize KDTree for the target point cloud
    open3d::geometry::KDTreeFlann target_kd_tree(target_);

    std::vector<int> idx(1);  // Index of the nearest neighbor in the target cloud
    std::vector<double> dist2(1);  // Squared distance to the nearest neighbor

    for (size_t i = 0; i < source_.points_.size(); ++i)
    {
        source_point = source_.points_[i];
        target_kd_tree.SearchKNN(source_point, 1, idx, dist2);

        // Check if the distance is within the threshold
        if (dist2[0] <= threshold * threshold)
        {
            source_indices.push_back(i);
            target_indices.push_back(idx[0]);
            rmse += dist2[0];  // Sum of squared distances
        }
    }

    // Calculate RMSE
    if (!source_indices.empty())
    {
        rmse = std::sqrt(rmse / source_indices.size());
    }


  return {source_indices, target_indices, rmse};

}

Eigen::Matrix4d Registration::get_svd_icp_transformation(std::vector<size_t> source_indices, std::vector<size_t> target_indices){
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find point clouds centroids and subtract them. 
  //Use SVD (Eigen::JacobiSVD<Eigen::MatrixXd>) to find best rotation and translation matrix.
  //Use source_indices and target_indices to extract point to compute the 3x3 matrix to be decomposed.
  //Remember to manage the special reflection case.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);

      // Compute the centroid of the source and target point clouds
      Eigen::Vector3d source_centroid = Eigen::Vector3d::Zero();
      Eigen::Vector3d target_centroid = Eigen::Vector3d::Zero();

      for (size_t j = 0; j < source_indices.size(); ++j)
      {
          source_centroid += source_.points_[source_indices[j]];
          target_centroid += target_.points_[target_indices[j]];

      }

      source_centroid /= source_indices.size();
      target_centroid /= source_indices.size();

      // Compute the 3x3 matrix to be decomposed
      Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 3);

      for (size_t j = 0; j < source_indices.size(); ++j)
      {
          Eigen::Vector3d source_point = source_.points_[source_indices[j]] - source_centroid;
          Eigen::Vector3d target_point = target_.points_[target_indices[j]] - target_centroid;

          A += source_point * target_point.transpose();
      }

      // Perform SVD
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

      // Compute the rotation matrix
      Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();

      // Manage the special reflection case
      if (R.determinant() < 0)
      {
          Eigen::Matrix3d U = svd.matrixU();
          U.col(2) *= -1;
          R = U * svd.matrixV().transpose();
      }

      // Compute the translation vector
      Eigen::Vector3d t = target_centroid - R * source_centroid;

      // Update the transformation matrix
      transformation.block<3, 3>(0, 0) = R;
      transformation.block<3, 1>(0, 3) = t;




  return transformation;
}

Eigen::Matrix4d Registration::get_lm_icp_registration(std::vector<size_t> source_indices, std::vector<size_t> target_indices)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Use LM (Ceres) to find best rotation and translation matrix. 
  //Remember to convert the euler angles in a rotation matrix, store it coupled with the final translation on:
  //Eigen::Matrix4d transformation.
  //The first three elements of std::vector<double> transformation_arr represent the euler angles, the last ones
  //the translation.
  //use source_indices and target_indices to extract point to compute the matrix to be decomposed.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 4;
  options.max_num_iterations = 100;
  ceres::Problem problem;
  ceres::Solver::Summary summary;

  std::vector<double> transformation_arr(6, 0.0);
  int num_points = source_indices.size();

    // For each point....
    for (size_t i = 0; i < num_points; ++i) {
        ceres::CostFunction *cost_function = PointDistance::Create(target_.points_[target_indices[i]],
                                                                   source_.points_[source_indices[i]]);
        problem.AddResidualBlock(cost_function, nullptr, transformation_arr.data(), transformation_arr.data() + 3);
    }

    ceres::Solve(options, &problem, &summary);

    Eigen::Matrix3d rotation_matrix;
    ceres::AngleAxisToRotationMatrix(transformation_arr.data(), rotation_matrix.data());
    Eigen::Vector3d translation(transformation_arr[3], transformation_arr[4], transformation_arr[5]);
    transformation.block<3, 3>(0, 0) = rotation_matrix;
    transformation.block<3, 1>(0, 3) = translation;

  return transformation;
}


void Registration::set_transformation(Eigen::Matrix4d init_transformation)
{
  transformation_=init_transformation;
}


Eigen::Matrix4d  Registration::get_transformation()
{
  return transformation_;
}

double Registration::compute_rmse()
{
  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  open3d::geometry::PointCloud source_clone = source_;
  source_clone.Transform(transformation_);
  int num_source_points  = source_clone.points_.size();
  Eigen::Vector3d source_point;
  std::vector<int> idx(1);
  std::vector<double> dist2(1);
  double mse;
  for(size_t i=0; i < num_source_points; ++i) {
    source_point = source_clone.points_[i];
    target_kd_tree.SearchKNN(source_point, 1, idx, dist2);
    mse = mse * i/(i+1) + dist2[0]/(i+1);
  }
  return sqrt(mse);
}

void Registration::write_tranformation_matrix(std::string filename)
{
  std::ofstream outfile (filename);
  if (outfile.is_open())
  {
    outfile << transformation_;
    outfile.close();
  }
}

void Registration::save_merged_cloud(std::string filename)
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  source_clone.Transform(transformation_);
  open3d::geometry::PointCloud merged = target_clone+source_clone;
  open3d::io::WritePointCloud(filename, merged );
//  open3d::io::WritePointCloud("source.ply", source_clone );
//  open3d::io::WritePointCloud("target.ply", target_clone );
}


