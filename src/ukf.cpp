#include <iostream>
#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Calculate Normalized Innovation Squared (NIS)

inline const float UKF::Calculate_NIS_Value(const Eigen::VectorXd z_prediction, 
                                  const Eigen::VectorXd Z_measurement, 
                                  const Eigen::MatrixXd S) 
{
  VectorXd diff = Z_measurement - z_prediction;
  float NIS = diff.transpose() * S.inverse() * diff;
  return NIS;
}

// Initialize Unscented Kalman Filter

UKF::UKF() {
  n_x_ = 5; // State dimension
  n_aug_ = 7; // Augmented state dimension
  lambda_ = 3 - n_aug_; // Sigma point spreading parameter

  use_laser_ = true; 
  use_radar_ = true; 

  is_initialized_ = false; 

  x_ = VectorXd(n_x_); // Initial state vector
  P_ = MatrixXd(n_x_, n_x_); // Initial covariance matrix

  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_ + 1);  // Predicted Sigma Points (in the 5-dimensional space) considering noise
  weights_ = VectorXd(2*n_aug_ + 1); // 2*7+1 = 15. Considering 

  std_a_ = 3.0; // 30; // Process noise standard deviation longitudinal acceleration in m/s^2
  std_yawdd_ = 1.0; // 30; // Process noise standard deviation yaw acceleration in rad/s^2

  // DO NOT MODIFY measurement noise values below.
  // These are provided by the sensor manufacturer.

  std_laspx_ = 0.15; // Laser measurement noise standard deviation position1 in m (noise_px)
  std_laspy_ = 0.15; // Laser measurement noise standard deviation position2 in m (noise_py)
  
  std_radr_ = 0.3; // Radar measurement noise standard deviation radius in m (noise_ro)
  std_radphi_ = 0.03; // Radar measurement noise standard deviation angle in rad (noise_phi)
  std_radrd_ = 0.3; // Radar measurement noise standard deviation radius change in m/s (noise_ro_dot)
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  // 0. Initialize (Based on the measurements of the first sensor that arrives)

  if (!is_initialized_) // Run this code if it is the first measurement (either LiDAR or RADAR)
  // in order to initialize the values of x_ and P_
  {
    // Initialize state mean and covariance matrix if LiDAR is the first sensor to arrive (or
    // if we are only using LiDAR)

    if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
    {
      std::cout << "LiDAR init" << std::endl;

      double variance_xx = pow(std_laspx_,2);
      double variance_yy = pow(std_laspy_,2);

      P_ << variance_xx,0,0,0,0,
            0,variance_yy,0,0,0,
            0,0,5,0,0,
            0,0,0,1,0,
            0,0,0,0,1;
      
      x_(0) = meas_package.raw_measurements_(0); // px
      x_(1) = meas_package.raw_measurements_(1); // py        
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
    // Initialize state mean and covariance matrix if RADAR is the first sensor to arrive (or
    // if we are only using RADAR)
    {
      std::cout << "RADAR init" << std::endl;

      P_ << 1,0,0,0,0,
            0,1,0,0,0,
            0,0,1,0,0,
            0,0,0,1,0,
            0,0,0,0,1;

      double ro = meas_package.raw_measurements_(0);
      double bearing = meas_package.raw_measurements_(1);
      double ro_dot = meas_package.raw_measurements_(2);

      x_(0) = ro * cos(bearing); // ro · cos(phi)
      x_(1) = ro * sin(bearing);; // ro · sin(phi)

      double vel_cartesian_plane = sqrt(pow(ro_dot*cos(bearing),2) + pow(ro_dot*sin(bearing),2));
      x_(2) = vel_cartesian_plane;
    }

    previous_time_us_ = meas_package.timestamp_;
    std::cout << "Previous time in s: " << previous_time_us_ << std::endl; // / 1000000.0 << std::endl;
    is_initialized_ = true;
    return; // End initialization
  }

  // ###### UKF cycle ######

  // 1. Predict

  //std::cout << "\nCurrent time in s: " << current_timestamp / 1000000.0 << std::endl;
  const double delta_t = (meas_package.timestamp_ - previous_time_us_) / 1000000.0; // From us to s
  //std::cout << "\nDelta_t: " << delta_t << std::endl;
  previous_time_us_ = meas_package.timestamp_;
  
  Prediction(delta_t);

  // 2. Update

  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
  {
    std::cout << "LiDAR" << std::endl;
    UpdateLidar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
  {
    std::cout << "RADAR" << std::endl;
    UpdateRadar(meas_package);
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  //std::cout << "x (k | k): " << x_ << std::endl;
  //std::cout << "P (k | k): " << P_ << std::endl << std::endl;
  // 0. Initialize variables

  VectorXd x_aug = VectorXd(n_aug_); // Augmented state mean vector
  MatrixXd P_aug = MatrixXd(n_aug_,n_aug_); // Augmented state covariance matrix

  // Augmented state mean vector

  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0; // Mean for the normal distribution of longitudinal acceleration noise
  x_aug(n_x_ + 1) = 0; // Mean for the normal distribution of longitudinal acceleration noise

  // Augmented state covariance matrix 

  P_aug.fill(0); // Initialize with 0s
  P_aug.topLeftCorner(n_x_,n_x_) = P_; // P
  P_aug(n_x_,n_x_) = pow(std_a_,2); // Q
  P_aug(n_x_ + 1,n_x_ + 1) = pow(std_yawdd_,2); // Q

  // Create square root matrix

  MatrixXd L = P_aug.llt().matrixL();

  // 1. Get Sigma Points considering noise (Augmented Sigma Points)

  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1); // Sigma Points matrix

  Xsig_aug.fill(0);
  Xsig_aug.col(0) = x_aug; // Mean of the current state (including noise, so 7 rows)

  for (size_t i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda_ + n_aug_)*L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_)*L.col(i);
  }

  // 2. Predict (Augmented) Sigma Points

  for (size_t i = 0; i < 2*n_aug_ + 1; i++)
  {
      double px = Xsig_aug(0,i);
      double py = Xsig_aug(1,i);
      double v = Xsig_aug(2,i);
      double yaw = Xsig_aug(3,i);
      double yawd = Xsig_aug(4,i); // Yaw change rate
      double nu_a = Xsig_aug(5,i); // Longitudinal acceleration noise
      double nu_yawdd = Xsig_aug(6,i); // Yaw acceleration noise

      // Predicted state values

      double px_p, py_p, v_p, yaw_p, yawd_p;

      // Avoid division by zero

      if (fabs(yawd) < 0.0001) // Moving on straight line 
      {
          px_p = px + v*cos(yaw)*delta_t;
          py_p = py + v*sin(yaw)*delta_t;
      }
      else // Moving on curve line
      {
          px_p = px + (v/yawd)*(sin(yaw + yawd*delta_t) - sin(yaw)); 
          py_p = py + (v/yawd)*(-cos(yaw + yawd*delta_t) + cos(yaw));  
      }

      v_p = v; // Assuming CTRV model
      yaw_p = yaw + yawd*delta_t; 
      yawd_p = yawd; // Assuming CTRV model

      // Add noise

      px_p += 1/2*pow(delta_t,2)*cos(yaw)*nu_a;
      py_p += 1/2*pow(delta_t,2)*sin(yaw)*nu_a;
      v_p += nu_a*delta_t;
      yaw_p += 1/2*nu_yawdd*pow(delta_t,2);
      yawd_p += nu_yawdd*delta_t;

      // Write predicted sigma points

      Xsig_pred_(0,i) = px_p;
      Xsig_pred_(1,i) = py_p;
      Xsig_pred_(2,i) = v_p;
      Xsig_pred_(3,i) = yaw_p;
      Xsig_pred_(4,i) = yawd_p;
  }

  // 3. Predict Mean and Covariance

  // Set weights

  for (size_t i = 0; i < 2*n_aug_ + 1; i++)
  {
      if (i==0)
      {
          weights_(i) = lambda_ / (lambda_ + n_aug_);
      }
      else
      {
          weights_(i) = 1 / (2*(lambda_+n_aug_));
      }
  }

  // Predict state mean

  x_.fill(0.0); // Initialize the vector with zeros

  for (size_t i = 0; i < 2*n_aug_ + 1; i++)
  {
      x_ += weights_(i)*Xsig_pred_.col(i);
  }

  // Predict state covariance matrix

  P_.fill(0.0); // Initialize the matrix with zeros

  for (size_t i = 0; i < 2*n_aug_+1; i++)
  {
      // State difference (with respect to the mean)

      VectorXd x_diff = Xsig_pred_.col(i) - x_;

      // Angle normalization (Yaw angle must be between pi and -pi)

      while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
      while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

      P_ += weights_(i) * x_diff * x_diff.transpose();
  }

  //std::cout << "x (k+1 | k): " << x_ << std::endl;
  //std::cout << "P (k+1 | k): " << P_ << std::endl;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  // In order to update x_ and P_ it is much easier since LiDAR provides linear measurements
  // so standard update stage of Kalman Filter can be applied

  int n_z = 2; // LiDAR measures px and py

  // It can be considered current x_ (which is x_ (k+1|k) after prediction stage) and then 
  // just map from x_ to consider only LiDAR measurements, and then take current measurements (k+1)
  // to get the error y

  // 1. Compute error between current measurement and predicted state

  MatrixXd H = MatrixXd(n_z,n_x_);
  H.fill(0);
  H(0,0) = 1; // To map px
  H(1,1) = 1; // To map py

  VectorXd y = VectorXd(n_z); // Error
  VectorXd z = VectorXd(n_z); // Current measurement
  VectorXd H_times_x_ = VectorXd(n_z);

  z = meas_package.raw_measurements_;
  H_times_x_ = H * x_;
  y = z - H_times_x_;

  // 2. Compute remaining equations to get the Kalman gain

  MatrixXd I = MatrixXd::Identity(n_x_,n_x_);
  MatrixXd Ht = H.transpose();

  MatrixXd R = MatrixXd(n_z,n_z); // LiDAR measurement covariance noise
  R << pow(std_laspx_,2),0,
       0, pow(std_laspy_,2);

  MatrixXd S = MatrixXd(n_z,n_z);
  S = H * P_ * Ht + R; // LiDAR measurement covariance
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  // 3. Get new state

  x_ = x_ + K*y;
  P_ = (I - K*H) * P_;

  std::cout << "LiDAR NIS: " << Calculate_NIS_Value(H_times_x_, z, S) << std::endl << std::endl;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  int n_z = 3; // Measurement dimension (RADAR can measure r, phi and r_dot (== radial velocity))

  // 1. Transform Sigma Points in measurement space for predicted state x_ (x(k+1 | k))

  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_ + 1); // Matrix for sigma points in measurement space
  VectorXd z_pred = VectorXd(n_z); // Mean predicted measurement
  MatrixXd S = MatrixXd(n_z,n_z); // Measurement covariance matrix

  // Transform Sigma Points into measurement space

  for (size_t i = 0; i < 2*n_aug_ + 1; i++)
  {
    double px = Xsig_pred_(0,i);
    double py = Xsig_pred_(1,i);
    double norm = sqrt(pow(px,2)+pow(py,2));
    double v = Xsig_pred_(2,i);
    double shi = Xsig_pred_(3,i);

    // RADAR measurement space (ro, phi and ro_dot)
    Zsig(0,i) = norm;
    Zsig(1,i) = atan2(py,px);
    Zsig(2,i) = (px*cos(shi)*v + py*sin(shi)*v) / norm;
  }

  // Calculate mean predicted measurement

  z_pred.fill(0.0);

  for (size_t i = 0; i < 2*n_aug_ + 1; i++)
  {
      z_pred = z_pred + weights_(i)*Zsig.col(i);
  }

  // Calculate measurement covariance matrix

  S.fill(0.0); // Initialize the matrix with zeros

  for (size_t i = 0; i < 2*n_aug_ + 1; i++)
  {
      // State difference (with respect to the mean)

      VectorXd z_diff = Zsig.col(i) - z_pred;

      // Angle normalization (Phi angle must be between pi and -pi)
      // Note that z has 3 rows (r, phi and r_dot, so phi is the 1 index)
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

      S += weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix 

  MatrixXd R = MatrixXd(n_z,n_z); // Since S is 3x3, since z has 3 rows!!!

  R <<  pow(std_radr_,2), 0, 0, // Considering the expected value and uncorrelated noises,
  // it is represented by the variances of the corresponding noises, assuming normal distributions
        0, pow(std_radphi_,2), 0,
        0, 0, pow(std_radrd_,2);

  S += R;

  // 2. Considering predicted sigma points (so, associated mean and covariance), take current
  // RADAR measurements and compute the final state of x_ and P_ after this cycle

  VectorXd z = meas_package.raw_measurements_;

  // Create Matrix for Cross-Correlation Tc

  MatrixXd Tc = MatrixXd(n_x_, n_z); 

  // Calculate cross-correlation matrix

  Tc.fill(0.0);

  for (size_t i = 0; i < 2*n_aug_ + 1; i++)
  {
    // 1. Xk+1 - x_mean

    // Residual

    VectorXd x_diff = Xsig_pred_.col(i) - x_; 

    // Angle normalization

    while (x_diff(3) > M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.*M_PI;

    // 2. Zk+1 - z_mean

    // Residual
    
    VectorXd z_diff = Zsig.col(i) - z_pred; 

    // Angle normalization

    while (z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Calculate Kalman gain K (in this it is a matrix)

  MatrixXd K = Tc * S.inverse();

  // Update state

  // Residual

  VectorXd z_diff = z - z_pred;

  // Angle normalization

  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  x_ = x_ + K * z_diff; // Note that z are the new measurements and z_pred the result
  // of proyecting the predicted state onto the measurement space
  P_ = P_ - K * S * K.transpose();

  std::cout << "RADAR NIS: " << Calculate_NIS_Value(z_pred, z, S) << std::endl << std::endl;
}


