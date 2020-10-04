#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"

class UKF {
 public:
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  // Methods

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);

  // Calculate NIS Value

  inline const float Calculate_NIS_Value(const Eigen::VectorXd z_prediction, 
                                  const Eigen::VectorXd Z_measurement, 
                                  const Eigen::MatrixXd S); // Measurement covariance

  // Attributes

  int n_x_; // State dimension
  int n_aug_; // Augmented state dimension
  double lambda_; // Sigma point spreading parameter

  bool use_laser_; // If this is false, laser measurements will be ignored (except for init)
  bool use_radar_; // If this is false, radar measurements will be ignored (except for init)
 
  bool is_initialized_; // Initially set to false, set to true in first call of ProcessMeasurement

  Eigen::VectorXd x_; // State vector (5 x 1): [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::MatrixXd P_; // State covariance matrix (5 x 5)
  Eigen::MatrixXd Xsig_pred_; // Predicted sigma points matrix (5 x 15)

  long previous_time_us_; // Time when the state is true, in us

  // Process noise

  double std_a_; // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_yawdd_; // Process noise standard deviation yaw acceleration in rad/s^2

  // Measurement noise

  double std_laspx_; // Laser measurement noise standard deviation position1 in m
  double std_laspy_; // Laser measurement noise standard deviation position2 in m

  double std_radr_; // Radar measurement noise standard deviation radius in m
  double std_radphi_; // Radar measurement noise standard deviation angle in rad
  double std_radrd_ ; // Radar measurement noise standard deviation radius change in m/s

  Eigen::VectorXd weights_; // Weights of sigma points
};

#endif  // UKF_H