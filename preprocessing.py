"""
Preprocessing tools
@author: victoria
"""

import pandas as pd

train_flight_id0 = "carbonZ_2018-09-11-15-05-11_2_no_failure"
train_flight_id1 = "carbonZ_2018-07-30-16-39-00_3_no_failure"
train_flight_id2 = "carbonZ_2018-10-05-14-34-20_1_no_failure"
train_flight_id3 = "carbonZ_2018-07-18-16-37-39_1_no_failure"
train_flight_id4 = "carbonZ_2018-09-11-14-16-55_no_failure"
train_flight_id5 = "carbonZ_2018-09-11-14-41-38_no_failure"
train_flight_id6 = "carbonZ_2018-10-05-14-37-22_1_no_failure"
train_flight_id7 = "carbonZ_2018-10-05-15-52-12_1_no_failure"
train_flight_id8 = "carbonZ_2018-10-05-15-52-12_2_no_failure"
train_flight_id9 = "carbonZ_2018-10-18-11-08-24_no_failure"

train_flight_idp10 = ["carbonZ_2018-10-18-11-04-35_engine_failure_with_emr_traj", 50]
train_flight_idp11 = ["carbonZ_2018-10-18-11-04-08_1_engine_failure_with_emr_traj", 50]
train_flight_idp12 = ["carbonZ_2018-10-18-11-04-00_engine_failure_with_emr_traj", 50]
train_flight_idp13 = ["carbonZ_2018-10-18-11-03-57_engine_failure_with_emr_traj", 50]
train_flight_idp14 = ["carbonZ_2018-10-05-16-04-46_engine_failure_with_emr_traj", 50]
train_flight_idp15 = ["carbonZ_2018-07-18-16-37-39_2_engine_failure_with_emr_traj", 50]
train_flight_idp16 = ["carbonZ_2018-07-30-16-29-45_engine_failure_with_emr_traj", 50]
train_flight_idp17 = ["carbonZ_2018-07-30-16-39-00_1_engine_failure", 50]
train_flight_idp18 = ["carbonZ_2018-07-30-16-39-00_2_engine_failure", 50]
train_flight_idp19 = ["carbonZ_2018-07-30-17-10-45_engine_failure_with_emr_traj", 50]
train_flight_idp20 = ["carbonZ_2018-07-30-17-20-01_engine_failure_with_emr_traj", 50]
train_flight_idp21 = ["carbonZ_2018-07-30-17-36-35_engine_failure_with_emr_traj", 50]
train_flight_idp22 = ["carbonZ_2018-07-30-17-46-31_engine_failure_with_emr_traj", 50]
train_flight_idp23 = ["carbonZ_2018-09-11-11-56-30_engine_failure", 50]

partial_flights = [train_flight_idp10, train_flight_idp11, train_flight_idp12, train_flight_idp13, train_flight_idp14,
                   train_flight_idp15, train_flight_idp16, train_flight_idp17, train_flight_idp18, train_flight_idp19,
                   train_flight_idp20, train_flight_idp21, train_flight_idp22, train_flight_idp23]


def Process_Misc(flight_id):
    Temp1 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-rc-out.csv")
    Temp3 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-setpoint_raw-target_global.csv")

    Temp6 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-vfr_hud.csv")
    Temp7 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-wind_estimation.csv")

    Temp1.drop('field.header.frame_id', inplace=True, axis=1)
    Temp1.drop('field.header.stamp', inplace=True, axis=1)
    Temp1.drop('field.header.seq', inplace=True, axis=1)

    Temp3.drop('field.header.frame_id', inplace=True, axis=1)
    Temp3.drop('field.header.stamp', inplace=True, axis=1)
    Temp3.drop('field.header.seq', inplace=True, axis=1)

    Temp6.drop('field.header.frame_id', inplace=True, axis=1)
    Temp6.drop('field.header.stamp', inplace=True, axis=1)
    Temp6.drop('field.header.seq', inplace=True, axis=1)

    Temp7.drop('field.header.frame_id', inplace=True, axis=1)
    Temp7.drop('field.header.stamp', inplace=True, axis=1)
    Temp7.drop('field.header.seq', inplace=True, axis=1)
    Temp7.drop('field.twist.linear.z', inplace=True, axis=1)
    Temp7.drop('field.twist.linear.x', inplace=True, axis=1)
    Temp7.drop('field.twist.angular.x', inplace=True, axis=1)
    Temp7.drop('field.twist.angular.y', inplace=True, axis=1)
    Temp7.drop('field.twist.angular.z', inplace=True, axis=1)

    merged = pd.merge_asof(Temp1, Temp3, on='%time', tolerance=100000000)
    merged = pd.merge_asof(merged, Temp6, on='%time', tolerance=100000000)
    merged = pd.merge_asof(merged, Temp7, on='%time', tolerance=100000000)

    merged = merged.dropna()

    return merged


def Process_IMU(flight_id):
    Temp1 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-imu-data_raw.csv")
    Temp2 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-imu-mag.csv")
    Temp3 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-imu-temperature.csv")
    Temp4 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-imu-atm_pressure.csv")

    Temp1.drop('field.header.frame_id', inplace=True, axis=1)
    Temp1.drop('field.header.stamp', inplace=True, axis=1)
    Temp1.drop('field.header.seq', inplace=True, axis=1)
    Temp1.drop('field.orientation.x', inplace=True, axis=1)
    Temp1.drop('field.orientation.y', inplace=True, axis=1)
    Temp1.drop('field.orientation.z', inplace=True, axis=1)
    Temp1.drop('field.orientation.w', inplace=True, axis=1)
    Temp1.drop('field.orientation_covariance0', inplace=True, axis=1)
    Temp1.drop('field.orientation_covariance1', inplace=True, axis=1)
    Temp1.drop('field.orientation_covariance2', inplace=True, axis=1)
    Temp1.drop('field.orientation_covariance3', inplace=True, axis=1)
    Temp1.drop('field.orientation_covariance4', inplace=True, axis=1)
    Temp1.drop('field.orientation_covariance5', inplace=True, axis=1)
    Temp1.drop('field.orientation_covariance6', inplace=True, axis=1)
    Temp1.drop('field.orientation_covariance7', inplace=True, axis=1)
    Temp1.drop('field.orientation_covariance8', inplace=True, axis=1)
    Temp1.drop('field.angular_velocity_covariance0', inplace=True, axis=1)
    Temp1.drop('field.angular_velocity_covariance1', inplace=True, axis=1)
    Temp1.drop('field.angular_velocity_covariance2', inplace=True, axis=1)
    Temp1.drop('field.angular_velocity_covariance4', inplace=True, axis=1)
    Temp1.drop('field.angular_velocity_covariance3', inplace=True, axis=1)
    Temp1.drop('field.angular_velocity_covariance5', inplace=True, axis=1)
    Temp1.drop('field.angular_velocity_covariance6', inplace=True, axis=1)
    Temp1.drop('field.angular_velocity_covariance7', inplace=True, axis=1)
    Temp1.drop('field.angular_velocity_covariance8', inplace=True, axis=1)

    Temp1.drop('field.linear_acceleration_covariance0', inplace=True, axis=1)

    Temp1.drop('field.linear_acceleration_covariance1', inplace=True, axis=1)
    Temp1.drop('field.linear_acceleration_covariance2', inplace=True, axis=1)
    Temp1.drop('field.linear_acceleration_covariance3', inplace=True, axis=1)
    Temp1.drop('field.linear_acceleration_covariance4', inplace=True, axis=1)
    Temp1.drop('field.linear_acceleration_covariance5', inplace=True, axis=1)
    Temp1.drop('field.linear_acceleration_covariance6', inplace=True, axis=1)
    Temp1.drop('field.linear_acceleration_covariance7', inplace=True, axis=1)
    Temp1.drop('field.linear_acceleration_covariance8', inplace=True, axis=1)

    Temp2.drop('field.header.frame_id', inplace=True, axis=1)
    Temp2.drop('field.header.stamp', inplace=True, axis=1)
    Temp2.drop('field.header.seq', inplace=True, axis=1)
    Temp2.drop('field.magnetic_field_covariance0', inplace=True, axis=1)
    Temp2.drop('field.magnetic_field_covariance1', inplace=True, axis=1)
    Temp2.drop('field.magnetic_field_covariance2', inplace=True, axis=1)
    Temp2.drop('field.magnetic_field_covariance3', inplace=True, axis=1)
    Temp2.drop('field.magnetic_field_covariance4', inplace=True, axis=1)
    Temp2.drop('field.magnetic_field_covariance5', inplace=True, axis=1)
    Temp2.drop('field.magnetic_field_covariance6', inplace=True, axis=1)
    Temp2.drop('field.magnetic_field_covariance7', inplace=True, axis=1)
    Temp2.drop('field.magnetic_field_covariance8', inplace=True, axis=1)

    Temp3.drop('field.header.frame_id', inplace=True, axis=1)
    Temp3.drop('field.header.stamp', inplace=True, axis=1)
    Temp3.drop('field.header.seq', inplace=True, axis=1)
    Temp3.drop('field.variance', inplace=True, axis=1)

    Temp4.drop('field.header.frame_id', inplace=True, axis=1)
    Temp4.drop('field.header.stamp', inplace=True, axis=1)
    Temp4.drop('field.header.seq', inplace=True, axis=1)
    Temp4.drop('field.variance', inplace=True, axis=1)

    merged = pd.merge_asof(Temp1, Temp2, on='%time', tolerance=100000000)
    merged = pd.merge_asof(merged, Temp3, on='%time', tolerance=100000000)
    merged = pd.merge_asof(merged, Temp4, on='%time', tolerance=100000000)

    merged = merged.dropna()

    return merged


def Process_NAV_Info(flight_id):
    Temp1 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-nav_info-airspeed.csv")
    Temp2 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-nav_info-errors.csv")
    Temp3 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-nav_info-pitch.csv")
    Temp4 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-nav_info-roll.csv")
    Temp5 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-nav_info-velocity.csv")
    Temp6 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-nav_info-yaw.csv")

    Temp1.drop('field.header.frame_id', inplace=True, axis=1)
    Temp1.drop('field.header.seq', inplace=True, axis=1)
    Temp1.drop('field.header.stamp', inplace=True, axis=1)
    Temp1.drop('field.measured', inplace=True, axis=1)
    Temp1.drop('field.commanded', inplace=True, axis=1)

    Temp2.drop('field.header.stamp', inplace=True, axis=1)
    Temp2.drop('field.header.frame_id', inplace=True, axis=1)
    Temp2.drop('field.header.seq', inplace=True, axis=1)

    Temp3.drop('field.header.stamp', inplace=True, axis=1)
    Temp3.drop('field.header.frame_id', inplace=True, axis=1)
    Temp3.drop('field.header.seq', inplace=True, axis=1)

    Temp4.drop('field.header.stamp', inplace=True, axis=1)
    Temp4.drop('field.header.frame_id', inplace=True, axis=1)
    Temp4.drop('field.header.seq', inplace=True, axis=1)

    Temp5.drop('field.header.stamp', inplace=True, axis=1)
    Temp5.drop('field.header.frame_id', inplace=True, axis=1)
    Temp5.drop('field.header.seq', inplace=True, axis=1)
    Temp5.drop('field.coordinate_frame', inplace=True, axis=1)
    Temp5.drop('field.des_x', inplace=True, axis=1)
    Temp5.drop('field.des_y', inplace=True, axis=1)
    Temp5.drop('field.des_z', inplace=True, axis=1)

    Temp6.drop('field.header.stamp', inplace=True, axis=1)
    Temp6.drop('field.header.frame_id', inplace=True, axis=1)
    Temp6.drop('field.header.seq', inplace=True, axis=1)

    merged = pd.merge_asof(Temp1, Temp2, on='%time', tolerance=100000000)
    merged = pd.merge_asof(merged, Temp3, on='%time', tolerance=100000000)
    merged = pd.merge_asof(merged, Temp4, on='%time', tolerance=100000000)
    merged = pd.merge_asof(merged, Temp5, on='%time', tolerance=100000000)
    merged = pd.merge_asof(merged, Temp6, on='%time', tolerance=100000000)

    merged = pd.merge_asof(Temp2, Temp5, on='%time', tolerance=100000000)
    merged = merged.dropna()

    return merged


def Process_Global_Position(flight_id):
    Temp1 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-global_position-rel_alt.csv")
    Temp2 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-global_position-raw-gps_vel.csv")
    Temp3 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-global_position-raw-fix.csv")
    Temp4 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-global_position-local.csv")
    Temp5 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-global_position-global.csv")
    Temp6 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-global_position-compass_hdg.csv")

    Temp2.drop('field.header.stamp', inplace=True, axis=1)
    Temp2.drop('field.header.frame_id', inplace=True, axis=1)
    Temp2.drop('field.header.seq', inplace=True, axis=1)
    Temp2.drop('field.twist.linear.z', inplace=True, axis=1)
    Temp2.drop('field.twist.angular.x', inplace=True, axis=1)
    Temp2.drop('field.twist.angular.y', inplace=True, axis=1)
    Temp2.drop('field.twist.angular.z', inplace=True, axis=1)

    Temp3.drop('field.header.stamp', inplace=True, axis=1)
    Temp3.drop('field.header.frame_id', inplace=True, axis=1)
    Temp3.drop('field.header.seq', inplace=True, axis=1)
    Temp3.drop('field.status.status', inplace=True, axis=1)
    Temp3.drop('field.status.service', inplace=True, axis=1)
    Temp3.drop('field.position_covariance1', inplace=True, axis=1)
    Temp3.drop('field.position_covariance2', inplace=True, axis=1)
    Temp3.drop('field.position_covariance3', inplace=True, axis=1)
    Temp3.drop('field.position_covariance5', inplace=True, axis=1)
    Temp3.drop('field.position_covariance6', inplace=True, axis=1)
    Temp3.drop('field.position_covariance7', inplace=True, axis=1)
    Temp3.drop('field.position_covariance_type', inplace=True, axis=1)

    Temp4.drop('field.header.stamp', inplace=True, axis=1)
    Temp4.drop('field.header.frame_id', inplace=True, axis=1)
    Temp4.drop('field.child_frame_id', inplace=True, axis=1)
    Temp4.drop('field.header.seq', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance1', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance2', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance3', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance4', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance5', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance6', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance8', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance9', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance10', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance11', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance12', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance13', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance15', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance16', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance17', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance18', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance19', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance20', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance22', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance23', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance24', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance25', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance26', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance27', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance29', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance30', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance31', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance32', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance33', inplace=True, axis=1)
    Temp4.drop('field.pose.covariance34', inplace=True, axis=1)
    Temp4.drop('field.twist.twist.angular.x', inplace=True, axis=1)
    Temp4.drop('field.twist.twist.angular.y', inplace=True, axis=1)
    Temp4.drop('field.twist.twist.angular.z', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance0', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance1', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance2', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance3', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance4', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance5', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance6', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance7', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance8', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance9', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance10', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance11', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance12', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance13', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance14', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance15', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance16', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance17', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance18', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance19', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance20', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance21', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance22', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance23', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance24', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance25', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance26', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance27', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance28', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance29', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance30', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance31', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance32', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance33', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance34', inplace=True, axis=1)
    Temp4.drop('field.twist.covariance35', inplace=True, axis=1)

    Temp5.drop('field.header.stamp', inplace=True, axis=1)
    Temp5.drop('field.header.frame_id', inplace=True, axis=1)
    Temp5.drop('field.header.seq', inplace=True, axis=1)

    merged = pd.merge_asof(Temp1, Temp2, on='%time', tolerance=100000000)
    merged = pd.merge_asof(merged, Temp3, on='%time', tolerance=100000000)
    merged = pd.merge_asof(merged, Temp4, on='%time', tolerance=100000000)
    merged = pd.merge_asof(merged, Temp5, on='%time', tolerance=100000000)
    merged = pd.merge_asof(merged, Temp6, on='%time', tolerance=100000000)

    merged = merged.dropna()

    return merged


def Process_Local_Position(flight_id):
    Temp1 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-local_position-odom.csv")
    Temp2 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-local_position-pose.csv")
    Temp3 = pd.read_csv("processed\\" + flight_id + "\\" + flight_id + "-mavros-local_position-velocity.csv")

    Temp1.drop('field.header.frame_id', inplace=True, axis=1)
    Temp1.drop('field.child_frame_id', inplace=True, axis=1)
    Temp1.drop('field.header.seq', inplace=True, axis=1)
    Temp1.drop('field.header.stamp', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance1', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance2', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance3', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance4', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance5', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance6', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance8', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance9', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance10', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance11', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance12', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance13', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance15', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance16', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance17', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance18', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance19', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance20', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance22', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance23', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance24', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance25', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance26', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance27', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance29', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance30', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance31', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance32', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance33', inplace=True, axis=1)
    Temp1.drop('field.pose.covariance34', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance0', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance1', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance2', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance3', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance4', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance5', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance6', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance7', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance8', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance9', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance10', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance11', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance12', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance13', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance14', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance15', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance16', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance17', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance18', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance19', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance20', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance21', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance22', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance23', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance24', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance25', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance26', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance27', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance28', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance29', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance30', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance31', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance32', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance33', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance34', inplace=True, axis=1)
    Temp1.drop('field.twist.covariance35', inplace=True, axis=1)

    Temp2.drop('field.header.stamp', inplace=True, axis=1)
    Temp2.drop('field.header.frame_id', inplace=True, axis=1)
    Temp2.drop('field.header.seq', inplace=True, axis=1)

    Temp3.drop('field.header.stamp', inplace=True, axis=1)
    Temp3.drop('field.header.frame_id', inplace=True, axis=1)
    Temp3.drop('field.header.seq', inplace=True, axis=1)

    merged = pd.merge_asof(Temp1, Temp2, on='%time', tolerance=100000000)
    merged = pd.merge_asof(merged, Temp3, on='%time', tolerance=100000000)

    merged = merged.dropna()

    return merged


def process_flight(flight_id):
    NAV_data = Process_NAV_Info(flight_id)
    IMU_data = Process_IMU(flight_id)
    # GP_data = Process_Global_Position(flight_id)
    # LP_data = Process_Local_Position(flight_id)
    # Misc_data = Process_Misc(flight_id)

    x_merge = pd.merge_asof(NAV_data, IMU_data, on='%time', tolerance=100000000)
    # x_merge = pd.merge_asof(x_merge,GP_data, on='%time', tolerance =  1000000000  )
    # x_merge = pd.merge_asof(x_merge,LP_data, on='%time', tolerance =  1000000000  )
    # x_merge = pd.merge_asof(x_merge,Misc_data, on='%time', tolerance =  1000000000)

    x_merge = x_merge.dropna()

    n = (10 * 1000000000) + x_merge['%time'].iloc[0]

    data = x_merge.loc[x_merge['%time'] > n]

    data = data.append(x_merge.iloc[0]).dropna()

    return data


def Process_Train_Data():
    train = process_flight(train_flight_id0)
    train = train.append(process_flight(train_flight_id1))
    train = train.append(process_flight(train_flight_id2))
    train = train.append(process_flight(train_flight_id3))
    train = train.append(process_flight(train_flight_id4))
    train = train.append(process_flight(train_flight_id5))
    train = train.append(process_flight(train_flight_id6))
    train = train.append(process_flight(train_flight_id7))
    train = train.append(process_flight(train_flight_id8))
    train = train.append(process_flight(train_flight_id9))

    # x_train = x_train.append(Process_Partial_Flights())

    return train


def Process_Test_Data(fid):
    test_flight_id = fid
    test = process_flight(test_flight_id)
    return test


def Remove_Failure_Data(data, fail_time):
    n = (fail_time * 1000000000) + data['%time'].iloc[0]
    data = data.loc[data['%time'] < n]
    return data


def Process_Partial_Flights():
    pf_data_unmerged = []

    for pf in partial_flights:
        pf_data_unmerged.append(process_flight(pf[0]))

    cut_data = []
    full_data = Remove_Failure_Data(pf_data_unmerged[0], partial_flights[0][1])

    for x in range(1, len(partial_flights)):
        flight_cutoff = Remove_Failure_Data(pf_data_unmerged[x], partial_flights[x][1])

        cut_data.append(flight_cutoff)

    for i in cut_data:
        full_data = full_data.append(i)

    return full_data
