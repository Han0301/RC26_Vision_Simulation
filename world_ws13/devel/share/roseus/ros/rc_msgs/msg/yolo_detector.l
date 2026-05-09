;; Auto-generated. Do not edit!


(when (boundp 'rc_msgs::yolo_detector)
  (if (not (find-package "RC_MSGS"))
    (make-package "RC_MSGS"))
  (shadow 'yolo_detector (find-package "RC_MSGS")))
(unless (find-package "RC_MSGS::YOLO_DETECTOR")
  (make-package "RC_MSGS::YOLO_DETECTOR"))

(in-package "ROS")
;;//! \htmlinclude yolo_detector.msg.html
(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))


(defclass rc_msgs::yolo_detector
  :super ros::object
  :slots (_center _size_x _size_y _cls ))

(defmethod rc_msgs::yolo_detector
  (:init
   (&key
    ((:center __center) (instance geometry_msgs::Pose2D :init))
    ((:size_x __size_x) 0.0)
    ((:size_y __size_y) 0.0)
    ((:cls __cls) 0)
    )
   (send-super :init)
   (setq _center __center)
   (setq _size_x (float __size_x))
   (setq _size_y (float __size_y))
   (setq _cls (round __cls))
   self)
  (:center
   (&rest __center)
   (if (keywordp (car __center))
       (send* _center __center)
     (progn
       (if __center (setq _center (car __center)))
       _center)))
  (:size_x
   (&optional __size_x)
   (if __size_x (setq _size_x __size_x)) _size_x)
  (:size_y
   (&optional __size_y)
   (if __size_y (setq _size_y __size_y)) _size_y)
  (:cls
   (&optional __cls)
   (if __cls (setq _cls __cls)) _cls)
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/Pose2D _center
    (send _center :serialization-length)
    ;; float32 _size_x
    4
    ;; float32 _size_y
    4
    ;; int32 _cls
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/Pose2D _center
       (send _center :serialize s)
     ;; float32 _size_x
       (sys::poke _size_x (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _size_y
       (sys::poke _size_y (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; int32 _cls
       (write-long _cls s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/Pose2D _center
     (send _center :deserialize buf ptr-) (incf ptr- (send _center :serialization-length))
   ;; float32 _size_x
     (setq _size_x (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _size_y
     (setq _size_y (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; int32 _cls
     (setq _cls (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get rc_msgs::yolo_detector :md5sum-) "eb27bdcd0f972d1bdf70000491495017")
(setf (get rc_msgs::yolo_detector :datatype-) "rc_msgs/yolo_detector")
(setf (get rc_msgs::yolo_detector :definition-)
      "# 描述单个2D边界框的消息
geometry_msgs/Pose2D center  # 边界框中心坐标 (x, y)，单位通常为像素或米（取决于坐标系）
float32 size_x                # x方向尺寸（宽度）
float32 size_y                # y方向尺寸（高度）
int32 cls #目标类别

   


================================================================================
MSG: geometry_msgs/Pose2D
# Deprecated
# Please use the full 3D pose.

# In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.

# If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.


# This expresses a position and orientation on a 2D manifold.

float64 x
float64 y
float64 theta

")



(provide :rc_msgs/yolo_detector "eb27bdcd0f972d1bdf70000491495017")


