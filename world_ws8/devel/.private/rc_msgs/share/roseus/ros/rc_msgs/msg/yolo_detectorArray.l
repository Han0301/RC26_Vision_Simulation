;; Auto-generated. Do not edit!


(when (boundp 'rc_msgs::yolo_detectorArray)
  (if (not (find-package "RC_MSGS"))
    (make-package "RC_MSGS"))
  (shadow 'yolo_detectorArray (find-package "RC_MSGS")))
(unless (find-package "RC_MSGS::YOLO_DETECTORARRAY")
  (make-package "RC_MSGS::YOLO_DETECTORARRAY"))

(in-package "ROS")
;;//! \htmlinclude yolo_detectorArray.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass rc_msgs::yolo_detectorArray
  :super ros::object
  :slots (_header _boxes ))

(defmethod rc_msgs::yolo_detectorArray
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:boxes __boxes) ())
    )
   (send-super :init)
   (setq _header __header)
   (setq _boxes __boxes)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:boxes
   (&rest __boxes)
   (if (keywordp (car __boxes))
       (send* _boxes __boxes)
     (progn
       (if __boxes (setq _boxes (car __boxes)))
       _boxes)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; rc_msgs/yolo_detector[] _boxes
    (apply #'+ (send-all _boxes :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; rc_msgs/yolo_detector[] _boxes
     (write-long (length _boxes) s)
     (dolist (elem _boxes)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; rc_msgs/yolo_detector[] _boxes
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _boxes (let (r) (dotimes (i n) (push (instance rc_msgs::yolo_detector :init) r)) r))
     (dolist (elem- _boxes)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get rc_msgs::yolo_detectorArray :md5sum-) "276533dc1a5df7655d0543cac92ab85f")
(setf (get rc_msgs::yolo_detectorArray :datatype-) "rc_msgs/yolo_detectorArray")
(setf (get rc_msgs::yolo_detectorArray :definition-)
      "# 带类别信息的2D边界框数组消息
std_msgs/Header header        # 消息头部，包含时间戳和坐标系
yolo_detector[] boxes       # 边界框数组
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: rc_msgs/yolo_detector
# 描述单个2D边界框的消息
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



(provide :rc_msgs/yolo_detectorArray "276533dc1a5df7655d0543cac92ab85f")


