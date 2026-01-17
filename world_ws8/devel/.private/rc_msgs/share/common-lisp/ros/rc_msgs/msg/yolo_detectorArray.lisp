; Auto-generated. Do not edit!


(cl:in-package rc_msgs-msg)


;//! \htmlinclude yolo_detectorArray.msg.html

(cl:defclass <yolo_detectorArray> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (boxes
    :reader boxes
    :initarg :boxes
    :type (cl:vector rc_msgs-msg:yolo_detector)
   :initform (cl:make-array 0 :element-type 'rc_msgs-msg:yolo_detector :initial-element (cl:make-instance 'rc_msgs-msg:yolo_detector))))
)

(cl:defclass yolo_detectorArray (<yolo_detectorArray>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <yolo_detectorArray>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'yolo_detectorArray)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rc_msgs-msg:<yolo_detectorArray> is deprecated: use rc_msgs-msg:yolo_detectorArray instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <yolo_detectorArray>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rc_msgs-msg:header-val is deprecated.  Use rc_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'boxes-val :lambda-list '(m))
(cl:defmethod boxes-val ((m <yolo_detectorArray>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rc_msgs-msg:boxes-val is deprecated.  Use rc_msgs-msg:boxes instead.")
  (boxes m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <yolo_detectorArray>) ostream)
  "Serializes a message object of type '<yolo_detectorArray>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'boxes))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'boxes))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <yolo_detectorArray>) istream)
  "Deserializes a message object of type '<yolo_detectorArray>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'boxes) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'boxes)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'rc_msgs-msg:yolo_detector))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<yolo_detectorArray>)))
  "Returns string type for a message object of type '<yolo_detectorArray>"
  "rc_msgs/yolo_detectorArray")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'yolo_detectorArray)))
  "Returns string type for a message object of type 'yolo_detectorArray"
  "rc_msgs/yolo_detectorArray")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<yolo_detectorArray>)))
  "Returns md5sum for a message object of type '<yolo_detectorArray>"
  "276533dc1a5df7655d0543cac92ab85f")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'yolo_detectorArray)))
  "Returns md5sum for a message object of type 'yolo_detectorArray"
  "276533dc1a5df7655d0543cac92ab85f")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<yolo_detectorArray>)))
  "Returns full string definition for message of type '<yolo_detectorArray>"
  (cl:format cl:nil "# 带类别信息的2D边界框数组消息~%std_msgs/Header header        # 消息头部，包含时间戳和坐标系~%yolo_detector[] boxes       # 边界框数组~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: rc_msgs/yolo_detector~%# 描述单个2D边界框的消息~%geometry_msgs/Pose2D center  # 边界框中心坐标 (x, y)，单位通常为像素或米（取决于坐标系）~%float32 size_x                # x方向尺寸（宽度）~%float32 size_y                # y方向尺寸（高度）~%int32 cls #目标类别~%~%   ~%~%~%================================================================================~%MSG: geometry_msgs/Pose2D~%# Deprecated~%# Please use the full 3D pose.~%~%# In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.~%~%# If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.~%~%~%# This expresses a position and orientation on a 2D manifold.~%~%float64 x~%float64 y~%float64 theta~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'yolo_detectorArray)))
  "Returns full string definition for message of type 'yolo_detectorArray"
  (cl:format cl:nil "# 带类别信息的2D边界框数组消息~%std_msgs/Header header        # 消息头部，包含时间戳和坐标系~%yolo_detector[] boxes       # 边界框数组~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: rc_msgs/yolo_detector~%# 描述单个2D边界框的消息~%geometry_msgs/Pose2D center  # 边界框中心坐标 (x, y)，单位通常为像素或米（取决于坐标系）~%float32 size_x                # x方向尺寸（宽度）~%float32 size_y                # y方向尺寸（高度）~%int32 cls #目标类别~%~%   ~%~%~%================================================================================~%MSG: geometry_msgs/Pose2D~%# Deprecated~%# Please use the full 3D pose.~%~%# In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.~%~%# If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.~%~%~%# This expresses a position and orientation on a 2D manifold.~%~%float64 x~%float64 y~%float64 theta~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <yolo_detectorArray>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'boxes) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <yolo_detectorArray>))
  "Converts a ROS message object to a list"
  (cl:list 'yolo_detectorArray
    (cl:cons ':header (header msg))
    (cl:cons ':boxes (boxes msg))
))
