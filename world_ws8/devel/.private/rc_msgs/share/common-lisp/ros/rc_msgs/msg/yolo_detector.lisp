; Auto-generated. Do not edit!


(cl:in-package rc_msgs-msg)


;//! \htmlinclude yolo_detector.msg.html

(cl:defclass <yolo_detector> (roslisp-msg-protocol:ros-message)
  ((center
    :reader center
    :initarg :center
    :type geometry_msgs-msg:Pose2D
    :initform (cl:make-instance 'geometry_msgs-msg:Pose2D))
   (size_x
    :reader size_x
    :initarg :size_x
    :type cl:float
    :initform 0.0)
   (size_y
    :reader size_y
    :initarg :size_y
    :type cl:float
    :initform 0.0)
   (cls
    :reader cls
    :initarg :cls
    :type cl:integer
    :initform 0))
)

(cl:defclass yolo_detector (<yolo_detector>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <yolo_detector>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'yolo_detector)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name rc_msgs-msg:<yolo_detector> is deprecated: use rc_msgs-msg:yolo_detector instead.")))

(cl:ensure-generic-function 'center-val :lambda-list '(m))
(cl:defmethod center-val ((m <yolo_detector>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rc_msgs-msg:center-val is deprecated.  Use rc_msgs-msg:center instead.")
  (center m))

(cl:ensure-generic-function 'size_x-val :lambda-list '(m))
(cl:defmethod size_x-val ((m <yolo_detector>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rc_msgs-msg:size_x-val is deprecated.  Use rc_msgs-msg:size_x instead.")
  (size_x m))

(cl:ensure-generic-function 'size_y-val :lambda-list '(m))
(cl:defmethod size_y-val ((m <yolo_detector>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rc_msgs-msg:size_y-val is deprecated.  Use rc_msgs-msg:size_y instead.")
  (size_y m))

(cl:ensure-generic-function 'cls-val :lambda-list '(m))
(cl:defmethod cls-val ((m <yolo_detector>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader rc_msgs-msg:cls-val is deprecated.  Use rc_msgs-msg:cls instead.")
  (cls m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <yolo_detector>) ostream)
  "Serializes a message object of type '<yolo_detector>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'center) ostream)
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'size_x))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'size_y))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let* ((signed (cl:slot-value msg 'cls)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 4294967296) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <yolo_detector>) istream)
  "Deserializes a message object of type '<yolo_detector>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'center) istream)
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'size_x) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'size_y) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'cls) (cl:if (cl:< unsigned 2147483648) unsigned (cl:- unsigned 4294967296))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<yolo_detector>)))
  "Returns string type for a message object of type '<yolo_detector>"
  "rc_msgs/yolo_detector")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'yolo_detector)))
  "Returns string type for a message object of type 'yolo_detector"
  "rc_msgs/yolo_detector")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<yolo_detector>)))
  "Returns md5sum for a message object of type '<yolo_detector>"
  "eb27bdcd0f972d1bdf70000491495017")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'yolo_detector)))
  "Returns md5sum for a message object of type 'yolo_detector"
  "eb27bdcd0f972d1bdf70000491495017")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<yolo_detector>)))
  "Returns full string definition for message of type '<yolo_detector>"
  (cl:format cl:nil "# 描述单个2D边界框的消息~%geometry_msgs/Pose2D center  # 边界框中心坐标 (x, y)，单位通常为像素或米（取决于坐标系）~%float32 size_x                # x方向尺寸（宽度）~%float32 size_y                # y方向尺寸（高度）~%int32 cls #目标类别~%~%   ~%~%~%================================================================================~%MSG: geometry_msgs/Pose2D~%# Deprecated~%# Please use the full 3D pose.~%~%# In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.~%~%# If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.~%~%~%# This expresses a position and orientation on a 2D manifold.~%~%float64 x~%float64 y~%float64 theta~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'yolo_detector)))
  "Returns full string definition for message of type 'yolo_detector"
  (cl:format cl:nil "# 描述单个2D边界框的消息~%geometry_msgs/Pose2D center  # 边界框中心坐标 (x, y)，单位通常为像素或米（取决于坐标系）~%float32 size_x                # x方向尺寸（宽度）~%float32 size_y                # y方向尺寸（高度）~%int32 cls #目标类别~%~%   ~%~%~%================================================================================~%MSG: geometry_msgs/Pose2D~%# Deprecated~%# Please use the full 3D pose.~%~%# In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.~%~%# If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.~%~%~%# This expresses a position and orientation on a 2D manifold.~%~%float64 x~%float64 y~%float64 theta~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <yolo_detector>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'center))
     4
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <yolo_detector>))
  "Converts a ROS message object to a list"
  (cl:list 'yolo_detector
    (cl:cons ':center (center msg))
    (cl:cons ':size_x (size_x msg))
    (cl:cons ':size_y (size_y msg))
    (cl:cons ':cls (cls msg))
))
