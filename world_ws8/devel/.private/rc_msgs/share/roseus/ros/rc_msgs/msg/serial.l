;; Auto-generated. Do not edit!


(when (boundp 'rc_msgs::serial)
  (if (not (find-package "RC_MSGS"))
    (make-package "RC_MSGS"))
  (shadow 'serial (find-package "RC_MSGS")))
(unless (find-package "RC_MSGS::SERIAL")
  (make-package "RC_MSGS::SERIAL"))

(in-package "ROS")
;;//! \htmlinclude serial.msg.html


(defclass rc_msgs::serial
  :super ros::object
  :slots (_stamp _data ))

(defmethod rc_msgs::serial
  (:init
   (&key
    ((:stamp __stamp) (instance ros::time :init))
    ((:data __data) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _stamp __stamp)
   (setq _data __data)
   self)
  (:stamp
   (&optional __stamp)
   (if __stamp (setq _stamp __stamp)) _stamp)
  (:data
   (&optional __data)
   (if __data (setq _data __data)) _data)
  (:serialization-length
   ()
   (+
    ;; time _stamp
    8
    ;; float32[] _data
    (* 4    (length _data)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; time _stamp
       (write-long (send _stamp :sec) s) (write-long (send _stamp :nsec) s)
     ;; float32[] _data
     (write-long (length _data) s)
     (dotimes (i (length _data))
       (sys::poke (elt _data i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; time _stamp
     (send _stamp :sec (sys::peek buf ptr- :integer)) (incf ptr- 4)  (send _stamp :nsec (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; float32[] _data
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _data (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _data i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(setf (get rc_msgs::serial :md5sum-) "2c2996107ba4a77d3150497cc03c74e2")
(setf (get rc_msgs::serial :datatype-) "rc_msgs/serial")
(setf (get rc_msgs::serial :definition-)
      "# 包含时间戳
time stamp

# 包含float类型的数组
float32[] data

")



(provide :rc_msgs/serial "2c2996107ba4a77d3150497cc03c74e2")


