
(cl:in-package :asdf)

(defsystem "rc_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
               :std_msgs-msg
)
  :components ((:file "_package")
    (:file "serial" :depends-on ("_package_serial"))
    (:file "_package_serial" :depends-on ("_package"))
    (:file "yolo_detector" :depends-on ("_package_yolo_detector"))
    (:file "_package_yolo_detector" :depends-on ("_package"))
    (:file "yolo_detectorArray" :depends-on ("_package_yolo_detectorArray"))
    (:file "_package_yolo_detectorArray" :depends-on ("_package"))
  ))