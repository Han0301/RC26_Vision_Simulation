// Auto-generated. Do not edit!

// (in-package rc_msgs.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------

class yolo_detector {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.center = null;
      this.size_x = null;
      this.size_y = null;
      this.cls = null;
    }
    else {
      if (initObj.hasOwnProperty('center')) {
        this.center = initObj.center
      }
      else {
        this.center = new geometry_msgs.msg.Pose2D();
      }
      if (initObj.hasOwnProperty('size_x')) {
        this.size_x = initObj.size_x
      }
      else {
        this.size_x = 0.0;
      }
      if (initObj.hasOwnProperty('size_y')) {
        this.size_y = initObj.size_y
      }
      else {
        this.size_y = 0.0;
      }
      if (initObj.hasOwnProperty('cls')) {
        this.cls = initObj.cls
      }
      else {
        this.cls = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type yolo_detector
    // Serialize message field [center]
    bufferOffset = geometry_msgs.msg.Pose2D.serialize(obj.center, buffer, bufferOffset);
    // Serialize message field [size_x]
    bufferOffset = _serializer.float32(obj.size_x, buffer, bufferOffset);
    // Serialize message field [size_y]
    bufferOffset = _serializer.float32(obj.size_y, buffer, bufferOffset);
    // Serialize message field [cls]
    bufferOffset = _serializer.int32(obj.cls, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type yolo_detector
    let len;
    let data = new yolo_detector(null);
    // Deserialize message field [center]
    data.center = geometry_msgs.msg.Pose2D.deserialize(buffer, bufferOffset);
    // Deserialize message field [size_x]
    data.size_x = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [size_y]
    data.size_y = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [cls]
    data.cls = _deserializer.int32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 36;
  }

  static datatype() {
    // Returns string type for a message object
    return 'rc_msgs/yolo_detector';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'eb27bdcd0f972d1bdf70000491495017';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
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
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new yolo_detector(null);
    if (msg.center !== undefined) {
      resolved.center = geometry_msgs.msg.Pose2D.Resolve(msg.center)
    }
    else {
      resolved.center = new geometry_msgs.msg.Pose2D()
    }

    if (msg.size_x !== undefined) {
      resolved.size_x = msg.size_x;
    }
    else {
      resolved.size_x = 0.0
    }

    if (msg.size_y !== undefined) {
      resolved.size_y = msg.size_y;
    }
    else {
      resolved.size_y = 0.0
    }

    if (msg.cls !== undefined) {
      resolved.cls = msg.cls;
    }
    else {
      resolved.cls = 0
    }

    return resolved;
    }
};

module.exports = yolo_detector;
